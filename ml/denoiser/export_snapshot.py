"""Immutable training snapshot exporter for denoiser v1."""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from sqlalchemy import text
from api.db import get_engine
from .dataset import build_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("export_snapshot")

def export_training_snapshot(
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    rule_version: str,
    out_dir: str,
    region_name: Optional[str] = None,
    run_id: Optional[str] = None
):
    """Export training and evaluation snapshots to Parquet."""
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = os.path.join(out_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    engine = get_engine()
    min_lon, min_lat, max_lon, max_lat = aoi_bbox

    # 1. Load detections + labels
    LOGGER.info(f"Loading labeled detections for {rule_version}...")
    query = text("""
        SELECT 
            d.id, d.lat, d.lon, d.acq_time, d.confidence, d.frp, 
            d.brightness, d.bright_t31, d.scan, d.track, d.sensor, d.source,
            d.raw_properties,
            l.label
        FROM fire_detections d
        JOIN fire_labels l ON d.id = l.fire_detection_id
        WHERE l.rule_version = :version
          AND d.acq_time BETWEEN :start AND :end
          AND d.geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            "version": rule_version,
            "start": start_time,
            "end": end_time,
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat
        })

    if df.empty:
        LOGGER.error("No labeled detections found for export.")
        return

    LOGGER.info(f"Loaded {len(df)} labeled detections. Building features...")

    # 2. Build features (X, y, meta)
    # Note: build_dataset handles add_firms_features, add_time_features, add_spatiotemporal_context, etc.
    X, y, meta = build_dataset(df, engine, region_name=region_name, include_terrain=(region_name is not None))

    if X.empty:
        LOGGER.error("No valid training samples (POSITIVE/NEGATIVE) found after feature building.")
        return

    # Combine into a single dataframe for export
    full_df = pd.concat([X, meta], axis=1)
    full_df["label_numeric"] = y

    # 3. Create temporal split.
    # Default: percentile-based time split. This is robust for short windows (e.g., <= 1 month),
    # where a "last month" split would otherwise yield an empty train set.
    full_df["acq_time"] = pd.to_datetime(full_df["acq_time"])
    min_time = full_df["acq_time"].min()
    max_time = full_df["acq_time"].max()
    split_percentile = 0.8
    split_date = full_df["acq_time"].quantile(split_percentile)

    # Safety clamp: ensure split_date is inside the observed range
    if split_date <= min_time or split_date >= max_time:
        split_date = full_df["acq_time"].quantile(0.7)
    
    train_df = full_df[full_df["acq_time"] < split_date].copy()
    eval_df = full_df[full_df["acq_time"] >= split_date].copy()
    
    LOGGER.info(f"Split data: {len(train_df)} train, {len(eval_df)} eval (Split date: {split_date})")

    # 4. Save Parquets
    train_path = os.path.join(run_dir, "train.parquet")
    eval_path = os.path.join(run_dir, "eval.parquet")
    
    train_df.to_parquet(train_path, index=False)
    eval_df.to_parquet(eval_path, index=False)
    
    # 5. Save Metadata JSON
    metadata = {
        "run_id": run_id,
        "exported_at": datetime.now().isoformat(),
        "aoi_bbox": aoi_bbox,
        "time_range": [start_time.isoformat(), end_time.isoformat()],
        "rule_version": rule_version,
        "region_name": region_name,
        "counts": {
            "total_labeled": len(df),
            "train": len(train_df),
            "eval": len(eval_df),
            "positive": int((full_df["label"] == "POSITIVE").sum()),
            "negative": int((full_df["label"] == "NEGATIVE").sum()),
        },
        "split": {
            "strategy": "time_percentile",
            "percentile": split_percentile,
            "split_date": split_date.isoformat(),
            "min_time": min_time.isoformat(),
            "max_time": max_time.isoformat(),
        },
        "features": list(X.columns),
        "paths": {
            "train": train_path,
            "eval": eval_path
        }
    }
    
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    LOGGER.info(f"Export complete. Snapshot saved to {run_dir}")

def main():
    parser = argparse.ArgumentParser(description="Export training snapshot for denoiser.")
    parser.add_argument("--bbox", type=float, nargs=4, required=True, help="min_lon min_lat max_lon max_lat")
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--version", type=str, required=True, help="Rule version to export")
    parser.add_argument("--out", type=str, default="data/denoiser/snapshots", help="Output directory")
    parser.add_argument("--aoi", type=str, help="Region name for terrain features")
    
    args = parser.parse_args()
    
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    
    export_training_snapshot(
        tuple(args.bbox), start, end, 
        rule_version=args.version, 
        out_dir=args.out, 
        region_name=args.aoi
    )

if __name__ == "__main__":
    main()

