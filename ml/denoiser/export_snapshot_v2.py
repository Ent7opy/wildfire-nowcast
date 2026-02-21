"""Event-level snapshot export for denoiser v2."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from api.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("export_snapshot_v2")


def _derive_event_id(df: pd.DataFrame) -> pd.Series:
    # Fall back to a deterministic singleton event id if upstream eventization has not run.
    return df["event_id"].fillna(df["fire_detection_id"].map(lambda x: f"det_{int(x)}"))


def _mode_or_unknown(series: pd.Series) -> str:
    if series.dropna().empty:
        return "unknown"
    mode = series.dropna().mode()
    return str(mode.iloc[0]) if not mode.empty else "unknown"


def export_training_snapshot_v2(
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    *,
    rule_version: str,
    out_dir: str,
    run_id: Optional[str] = None,
) -> str:
    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(out_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    min_lon, min_lat, max_lon, max_lat = aoi_bbox

    query = text(
        """
        SELECT
            l.fire_detection_id,
            COALESCE(l.event_id, d.event_id) AS event_id,
            l.label,
            l.weak_supervision,
            d.acq_time,
            d.sensor,
            d.source,
            d.confidence,
            d.frp,
            d.brightness,
            d.bright_t31,
            d.scan,
            d.track,
            d.landcover_score,
            d.lat,
            d.lon,
            d.raw_properties
        FROM denoiser_labels_v2 l
        JOIN fire_detections d ON d.id = l.fire_detection_id
        WHERE l.rule_version = :rule_version
          AND d.acq_time BETWEEN :start_time AND :end_time
          AND d.geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
        """
    )

    with get_engine().begin() as conn:
        rows = pd.read_sql(
            query,
            conn,
            params={
                "rule_version": rule_version,
                "start_time": start_time,
                "end_time": end_time,
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
            },
        )

    if rows.empty:
        raise SystemExit("No labeled v2 rows found for export.")

    rows["acq_time"] = pd.to_datetime(rows["acq_time"], utc=True)
    rows["event_id"] = _derive_event_id(rows)
    rows["is_day"] = rows["raw_properties"].apply(
        lambda x: 1 if isinstance(x, dict) and x.get("daynight") == "D" else 0
    )

    event_df = (
        rows.groupby("event_id", dropna=False)
        .agg(
            start_time=("acq_time", "min"),
            end_time=("acq_time", "max"),
            detection_count=("fire_detection_id", "count"),
            positive_count=("label", lambda s: int((s == "POSITIVE").sum())),
            negative_count=("label", lambda s: int((s == "NEGATIVE").sum())),
            probable_positive_count=("label", lambda s: int((s == "PROBABLE_POSITIVE").sum())),
            weak_supervision_count=("weak_supervision", lambda s: int(pd.Series(s).fillna(False).sum())),
            sensor=("sensor", _mode_or_unknown),
            source=("source", _mode_or_unknown),
            confidence_mean=("confidence", "mean"),
            confidence_max=("confidence", "max"),
            frp_mean=("frp", "mean"),
            frp_max=("frp", "max"),
            brightness_mean=("brightness", "mean"),
            bright_t31_mean=("bright_t31", "mean"),
            scan_mean=("scan", "mean"),
            track_mean=("track", "mean"),
            landcover_mean=("landcover_score", "mean"),
            is_day_ratio=("is_day", "mean"),
            lat_centroid=("lat", "mean"),
            lon_centroid=("lon", "mean"),
        )
        .reset_index()
    )

    event_df["duration_hours"] = (
        (event_df["end_time"] - event_df["start_time"]).dt.total_seconds() / 3600.0
    )
    doy = event_df["start_time"].dt.dayofyear.fillna(1)
    event_df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    event_df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # Perimeter-covered-first proxy slice.
    event_df["biome_slice"] = pd.cut(
        event_df["landcover_mean"].fillna(0.5),
        bins=[-np.inf, 0.25, 0.6, np.inf],
        labels=["low_fuel", "mixed_fuel", "high_fuel"],
    ).astype(str)

    event_df["event_label"] = "UNKNOWN"
    event_df.loc[event_df["positive_count"] > 0, "event_label"] = "POSITIVE"
    event_df.loc[(event_df["positive_count"] == 0) & (event_df["negative_count"] > 0), "event_label"] = "NEGATIVE"

    event_df["label_numeric"] = event_df["event_label"].map({"NEGATIVE": 0, "POSITIVE": 1}).astype("float")

    event_df = event_df.sort_values("start_time").reset_index(drop=True)
    split_dt = event_df["start_time"].quantile(0.8)
    train_df = event_df[event_df["start_time"] < split_dt].copy()
    eval_df = event_df[event_df["start_time"] >= split_dt].copy()

    train_path = os.path.join(run_dir, "train.parquet")
    eval_path = os.path.join(run_dir, "eval.parquet")
    full_path = os.path.join(run_dir, "full.parquet")

    train_df.to_parquet(train_path, index=False)
    eval_df.to_parquet(eval_path, index=False)
    event_df.to_parquet(full_path, index=False)

    metadata = {
        "run_id": run_id,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "rule_version": rule_version,
        "aoi_bbox": list(aoi_bbox),
        "time_range": [start_time.isoformat(), end_time.isoformat()],
        "counts": {
            "event_total": int(len(event_df)),
            "event_train": int(len(train_df)),
            "event_eval": int(len(eval_df)),
            "event_positive": int((event_df["event_label"] == "POSITIVE").sum()),
            "event_negative": int((event_df["event_label"] == "NEGATIVE").sum()),
            "event_unknown": int((event_df["event_label"] == "UNKNOWN").sum()),
        },
        "split": {
            "strategy": "time_percentile",
            "percentile": 0.8,
            "split_dt": split_dt.isoformat() if pd.notna(split_dt) else None,
        },
        "features": [
            c
            for c in event_df.columns
            if c
            not in {
                "event_id",
                "event_label",
                "label_numeric",
                "start_time",
                "end_time",
            }
        ],
        "paths": {
            "train": train_path,
            "eval": eval_path,
            "full": full_path,
        },
    }

    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export event-level denoiser v2 snapshot.")
    parser.add_argument("--bbox", type=float, nargs=4, required=True, help="min_lon min_lat max_lon max_lat")
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--version", type=str, default="v2_default")
    parser.add_argument("--out", type=str, default="data/denoiser/snapshots_v2")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    run_dir = export_training_snapshot_v2(
        tuple(args.bbox),
        start,
        end,
        rule_version=args.version,
        out_dir=args.out,
    )
    LOGGER.info("Exported v2 snapshot: %s", run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
