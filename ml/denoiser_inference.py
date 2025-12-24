"""ML inference CLI for FIRMS hotspot denoiser."""

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.db import get_engine
from ml.denoiser.features import (
    add_firms_features,
    add_time_features,
    add_spatiotemporal_context_batch,
    add_terrain_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("denoiser_inference")

def load_model_artifacts(model_run_dir: str) -> tuple[Any, list[str]]:
    """Load model and feature list from a run directory."""
    model_path = os.path.join(model_run_dir, "model.pkl")
    features_path = os.path.join(model_run_dir, "feature_list.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature list file not found: {features_path}")

    LOGGER.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    with open(features_path, "r", encoding="utf-8") as f:
        feature_list = json.load(f)

    return model, feature_list

def get_pending_detections(engine: Engine, batch_id: int) -> pd.DataFrame:
    """Fetch detections from a batch that need denoising."""
    query = text("""
        SELECT 
            id, lat, lon, acq_time, confidence, frp, 
            brightness, bright_t31, scan, track, sensor, source,
            raw_properties
        FROM fire_detections
        WHERE ingest_batch_id = :batch_id
          AND denoised_score IS NULL
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"batch_id": batch_id})
    return df

def build_features(
    df: pd.DataFrame, 
    engine: Engine, 
    feature_list: list[str],
    region_name: Optional[str] = None
) -> pd.DataFrame:
    """Build all required features for inference."""
    if df.empty:
        return pd.DataFrame(columns=feature_list)

    # 1. Apply feature pipeline
    df = add_firms_features(df)
    df = add_time_features(df)
    df = add_spatiotemporal_context_batch(df, engine)
    
    if region_name:
        df = add_terrain_features(df, region_name)

    # 2. Align with training feature list
    for col in feature_list:
        if col not in df.columns:
            LOGGER.warning(f"Feature column {col} missing during inference. Filling with NaN.")
            df[col] = np.nan

    return df[feature_list]

def update_detections(
    engine: Engine, 
    ids: list[int], 
    scores: list[float], 
    is_noise: list[bool]
):
    """Update denoised_score and is_noise in the DB."""
    if not ids:
        return

    update_stmt = text("""
        UPDATE fire_detections
        SET 
            denoised_score = :score,
            is_noise = :is_noise
        WHERE id = :id
    """)

    params = [
        {"id": int(i), "score": float(s), "is_noise": bool(n)}
        for i, s, n in zip(ids, scores, is_noise)
    ]

    with engine.begin() as conn:
        conn.execute(update_stmt, params)

def run_inference(
    batch_id: int,
    model_run_dir: str,
    threshold: float,
    batch_size: int = 500,
    region_name: Optional[str] = None,
):
    """Run denoiser inference on a batch of detections."""
    engine = get_engine()
    
    # 1. Load artifacts
    model, feature_list = load_model_artifacts(model_run_dir)
    
    # 2. Get pending detections
    df_all = get_pending_detections(engine, batch_id)
    if df_all.empty:
        LOGGER.info(f"No pending detections found for batch {batch_id}.")
        print(json.dumps({"batch_id": batch_id, "count": 0}))
        return

    LOGGER.info(f"Found {len(df_all)} detections to score in batch {batch_id}.")
    
    all_scores = []
    all_is_noise = []
    
    # 3. Process in chunks
    for i in range(0, len(df_all), batch_size):
        chunk = df_all.iloc[i : i + batch_size].copy()
        LOGGER.info(
            "Processing chunk %s/%s (size %s)",
            i // batch_size + 1,
            (len(df_all) - 1) // batch_size + 1,
            len(chunk),
        )
        
        X = build_features(chunk, engine, feature_list, region_name=region_name)
        
        # Predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            # Fallback for models without predict_proba if any
            probs = model.predict(X)
            
        scores = probs.tolist()
        noise_flags = [bool(p < threshold) for p in scores]
        
        # Update DB
        update_detections(engine, chunk["id"].tolist(), scores, noise_flags)
        
        all_scores.extend(scores)
        all_is_noise.extend(noise_flags)

    # 4. Summary stats
    summary = {
        "batch_id": batch_id,
        "count": len(all_scores),
        "mean_score": float(np.mean(all_scores)),
        "median_score": float(np.median(all_scores)),
        "min_score": float(np.min(all_scores)),
        "max_score": float(np.max(all_scores)),
        "noise_count": sum(all_is_noise),
        "noise_percent": float(sum(all_is_noise) / len(all_is_noise) * 100) if all_scores else 0,
        "threshold": threshold,
    }
    
    LOGGER.info(
        "Inference complete for batch %s. Noise %%: %.2f%%",
        batch_id,
        summary["noise_percent"],
    )
    # Print JSON summary to stdout as requested by plan
    print(json.dumps(summary))

def main():
    parser = argparse.ArgumentParser(description="Run denoiser inference on ingested detections.")
    parser.add_argument("--batch-id", type=int, required=True, help="Ingest batch ID to process")
    parser.add_argument("--model-run", type=str, required=True, help="Path to model run directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for is_noise=True")
    parser.add_argument("--batch-size", type=int, default=500, help="Inference chunk size")
    parser.add_argument("--region", type=str, help="Region name for terrain features")
    
    args = parser.parse_args()
    
    try:
        run_inference(
            batch_id=args.batch_id,
            model_run_dir=args.model_run,
            threshold=args.threshold,
            batch_size=args.batch_size,
            region_name=args.region,
        )
    except Exception:
        LOGGER.exception(f"Inference failed for batch {args.batch_id}")
        sys.exit(1)

if __name__ == "__main__":
    main()

