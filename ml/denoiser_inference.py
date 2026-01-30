"""ML inference CLI for FIRMS hotspot denoiser."""

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional
from pathlib import Path

# Add project root to sys.path to allow importing from other subprojects
sys.path.append(str(Path(__file__).resolve().parents[1]))

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
from ingest.config import settings as ingest_settings

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

def get_pending_detections(engine: Engine, batch_id: Optional[int] = None) -> pd.DataFrame:
    """Fetch detections that need denoising."""
    if batch_id is not None:
        query = text("""
            SELECT 
                id, lat, lon, acq_time, confidence, frp, 
                brightness, bright_t31, scan, track, sensor, source,
                raw_properties
            FROM fire_detections
            WHERE ingest_batch_id = :batch_id
              AND denoised_score IS NULL
        """)
        params = {"batch_id": batch_id}
    else:
        query = text("""
            SELECT 
                id, lat, lon, acq_time, confidence, frp, 
                brightness, bright_t31, scan, track, sensor, source,
                raw_properties
            FROM fire_detections
            WHERE denoised_score IS NULL
        """)
        params = {}

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    return df

def build_features(
    df: pd.DataFrame, 
    engine: Engine, 
    feature_list: list[str],
    region_name: Optional[str] = None,
    allow_missing_features: bool = True,
) -> pd.DataFrame:
    """Build all required features for inference.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input detection data.
    engine : Engine
        SQLAlchemy database engine.
    feature_list : list[str]
        List of required feature columns.
    region_name : Optional[str]
        Region name for terrain features.
    allow_missing_features : bool
        If True (default for backwards compatibility), fills missing features with NaN and logs a warning.
        If False, raises an error if any required features are missing.
        Use allow_missing_features=False in production to ensure model integrity.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features aligned to feature_list.
        
    Raises
    ------
    ValueError
        If required features are missing and allow_missing_features is False.
    """
    if df.empty:
        return pd.DataFrame(columns=feature_list)

    # 1. Apply feature pipeline
    df = add_firms_features(df)
    df = add_time_features(df)
    df = add_spatiotemporal_context_batch(df, engine)
    
    if region_name:
        df = add_terrain_features(df, region_name)

    # 2. Align with training feature list
    missing_features = [col for col in feature_list if col not in df.columns]
    
    if missing_features:
        if allow_missing_features:
            for col in missing_features:
                LOGGER.warning(f"Feature column {col} missing during inference. Filling with NaN.")
                df[col] = np.nan
        else:
            error_msg = (
                f"Missing {len(missing_features)} required feature(s) during inference: {missing_features}. "
                "This may indicate that the feature pipeline is not producing expected outputs. "
                "Set allow_missing_features=True to fill with NaN (if your model supports it)."
            )
            LOGGER.error(error_msg)
            raise ValueError(error_msg)
    
    # Check for NaN values in required features (including newly filled ones)
    features_with_nan = [
        col for col in feature_list 
        if col in df.columns and df[col].isna().any()
    ]
    if features_with_nan:
        LOGGER.warning(
            f"Features with NaN values detected: {features_with_nan}. "
            "Model predictions may be affected."
        )

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
    batch_id: Optional[int],
    model_run_dir: str,
    threshold: float,
    batch_size: int = 500,
    region_name: Optional[str] = None,
):
    """Run denoiser inference on a batch of detections (or all pending)."""
    engine = get_engine()
    
    # 1. Load artifacts
    model, feature_list = load_model_artifacts(model_run_dir)
    
    # 2. Get pending detections
    df_all = get_pending_detections(engine, batch_id)
    if df_all.empty:
        msg = f"No pending detections found for batch {batch_id}." if batch_id else "No pending detections found."
        LOGGER.info(msg)
        print(json.dumps({"batch_id": batch_id, "count": 0}))
        return

    LOGGER.info(f"Found {len(df_all)} detections to score.")
    
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
        batch_id if batch_id else "ALL",
        summary["noise_percent"],
    )
    # Print JSON summary to stdout as requested by plan
    print(json.dumps(summary))

def main():
    parser = argparse.ArgumentParser(description="Run denoiser inference on ingested detections.")
    parser.add_argument("--batch-id", type=int, help="Ingest batch ID to process (if omitted, processes all pending)")
    parser.add_argument(
        "--model-run", 
        type=str, 
        default=ingest_settings.denoiser_model_run_dir,
        help="Path to model run directory (defaults to DENOISER_MODEL_RUN_DIR env)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=ingest_settings.denoiser_threshold, 
        help="Score threshold for is_noise=True"
    )
    parser.add_argument("--batch-size", type=int, default=ingest_settings.denoiser_batch_size, help="Inference chunk size")
    parser.add_argument(
        "--region", 
        type=str, 
        default=ingest_settings.denoiser_region,
        help="Region name for terrain features"
    )
    
    args = parser.parse_args()
    
    if not args.model_run:
        LOGGER.error("No model run directory provided and DENOISER_MODEL_RUN_DIR is not set.")
        sys.exit(1)
    
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

