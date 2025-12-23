"""Dataset builder for FIRMS hotspot denoiser."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sqlalchemy import text, Engine
from .features import add_firms_features, add_time_features, add_spatiotemporal_context, add_terrain_features

def load_labeled_data(
    engine: Engine,
    start_date: str,
    end_date: str,
    region_name: Optional[str] = None,
    label_table: str = "fire_labels"
) -> pd.DataFrame:
    """Load detections joined with labels for a given time window."""
    
    # We assume the labels table has fire_detection_id and label (POS/NEG/UNKNOWN)
    query = text(f"""
        SELECT 
            d.id, d.lat, d.lon, d.acq_time, d.confidence, d.frp, 
            d.brightness, d.bright_t31, d.scan, d.track, d.sensor, d.source,
            d.raw_properties,
            l.label
        FROM fire_detections d
        JOIN {label_table} l ON d.id = l.fire_detection_id
        WHERE d.acq_time BETWEEN :start AND :end
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"start": start_date, "end": end_date})
    
    return df

def build_dataset(
    df: pd.DataFrame,
    engine: Engine,
    region_name: Optional[str] = None,
    include_terrain: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build features and return X, y, and meta.
    Exclude UNKNOWN labels from training output.
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    # 1. Add features
    df = add_firms_features(df)
    df = add_time_features(df)
    df = add_spatiotemporal_context(df, engine)
    
    if include_terrain and region_name:
        df = add_terrain_features(df, region_name)

    # 2. Filter for training (POSITIVE/NEGATIVE only)
    train_df = df[df["label"].isin(["POSITIVE", "NEGATIVE"])].copy()
    
    # 3. Split into X, y, meta
    # Define feature columns
    feature_cols = [
        "confidence_norm", "frp", "brightness", "bright_t31", "brightness_minus_t31",
        "scan", "track", "sin_hour", "cos_hour", "sin_doy", "cos_doy",
        "n_2km_6h", "n_2km_24h", "n_5km_24h", "n_same_cell_24h", "n_same_cell_7d",
        "dist_nn_24h_km", "seen_same_cell_past_3d", "days_with_detection_past_30d_in_cell"
    ]
    
    if "elevation_m" in train_df.columns:
        feature_cols.extend(["elevation_m", "slope_deg", "aspect_sin", "aspect_cos"])
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in train_df.columns:
            train_df[col] = np.nan

    X = train_df[feature_cols]
    y = train_df["label"].map({"POSITIVE": 1, "NEGATIVE": 0})
    
    meta_cols = ["id", "lat", "lon", "acq_time", "sensor", "source", "label"]
    meta = train_df[meta_cols]
    
    return X, y, meta
