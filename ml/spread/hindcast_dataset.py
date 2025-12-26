"""Hindcast dataset builder for learned spread models."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from api.db import get_engine
from api.fires.service import get_fire_cells_heatmap
from ml.spread_features import build_spread_inputs

LOGGER = logging.getLogger(__name__)


def sample_fire_reference_times(
    engine: Engine,
    bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    min_detections: int = 5,
    interval_hours: int = 24,
) -> List[datetime]:
    """Sample reference times that have active fires in the bbox.

    This ensures we build the dataset around actual fire events.
    We group detections by `interval_hours` time buckets and pick the start of those buckets
    if they have enough detections.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Note: we use date_trunc to group by time buckets.
    # We want to find candidate reference times (start of a prediction window).
    stmt = sa.text(
        """
        SELECT 
            date_trunc('hour', acq_time) - (CAST(extract(hour FROM acq_time) AS INTEGER) % :interval_h) * interval '1 hour' as ref_time,
            count(*) as detection_count
        FROM fire_detections
        WHERE acq_time BETWEEN :start AND :end
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
          AND ST_Intersects(geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))
        GROUP BY 1
        HAVING count(*) >= :min_det
        ORDER BY 1 ASC
        """
    )

    with engine.connect() as conn:
        result = conn.execute(
            stmt,
            {
                "start": start_time,
                "end": end_time,
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "min_det": min_detections,
                "interval_h": interval_hours,
            },
        ).mappings().all()

    # Convert to UTC-aware datetimes
    return [r["ref_time"].replace(tzinfo=timezone.utc) for r in result]


def _flatten_features(
    region_name: str,
    bbox: Tuple[float, float, float, float],
    ref_time: datetime,
    horizons_hours: List[int],
) -> List[pd.DataFrame]:
    """Extract features for a single reference time across all horizons.
    
    Returns a list of DataFrames, one per horizon.
    """
    # 1. Gather all inputs (weather, terrain, current fires)
    inputs = build_spread_inputs(
        region_name=region_name,
        bbox=bbox,
        forecast_reference_time=ref_time,
        horizons_hours=horizons_hours,
    )
    
    # 2. Extract static terrain features
    # (lat, lon) arrays from TerrainWindow
    slope = inputs.terrain.slope
    aspect = inputs.terrain.aspect
    elevation = inputs.terrain.elevation
    
    # Pre-calculate aspect components to avoid circularity issues
    aspect_rad = np.radians(aspect)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)
    
    # 3. Extract current fire state (T=0)
    fire_t0 = inputs.active_fires.heatmap
    
    # 4. Loop horizons to build per-horizon tabular data
    horizon_dfs = []
    for h_idx, horizon_h in enumerate(horizons_hours):
        # 4a. Weather at this horizon
        # weather_cube has (time, lat, lon)
        weather_h = inputs.weather_cube.isel(time=h_idx)
        u10 = weather_h["u10"].values
        v10 = weather_h["v10"].values
        t2m = weather_h.get("t2m")
        rh2m = weather_h.get("rh2m")
        
        # 4b. Load target label (future fire presence)
        # We look ahead from ref_time + horizon_h
        target_time = ref_time + timedelta(hours=horizon_h)
        # Look in a small window around the target time (e.g. +/- 3h) to capture detections
        # that might be slightly offset but represent fire at that time.
        target_start = target_time - timedelta(hours=3)
        target_end = target_time + timedelta(hours=3)
        
        target_heatmap = get_fire_cells_heatmap(
            region_name=region_name,
            bbox=bbox,
            start_time=target_start,
            end_time=target_end,
            mode="presence",
            clip=True,
        ).heatmap
        if target_heatmap.shape != fire_t0.shape:
            raise ValueError(
                "Target heatmap shape mismatch. "
                f"target={target_heatmap.shape} fire_t0={fire_t0.shape} "
                f"region={region_name!r} bbox={bbox!r} ref_time={ref_time!r} horizon_h={horizon_h}"
            )
        
        # 5. Flatten everything into a DataFrame for this (ref_time, horizon)
        ny, nx = fire_t0.shape
        # Create coordinate grids
        lat_grid, lon_grid = np.meshgrid(inputs.window.lat, inputs.window.lon, indexing="ij")
        
        data: Dict[str, Any] = {
            "ref_time": [ref_time] * (ny * nx),
            "horizon_h": [horizon_h] * (ny * nx),
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
            "fire_t0": fire_t0.ravel(),
            "slope_deg": slope.ravel(),
            "aspect_sin": aspect_sin.ravel(),
            "aspect_cos": aspect_cos.ravel(),
            "u10": u10.ravel(),
            "v10": v10.ravel(),
            "label": target_heatmap.ravel().astype(int),
        }
        
        if elevation is not None:
            data["elevation_m"] = elevation.ravel()
        if t2m is not None:
            data["t2m"] = t2m.values.ravel()
        if rh2m is not None:
            data["rh2m"] = rh2m.values.ravel()
            
        df = pd.DataFrame(data)
        
        # Add basic derived features
        df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        
        horizon_dfs.append(df)
        
    return horizon_dfs


def build_hindcast_dataset(
    region_name: str,
    bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    horizons_hours: List[int],
    min_detections: int = 5,
    interval_hours: int = 24,
    negative_ratio: Optional[float] = 5.0,
    min_negative_samples: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a consolidated hindcast dataset for spread training."""
    engine = get_engine()
    
    # 1. Sample reference times
    candidate_times = sample_fire_reference_times(
        engine, bbox, start_time, end_time, min_detections, interval_hours
    )
    LOGGER.info(f"Found {len(candidate_times)} candidate reference times for hindcast.")
    
    all_dfs = []
    for ref_time in candidate_times:
        try:
            LOGGER.info(f"Processing ref_time={ref_time}...")
            horizon_dfs = _flatten_features(region_name, bbox, ref_time, horizons_hours)
            
            for df in horizon_dfs:
                # 2. Negative sampling (if requested)
                # Spread is very sparse; most cells are 0.
                if negative_ratio is not None:
                    pos_mask = df["label"] == 1
                    neg_mask = df["label"] == 0
                    
                    n_pos = int(pos_mask.sum())
                    # Always include cells that had fire at T=0 (even if there are no future positives).
                    must_keep_neg = neg_mask & (df["fire_t0"] > 0)
                    n_must_keep = int(must_keep_neg.sum())

                    # Target number of negatives:
                    # - If we have positives, use the requested negative_ratio.
                    # - If we have zero positives, still include a small background sample so the
                    #   dataset isn't biased toward "only horizons with spread".
                    if n_pos > 0:
                        n_neg_target = int(n_pos * float(negative_ratio))
                    else:
                        n_neg_target = int(max(min_negative_samples, n_must_keep))
                    
                    other_neg = neg_mask & (~must_keep_neg)
                    n_other_target = max(0, n_neg_target - n_must_keep)
                    
                    if n_other_target < other_neg.sum():
                        # Make sampling stable but vary per (ref_time, horizon) so we don't select
                        # the exact same background negatives for every time bucket.
                        horizon_val = int(df["horizon_h"].iloc[0]) if "horizon_h" in df.columns and len(df) else 0
                        rs = int(seed + int(ref_time.timestamp()) + horizon_val * 1000) & 0x7FFFFFFF
                        sampled_other_neg_idx = df[other_neg].sample(
                            n=n_other_target, random_state=rs
                        ).index
                        keep_mask = pos_mask | must_keep_neg
                        final_df = pd.concat([df[keep_mask], df.loc[sampled_other_neg_idx]])
                    else:
                        final_df = df[pos_mask | neg_mask]
                    
                    all_dfs.append(final_df)
                else:
                    all_dfs.append(df)
                    
        except Exception:
            LOGGER.exception(f"Failed to process ref_time={ref_time}; skipping.")
            
    if not all_dfs:
        return pd.DataFrame()
        
    return pd.concat(all_dfs, ignore_index=True)

