"""Hindcast dataset builder for learned spread models."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sa
from scipy.ndimage import binary_dilation
from sqlalchemy.engine import Engine

from api.db import get_engine
from api.fires.service import get_fire_cells_heatmap
from ml.spread.region_key import deterministic_region_bucket
from ml.spread.runtime_contract import CANONICAL_V2_CHANNELS, CANONICAL_V3_CHANNELS
from ml.spread_features import build_spread_inputs

LOGGER = logging.getLogger(__name__)

# Channel order is owned by runtime_contract.py — do not redefine locally.
V2_TENSOR_CHANNELS: tuple[str, ...] = CANONICAL_V2_CHANNELS
V3_TENSOR_CHANNELS: tuple[str, ...] = CANONICAL_V3_CHANNELS


def sample_fire_reference_times(
    engine: Engine,
    bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    min_detections: int = 5,
    interval_hours: int = 24,
) -> List[datetime]:
    """Sample reference times that have active fires in the bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox

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

    return [r["ref_time"].replace(tzinfo=timezone.utc) for r in result]


def _load_fire_history(
    region_name: str,
    bbox: Tuple[float, float, float, float],
    ref_time: datetime,
    lookback_hours: int,
) -> np.ndarray:
    start_time = ref_time - timedelta(hours=int(lookback_hours))
    return get_fire_cells_heatmap(
        region_name=region_name,
        bbox=bbox,
        start_time=start_time,
        end_time=ref_time,
        mode="presence",
        clip=True,
    ).heatmap.astype(np.float32, copy=False)


def _derive_ruggedness_and_tpi(
    elevation: np.ndarray | None,
    slope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if elevation is None:
        zeros = np.zeros_like(slope, dtype=np.float32)
        return zeros, zeros

    # Simple local roughness proxy: gradient magnitude of elevation.
    grad_y, grad_x = np.gradient(elevation.astype(np.float32, copy=False))
    ruggedness = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32, copy=False)

    # Approximate topographic position index using global mean fallback.
    tpi = (elevation.astype(np.float32, copy=False) - np.float32(np.nanmean(elevation))).astype(np.float32, copy=False)
    return ruggedness, tpi


def _horizon_weighted_weather_mean(
    weather_cube: Any,
    var_name: str,
    horizons_hours: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    data = weather_cube.get(var_name)
    if data is None:
        return np.zeros(shape, dtype=np.float32)

    arr = np.asarray(data.values, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        return np.zeros(shape, dtype=np.float32)

    n_t = int(arr.shape[0])
    if n_t <= 0:
        return np.zeros(shape, dtype=np.float32)

    if horizons_hours and len(horizons_hours) == n_t:
        weights = np.asarray([max(1.0, float(h)) for h in horizons_hours], dtype=np.float32)
    else:
        weights = np.ones(n_t, dtype=np.float32)
    weights = weights / np.maximum(np.sum(weights), 1e-6)
    out = np.tensordot(weights, arr, axes=(0, 0)).astype(np.float32, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _flatten_features(
    region_name: str,
    bbox: Tuple[float, float, float, float],
    ref_time: datetime,
    horizons_hours: List[int],
) -> List[pd.DataFrame]:
    """Extract tabular features for a single reference time across all horizons."""
    inputs = build_spread_inputs(
        region_name=region_name,
        bbox=bbox,
        forecast_reference_time=ref_time,
        horizons_hours=horizons_hours,
    )

    slope = np.nan_to_num(
        inputs.terrain.slope.astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    aspect = np.nan_to_num(
        inputs.terrain.aspect.astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    elevation = (
        None
        if inputs.terrain.elevation is None
        else np.nan_to_num(
            inputs.terrain.elevation.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )

    aspect_rad = np.radians(aspect)
    aspect_sin = np.sin(aspect_rad).astype(np.float32, copy=False)
    aspect_cos = np.cos(aspect_rad).astype(np.float32, copy=False)

    fire_t0 = np.nan_to_num(inputs.active_fires.heatmap.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    fire_t_minus_6 = _load_fire_history(region_name, bbox, ref_time, lookback_hours=6)
    fire_t_minus_12 = _load_fire_history(region_name, bbox, ref_time, lookback_hours=12)
    ruggedness, tpi = _derive_ruggedness_and_tpi(elevation=elevation, slope=slope)

    ny, nx = fire_t0.shape
    lat_grid, lon_grid = np.meshgrid(inputs.window.lat, inputs.window.lon, indexing="ij")
    region_bucket = deterministic_region_bucket(region_name=region_name, bbox=bbox, n_buckets=1024)
    u10 = _horizon_weighted_weather_mean(inputs.weather_cube, "u10", horizons_hours, (ny, nx))
    v10 = _horizon_weighted_weather_mean(inputs.weather_cube, "v10", horizons_hours, (ny, nx))
    t2m = _horizon_weighted_weather_mean(inputs.weather_cube, "t2m", horizons_hours, (ny, nx))
    rh2m = _horizon_weighted_weather_mean(inputs.weather_cube, "rh2m", horizons_hours, (ny, nx))
    precip_24h = _horizon_weighted_weather_mean(inputs.weather_cube, "precip_24h", horizons_hours, (ny, nx))
    ndvi = _horizon_weighted_weather_mean(inputs.weather_cube, "ndvi", horizons_hours, (ny, nx))
    lfmc = _horizon_weighted_weather_mean(inputs.weather_cube, "lfmc", horizons_hours, (ny, nx))
    dfmc = _horizon_weighted_weather_mean(inputs.weather_cube, "dfmc", horizons_hours, (ny, nx))

    horizon_dfs: list[pd.DataFrame] = []
    for h_idx, horizon_h in enumerate(horizons_hours):
        _ = h_idx

        target_time = ref_time + timedelta(hours=horizon_h)
        target_start = target_time - timedelta(hours=3)
        target_end = target_time + timedelta(hours=3)

        target_heatmap = get_fire_cells_heatmap(
            region_name=region_name,
            bbox=bbox,
            start_time=target_start,
            end_time=target_end,
            mode="presence",
            clip=True,
        ).heatmap.astype(np.int8, copy=False)
        if target_heatmap.shape != fire_t0.shape:
            raise ValueError(
                "Target heatmap shape mismatch. "
                f"target={target_heatmap.shape} fire_t0={fire_t0.shape} "
                f"region={region_name!r} bbox={bbox!r} ref_time={ref_time!r} horizon_h={horizon_h}"
            )

        data: Dict[str, Any] = {
            "ref_time": [ref_time] * (ny * nx),
            "horizon_h": [horizon_h] * (ny * nx),
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
            "fire_t0": fire_t0.ravel(),
            "fire_t-6h": fire_t_minus_6.ravel(),
            "fire_t-12h": fire_t_minus_12.ravel(),
            "slope_deg": slope.ravel(),
            "aspect_sin": aspect_sin.ravel(),
            "aspect_cos": aspect_cos.ravel(),
            "u10": u10.ravel(),
            "v10": v10.ravel(),
            "t2m": t2m.ravel(),
            "rh2m": rh2m.ravel(),
            "precip_24h": precip_24h.ravel(),
            "ruggedness": ruggedness.ravel(),
            "tpi": tpi.ravel(),
            "ndvi": ndvi.ravel(),
            "lfmc": lfmc.ravel(),
            "dfmc": dfmc.ravel(),
            "region_id_embedding_input": np.full(ny * nx, region_bucket, dtype=np.int32),
            "label": target_heatmap.ravel().astype(np.int8, copy=False),
        }

        if elevation is not None:
            data["elevation_m"] = elevation.ravel()
        else:
            data["elevation_m"] = np.zeros(ny * nx, dtype=np.float32)

        df = pd.DataFrame(data)
        df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
        df["region_name"] = region_name
        df["region_bucket"] = region_bucket
        df["ref_year"] = pd.to_datetime(df["ref_time"], utc=True).dt.year.astype(int)

        horizon_dfs.append(df)

    return horizon_dfs


def _near_periphery_negative_mask(
    df: pd.DataFrame,
    *,
    radius_cells: int = 2,
) -> pd.Series:
    """Return a mask for negatives near the positive fire boundary."""
    ny = int(df["lat"].nunique())
    nx = int(df["lon"].nunique())
    if ny * nx != len(df):
        return pd.Series(False, index=df.index)

    labels = np.asarray(df["label"], dtype=np.int8).reshape(ny, nx)
    positives = labels > 0
    if not positives.any():
        return pd.Series(False, index=df.index)

    boundary = binary_dilation(positives, iterations=int(radius_cells)) & (~positives)
    return pd.Series(boundary.ravel(), index=df.index)


def _sample_negatives_with_periphery_priority(
    df: pd.DataFrame,
    *,
    pos_mask: pd.Series,
    neg_mask: pd.Series,
    must_keep_neg: pd.Series,
    n_other_target: int,
    random_state: int,
) -> pd.DataFrame:
    if n_other_target <= 0:
        return df[pos_mask | must_keep_neg]

    other_neg = neg_mask & (~must_keep_neg)
    if n_other_target >= int(other_neg.sum()):
        return df[pos_mask | neg_mask]

    near_boundary = _near_periphery_negative_mask(df)
    boundary_pool = other_neg & near_boundary
    background_pool = other_neg & (~near_boundary)

    boundary_target = int(round(n_other_target * 0.70))
    boundary_target = min(boundary_target, int(boundary_pool.sum()))
    background_target = n_other_target - boundary_target

    if background_target > int(background_pool.sum()):
        spill = background_target - int(background_pool.sum())
        background_target = int(background_pool.sum())
        boundary_target = min(boundary_target + spill, int(boundary_pool.sum()))

    sampled_parts = []
    if boundary_target > 0:
        sampled_parts.append(
            df[boundary_pool].sample(n=boundary_target, random_state=random_state)
        )
    if background_target > 0:
        sampled_parts.append(
            df[background_pool].sample(n=background_target, random_state=random_state + 17)
        )

    keep_mask = pos_mask | must_keep_neg
    if sampled_parts:
        return pd.concat([df[keep_mask], *sampled_parts], axis=0)
    return df[keep_mask]


def split_hindcast_dataset(
    df: pd.DataFrame,
    *,
    split_year: int | None = None,
    validation_region_buckets: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-safe train/eval split using year and region-bucket holdout."""
    if df.empty:
        return df.copy(), df.copy()

    out = df.copy()
    out["ref_time"] = pd.to_datetime(out["ref_time"], utc=True)
    if "ref_year" not in out.columns:
        out["ref_year"] = out["ref_time"].dt.year.astype(int)
    if "region_bucket" not in out.columns:
        out["region_bucket"] = (
            ((out["lat"].round(2) * 10).astype(int) + (out["lon"].round(2) * 10).astype(int)).abs() % 10
        )

    holdout_year = int(split_year if split_year is not None else out["ref_year"].max())
    holdout_buckets = validation_region_buckets or {0}

    eval_mask = (out["ref_year"] >= holdout_year) & (out["region_bucket"].isin(holdout_buckets))
    train_df = out[~eval_mask].copy()
    eval_df = out[eval_mask].copy()

    if train_df.empty or eval_df.empty:
        split_dt = out["ref_time"].quantile(0.8)
        train_df = out[out["ref_time"] < split_dt].copy()
        eval_df = out[out["ref_time"] >= split_dt].copy()

    return train_df, eval_df


def _to_tensor_case(
    df: pd.DataFrame,
    *,
    channel_names: tuple[str, ...],
) -> dict[str, Any]:
    ny = int(df["lat"].nunique())
    nx = int(df["lon"].nunique())
    if ny * nx != len(df):
        raise ValueError("Tensor conversion requires full (lat, lon) grid rows.")

    tensor = np.stack(
        [np.asarray(df[ch], dtype=np.float32).reshape(ny, nx) for ch in channel_names],
        axis=0,
    )
    label = np.asarray(df["label"], dtype=np.float32).reshape(ny, nx)
    lat = np.asarray(sorted(df["lat"].unique()), dtype=np.float32)
    lon = np.asarray(sorted(df["lon"].unique()), dtype=np.float32)

    return {
        "ref_time": pd.to_datetime(df["ref_time"].iloc[0], utc=True),
        "horizon_h": int(df["horizon_h"].iloc[0]),
        "region_name": str(df["region_name"].iloc[0]),
        "region_bucket": int(df["region_bucket"].iloc[0]),
        "x_tensor": tensor,
        "y_tensor": label,
        "lat": lat,
        "lon": lon,
        "channel_names": list(channel_names),
    }


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
    output_mode: Literal["tabular", "tensor"] = "tabular",
    tensor_channels: tuple[str, ...] = V2_TENSOR_CHANNELS,
) -> pd.DataFrame | list[dict[str, Any]]:
    """Build hindcast data in tabular (v1) or tensor (v2) mode."""
    engine = get_engine()
    candidate_times = sample_fire_reference_times(
        engine, bbox, start_time, end_time, min_detections, interval_hours
    )
    LOGGER.info("Found %s candidate reference times for hindcast.", len(candidate_times))

    all_dfs: list[pd.DataFrame] = []
    for ref_time in candidate_times:
        try:
            LOGGER.info("Processing ref_time=%s ...", ref_time)
            horizon_dfs = _flatten_features(region_name, bbox, ref_time, horizons_hours)
            for df in horizon_dfs:
                if negative_ratio is None:
                    all_dfs.append(df)
                    continue

                pos_mask = df["label"] == 1
                neg_mask = df["label"] == 0
                n_pos = int(pos_mask.sum())

                must_keep_neg = neg_mask & (df["fire_t0"] > 0)
                n_must_keep = int(must_keep_neg.sum())

                if n_pos > 0:
                    n_neg_target = int(n_pos * float(negative_ratio))
                else:
                    n_neg_target = int(max(min_negative_samples, n_must_keep))

                n_other_target = max(0, n_neg_target - n_must_keep)
                horizon_val = int(df["horizon_h"].iloc[0]) if "horizon_h" in df.columns and len(df) else 0
                rs = int(seed + int(ref_time.timestamp()) + horizon_val * 1000) & 0x7FFFFFFF
                sampled = _sample_negatives_with_periphery_priority(
                    df,
                    pos_mask=pos_mask,
                    neg_mask=neg_mask,
                    must_keep_neg=must_keep_neg,
                    n_other_target=n_other_target,
                    random_state=rs,
                )
                all_dfs.append(sampled)

        except Exception:
            LOGGER.exception("Failed to process ref_time=%s; skipping.", ref_time)

    if not all_dfs:
        return [] if output_mode == "tensor" else pd.DataFrame()

    merged = pd.concat(all_dfs, ignore_index=True)
    if output_mode == "tabular":
        return merged

    tensor_cases = []
    for (_, h), chunk in merged.groupby(["ref_time", "horizon_h"], sort=True):
        _ = h
        tensor_cases.append(_to_tensor_case(chunk, channel_names=tensor_channels))
    return tensor_cases


def build_hindcast_tensor_dataset(
    region_name: str,
    bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    horizons_hours: List[int],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Convenience wrapper for v2 spatial tensor training data."""
    out = build_hindcast_dataset(
        region_name=region_name,
        bbox=bbox,
        start_time=start_time,
        end_time=end_time,
        horizons_hours=horizons_hours,
        output_mode="tensor",
        **kwargs,
    )
    return list(out)
