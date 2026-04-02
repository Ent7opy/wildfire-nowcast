"""Training snapshot extractor for ignition probability model.

Generates a grid of 0.25° cells (matching the GFS forecast resolution) for a
given bbox and date range, extracts per-cell features from the database, and
constructs binary ignition labels.

Positive label (ignition = 1):
  Grid cell has a confirmed fire detection (is_noise=False) in the next 24 h
  window AND no confirmed detection at the same cell in the preceding 48 h
  (i.e., a new ignition, not spread from an existing fire).

Negative label (ignition = 0):
  All other cells in the same time windows.  The negative class is heavily
  dominant; callers should downsample before training.

Usage:
  python -m ml.ignition.snapshot \\
      --bbox -125 30 -100 50 \\
      --start 2025-06-01 --end 2025-09-30 \\
      --version v1 \\
      --out data/snapshots/ignition
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.db import get_engine
from ml.parquet_io import write_parquet_with_fallback

LOGGER = logging.getLogger("ignition.snapshot")

_GFS_GRID_DEG = 0.25  # 0.25-degree grid matches GFS and the task specification
_CELL_MATCH_DEG = _GFS_GRID_DEG / 2  # Half-cell radius for "same cell" test
_DAYS_SINCE_BURN_CAP = 3650  # 10-year cap for cells with no recent fire history
_PRECIP_LOOKBACK_DAYS = 7
_PRIOR_FIRE_WINDOW_H = 48  # No fire in preceding 48h → candidate new ignition
_LABEL_HORIZON_H = 24  # Ignition in next 24h → positive label

# Columns whose missing-value sentinel is False (bool) rather than NaN (float).
_BOOL_COLS_DEFAULT: dict[str, bool] = {"thunderstorm_active": False}

# Flammability mapping (mirrors denoiser _CLASS_SCORES from lulc_worldcover_ingest).
_LULC_CLASS_FLAMMABILITY: dict[int, float] = {
    10: 1.0,   # Tree cover
    20: 1.0,   # Shrubland
    30: 1.0,   # Grassland
    40: 0.7,   # Cropland
    50: 0.1,   # Built-up
    60: 0.1,   # Bare / sparse vegetation
    70: 0.0,   # Snow and ice
    80: 0.0,   # Permanent water bodies
    90: 0.2,   # Herbaceous wetland
    95: 0.5,   # Mangroves
    100: 0.2,  # Moss and lichen
}


def _build_grid(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> pd.DataFrame:
    """Return a DataFrame of grid cell centroids at GFS resolution."""
    lons = np.arange(
        np.ceil(min_lon / _GFS_GRID_DEG) * _GFS_GRID_DEG,
        max_lon + _GFS_GRID_DEG / 4,
        _GFS_GRID_DEG,
    )
    lats = np.arange(
        np.ceil(min_lat / _GFS_GRID_DEG) * _GFS_GRID_DEG,
        max_lat + _GFS_GRID_DEG / 4,
        _GFS_GRID_DEG,
    )
    lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    return pd.DataFrame(
        {"lon_grid": lons_mesh.ravel(), "lat_grid": lats_mesh.ravel()}
    )


def _grid_bbox(grid: pd.DataFrame, pad: float = 0.0) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) for a grid DataFrame with optional padding."""
    return (
        float(grid["lon_grid"].min()) - pad,
        float(grid["lat_grid"].min()) - pad,
        float(grid["lon_grid"].max()) + pad,
        float(grid["lat_grid"].max()) + pad,
    )


# SQL template shared by both the future-window and prior-window detection counts.
_DETECTION_COUNT_SQL = """
SELECT
    ROUND(lon / :grid_deg) * :grid_deg  AS lon_grid,
    ROUND(lat / :grid_deg) * :grid_deg  AS lat_grid,
    COUNT(*)                             AS det_count
FROM fire_detections
WHERE acq_time >= :t_start AND acq_time < :t_end
  AND lon BETWEEN :min_lon AND :max_lon
  AND lat BETWEEN :min_lat AND :max_lat
  AND is_noise = false
GROUP BY
    ROUND(lon / :grid_deg) * :grid_deg,
    ROUND(lat / :grid_deg) * :grid_deg
"""


def _query_weather_features(
    engine: Engine,
    grid: pd.DataFrame,
    ref_time: datetime,
) -> pd.DataFrame:
    """Return weather features for each grid cell at the nearest analysis time.

    Joins weather_point_cache with weather_runs to find the most recent GFS
    analysis (forecast_hour = 0) run on or before ref_time.  For precipitation,
    sums tp over the preceding PRECIP_LOOKBACK_DAYS days.
    """
    min_lon, min_lat, max_lon, max_lat = _grid_bbox(grid, pad=_GFS_GRID_DEG)

    stmt = text(
        """
        SELECT DISTINCT ON (wpc.lat_grid, wpc.lon_grid)
            wpc.lat_grid,
            wpc.lon_grid,
            wpc.t2m - 273.15                                  AS temperature_c,
            wpc.rh2m                                          AS relative_humidity,
            SQRT(wpc.u10 * wpc.u10 + wpc.v10 * wpc.v10) * 3.6 AS wind_speed_kmh
        FROM weather_point_cache wpc
        JOIN weather_runs wr ON wpc.run_id = wr.id
        WHERE wpc.lat_grid BETWEEN :min_lat AND :max_lat
          AND wpc.lon_grid BETWEEN :min_lon AND :max_lon
          AND wpc.forecast_hour = 0
          AND wr.run_time <= :ref_time
          AND wr.status = 'complete'
        ORDER BY wpc.lat_grid, wpc.lon_grid, wr.run_time DESC
        """
    )
    precip_stmt = text(
        """
        SELECT
            wpc.lat_grid,
            wpc.lon_grid,
            SUM(wpc.tp)  AS precip_last_7d_mm
        FROM weather_point_cache wpc
        JOIN weather_runs wr ON wpc.run_id = wr.id
        WHERE wpc.lat_grid BETWEEN :min_lat AND :max_lat
          AND wpc.lon_grid BETWEEN :min_lon AND :max_lon
          AND wpc.forecast_hour = 0
          AND wr.run_time >= :precip_start
          AND wr.run_time <= :ref_time
          AND wr.status = 'complete'
        GROUP BY wpc.lat_grid, wpc.lon_grid
        """
    )
    precip_start = ref_time - timedelta(days=_PRECIP_LOOKBACK_DAYS)
    shared_params = {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
    }

    with engine.connect() as conn:
        weather_df = pd.read_sql(
            stmt, conn, params={**shared_params, "ref_time": ref_time}
        )
        precip_df = pd.read_sql(
            precip_stmt, conn,
            params={**shared_params, "precip_start": precip_start, "ref_time": ref_time},
        )

    if weather_df.empty:
        LOGGER.warning(
            "WARNING [ignition-snapshot] No weather data found for ref_time=%s bbox=[%s,%s,%s,%s]. "
            "Weather features will be NaN. TARGET_STAGE: science_grade",
            ref_time.isoformat(), min_lon, min_lat, max_lon, max_lat,
        )
        return pd.DataFrame(
            columns=["lat_grid", "lon_grid", "temperature_c", "relative_humidity",
                     "wind_speed_kmh", "precip_last_7d_mm"],
        )

    if not precip_df.empty:
        weather_df = weather_df.merge(precip_df, on=["lat_grid", "lon_grid"], how="left")
    else:
        weather_df["precip_last_7d_mm"] = float("nan")

    return weather_df


def _query_lulc_features(
    engine: Engine,
    grid: pd.DataFrame,
) -> pd.DataFrame:
    """Return LULC flammability class for each grid cell.

    For cells with no nearby fire detections, lulc features will be NaN.
    Uses fire_detections.landcover_class which is populated by lulc_worldcover_ingest.
    """
    min_lon, min_lat, max_lon, max_lat = _grid_bbox(grid, pad=_GFS_GRID_DEG)

    stmt = text(
        """
        SELECT
            ROUND(lon / :grid_deg) * :grid_deg  AS lon_grid,
            ROUND(lat / :grid_deg) * :grid_deg  AS lat_grid,
            MODE() WITHIN GROUP (ORDER BY landcover_class)   AS lulc_class,
            AVG(landcover_score)                             AS lulc_flammability
        FROM fire_detections
        WHERE lon BETWEEN :min_lon AND :max_lon
          AND lat BETWEEN :min_lat AND :max_lat
          AND landcover_class IS NOT NULL
        GROUP BY
            ROUND(lon / :grid_deg) * :grid_deg,
            ROUND(lat / :grid_deg) * :grid_deg
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(
            stmt, conn,
            params={
                "grid_deg": _GFS_GRID_DEG,
                "min_lon": min_lon, "max_lon": max_lon,
                "min_lat": min_lat, "max_lat": max_lat,
            },
        )

    if df.empty:
        LOGGER.warning(
            "WARNING [ignition-snapshot] No LULC data found in bbox. "
            "lulc_flammability and lulc_class will be NaN. TARGET_STAGE: science_grade"
        )
        return pd.DataFrame(columns=["lon_grid", "lat_grid", "lulc_class", "lulc_flammability"])

    df["lulc_class"] = pd.to_numeric(df["lulc_class"], errors="coerce").astype("Int64")
    df["lulc_flammability"] = pd.to_numeric(df["lulc_flammability"], errors="coerce")

    # Fill missing flammability from the class → flammability mapping.
    mask = df["lulc_flammability"].isna() & df["lulc_class"].notna()
    df.loc[mask, "lulc_flammability"] = df.loc[mask, "lulc_class"].map(
        _LULC_CLASS_FLAMMABILITY
    )

    return df[["lon_grid", "lat_grid", "lulc_class", "lulc_flammability"]]


def _query_thunderstorm_features(
    engine: Engine,
    grid: pd.DataFrame,
    ref_time: datetime,
    tolerance_hours: float = 6.0,
) -> pd.DataFrame:
    """Return thunderstorm_active flag for each grid cell."""
    stmt = text(
        """
        SELECT
            ROUND(grid_lon / :grid_deg) * :grid_deg  AS lon_grid,
            ROUND(grid_lat / :grid_deg) * :grid_deg  AS lat_grid,
            BOOL_OR(thunderstorm_active)              AS thunderstorm_active
        FROM ignition_lightning_proxy
        WHERE valid_time BETWEEN :t_start AND :t_end
        GROUP BY
            ROUND(grid_lon / :grid_deg) * :grid_deg,
            ROUND(grid_lat / :grid_deg) * :grid_deg
        """
    )
    t_start = ref_time - timedelta(hours=tolerance_hours)
    t_end = ref_time + timedelta(hours=tolerance_hours)
    with engine.connect() as conn:
        return pd.read_sql(
            stmt, conn,
            params={"grid_deg": _GFS_GRID_DEG, "t_start": t_start, "t_end": t_end},
        )


def _load_raster_values_at_points(
    storage_path: str,
    lons: np.ndarray,
    lats: np.ndarray,
    variable: Optional[str] = None,
) -> np.ndarray:
    """Sample a NetCDF/GeoTIFF raster at the given (lon, lat) points.

    Returns an array of float values, NaN where sampling fails.
    """
    try:
        import xarray as xr  # noqa: PLC0415

        path = Path(storage_path)
        if not path.exists():
            return np.full(len(lons), float("nan"))

        ds = xr.open_dataset(str(path))
        da = ds[variable] if variable and variable in ds else ds[list(ds.data_vars)[0]]

        lon_coord = next((c for c in da.coords if c.lower() in ("lon", "longitude", "x")), None)
        lat_coord = next((c for c in da.coords if c.lower() in ("lat", "latitude", "y")), None)
        if lon_coord is None or lat_coord is None:
            ds.close()
            return np.full(len(lons), float("nan"))

        if "time" in da.dims:
            da = da.isel(time=-1)

        values = da.sel(
            {lon_coord: xr.DataArray(lons, dims="points"),
             lat_coord: xr.DataArray(lats, dims="points")},
            method="nearest",
        ).values.astype(float)

        ds.close()
        return values

    except Exception as exc:
        LOGGER.debug("Failed to load raster %s: %s", storage_path, exc)
        return np.full(len(lons), float("nan"))


def _prefetch_raster_feature(
    engine: Engine,
    table: str,
    time_col: str,
    grid: pd.DataFrame,
    start: datetime,
    end: datetime,
    result_col: str,
    tolerance_days: int = 30,
    variable: Optional[str] = None,
) -> pd.DataFrame:
    """Query the most recent raster run once for the full snapshot period and sample it.

    Called once before the daily loop to avoid re-querying and re-opening the
    raster file on every iteration.

    Args:
        table: DB table name ("fuel_moisture_runs" or "drought_index_runs").
        time_col: Timestamp column name in the table ("run_time" or "valid_time").
        result_col: Name for the output column ("fuel_moisture" or "drought_index").
    """
    min_lon, min_lat, max_lon, max_lat = _grid_bbox(grid)
    t_start = start - timedelta(days=tolerance_days)

    stmt = text(
        f"""
        SELECT id, {time_col}, storage_path, variable
        FROM {table}
        WHERE status = 'complete'
          AND {time_col} BETWEEN :t_start AND :ref_time
          AND bbox_min_lon <= :max_lon AND bbox_max_lon >= :min_lon
          AND bbox_min_lat <= :max_lat AND bbox_max_lat >= :min_lat
        ORDER BY {time_col} DESC
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        run = pd.read_sql(
            stmt, conn,
            params={
                "t_start": t_start, "ref_time": end,
                "min_lon": min_lon, "max_lon": max_lon,
                "min_lat": min_lat, "max_lat": max_lat,
            },
        )

    result = grid[["lon_grid", "lat_grid"]].copy()
    result[result_col] = float("nan")

    if run.empty:
        LOGGER.warning(
            "WARNING [ignition-snapshot] No %s found within %d days of %s–%s. "
            "%s will be NaN. TARGET_STAGE: science_grade",
            table, tolerance_days, start.date(), end.date(), result_col,
        )
        return result

    var = run.iloc[0].get("variable") or variable
    values = _load_raster_values_at_points(
        run.iloc[0]["storage_path"],
        grid["lon_grid"].values,
        grid["lat_grid"].values,
        variable=var,
    )
    result[result_col] = values
    return result


def compute_days_since_last_burn(
    engine: Engine,
    grid: pd.DataFrame,
    ref_time: datetime,
    proximity_m: float = 5000.0,
    cap_days: int = _DAYS_SINCE_BURN_CAP,
) -> pd.Series:
    """Return days-since-last-burn for each grid cell.

    Queries fire_perimeters and fire_events for the most recent confirmed fire
    within *proximity_m* metres of each cell centroid, returning the number of
    days elapsed from that event to ref_time.  Cells with no fire history within
    cap_days are assigned cap_days (representing high accumulated fuel load).

    Results are ordered to match *grid*.

    Args:
        engine: SQLAlchemy engine.
        grid: DataFrame with lon_grid, lat_grid columns.
        ref_time: Reference datetime (UTC).
        proximity_m: Search radius in metres.
        cap_days: Maximum days to return for cells with no recent fire.

    Returns:
        pd.Series of float days, indexed to match grid.
    """
    n = len(grid)
    if n == 0:
        return pd.Series([], dtype=float)

    lons = grid["lon_grid"].tolist()
    lats = grid["lat_grid"].tolist()

    stmt = text(
        """
        WITH cell_coords AS (
            SELECT
                ordinality - 1                             AS cell_idx,
                lon_val                                    AS lon,
                lat_val                                    AS lat,
                ST_SetSRID(ST_MakePoint(lon_val, lat_val), 4326) AS geom
            FROM UNNEST(:lons::float[], :lats::float[])
                 WITH ORDINALITY AS t(lon_val, lat_val, ordinality)
        ),

        perimeter_burns AS (
            SELECT
                cc.cell_idx,
                EXTRACT(EPOCH FROM (
                    :ref_time::timestamptz - GREATEST(
                        COALESCE(fp.fire_end, fp.fire_start, fp.created_at),
                        '-infinity'::timestamptz
                    )
                )) / 86400.0 AS days_ago
            FROM cell_coords cc
            JOIN fire_perimeters fp
              ON ST_DWithin(fp.geom::geography, cc.geom::geography, :proximity_m)
            WHERE COALESCE(fp.fire_end, fp.fire_start, fp.created_at) < :ref_time::timestamptz
        ),

        event_burns AS (
            SELECT
                cc.cell_idx,
                EXTRACT(EPOCH FROM (
                    :ref_time::timestamptz - GREATEST(
                        COALESCE(fe.end_time, fe.start_time, fe.created_at),
                        '-infinity'::timestamptz
                    )
                )) / 86400.0 AS days_ago
            FROM cell_coords cc
            JOIN fire_events fe
              ON ST_DWithin(fe.geom::geography, cc.geom::geography, :proximity_m)
            WHERE fe.denoiser_decision IN ('pass', 'downweight')
              AND COALESCE(fe.end_time, fe.start_time, fe.created_at) < :ref_time::timestamptz
        ),

        all_burns AS (
            SELECT cell_idx, days_ago FROM perimeter_burns
            UNION ALL
            SELECT cell_idx, days_ago FROM event_burns
        )

        SELECT
            cc.cell_idx,
            LEAST(
                COALESCE(MIN(ab.days_ago), :cap_days::float),
                :cap_days::float
            ) AS days_since_last_burn
        FROM cell_coords cc
        LEFT JOIN all_burns ab USING (cell_idx)
        WHERE ab.days_ago >= 0 OR ab.days_ago IS NULL
        GROUP BY cc.cell_idx
        ORDER BY cc.cell_idx
        """
    )

    with engine.connect() as conn:
        result_df = pd.read_sql(
            stmt, conn,
            params={
                "lons": lons, "lats": lats,
                "ref_time": ref_time,
                "proximity_m": proximity_m,
                "cap_days": float(cap_days),
            },
        )

    # Map results back to grid order, defaulting to cap_days for missing cells.
    # Python-level cap is defense-in-depth alongside SQL LEAST.
    cap = float(cap_days)
    days_map: dict[int, float] = dict(
        zip(result_df["cell_idx"].tolist(), result_df["days_since_last_burn"].tolist())
    )
    return pd.Series(
        [min(days_map.get(i, cap), cap) for i in range(n)],
        index=grid.index,
        name="days_since_last_burn",
        dtype=float,
    )


def _query_ignition_labels(
    engine: Engine,
    grid: pd.DataFrame,
    window_start: datetime,
    label_horizon_h: int = _LABEL_HORIZON_H,
    prior_fire_window_h: int = _PRIOR_FIRE_WINDOW_H,
    cell_radius_deg: float = _CELL_MATCH_DEG,
) -> pd.DataFrame:
    """Construct ignition labels for each grid cell.

    Positive (1): Cell has a confirmed, non-noise fire detection in
    [window_start, window_start + label_horizon_h) AND no confirmed detection
    in [window_start - prior_fire_window_h, window_start) at the same cell.

    Negative (0): all other cells.

    Returns a DataFrame with lon_grid, lat_grid, ignition_label columns.
    """
    window_end = window_start + timedelta(hours=label_horizon_h)
    prior_start = window_start - timedelta(hours=prior_fire_window_h)
    min_lon, min_lat, max_lon, max_lat = _grid_bbox(grid, pad=cell_radius_deg)

    shared_params = {
        "grid_deg": _GFS_GRID_DEG,
        "min_lon": min_lon, "max_lon": max_lon,
        "min_lat": min_lat, "max_lat": max_lat,
    }
    stmt = text(_DETECTION_COUNT_SQL)

    with engine.connect() as conn:
        future_df = pd.read_sql(
            stmt, conn,
            params={**shared_params, "t_start": window_start, "t_end": window_end},
        )
        prior_df = pd.read_sql(
            stmt, conn,
            params={**shared_params, "t_start": prior_start, "t_end": window_start},
        )

    labels = grid[["lon_grid", "lat_grid"]].copy()
    labels["ignition_label"] = 0

    if future_df.empty:
        return labels

    future_df = future_df.rename(columns={"det_count": "future_count"})
    prior_df = prior_df.rename(columns={"det_count": "prior_count"})

    merged = future_df.merge(prior_df, on=["lon_grid", "lat_grid"], how="left")
    merged["prior_count"] = pd.to_numeric(merged["prior_count"], errors="coerce").fillna(0).astype(int)

    new_ignition_mask = (merged["future_count"] > 0) & (merged["prior_count"] == 0)
    new_ignition = merged.loc[new_ignition_mask, ["lon_grid", "lat_grid"]].copy()
    new_ignition["ignition_label"] = 1

    labels = labels.merge(new_ignition, on=["lon_grid", "lat_grid"], how="left", suffixes=("", "_new"))
    if "ignition_label_new" in labels.columns:
        labels["ignition_label"] = (
            labels["ignition_label_new"].fillna(labels["ignition_label"]).astype(int)
        )
        labels.drop(columns=["ignition_label_new"], inplace=True)

    return labels


def extract_snapshot(
    *,
    engine: Engine,
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    train_split_percentile: float = 0.8,
) -> dict[str, pd.DataFrame]:
    """Extract the full ignition feature matrix for a bbox/date range.

    Args:
        engine: SQLAlchemy engine.
        bbox: (min_lon, min_lat, max_lon, max_lat).
        start: Start of the date range (inclusive, UTC).
        end: End of the date range (exclusive, UTC).
        train_split_percentile: Fraction of days (chronological) used for train.

    Returns:
        Dict with keys "train", "eval", "full" → DataFrames.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    grid = _build_grid(min_lon, min_lat, max_lon, max_lat)
    LOGGER.info(
        "Grid: %d cells at %.2f° resolution for bbox=%s",
        len(grid), _GFS_GRID_DEG, bbox,
    )

    # --- Static features: queried once for the full snapshot period ---
    lulc_df = _query_lulc_features(engine, grid)

    # Raster features are fetched once; the most recent run before `end` is used
    # for all training days (weekly/monthly cadence means one run covers the window).
    fm_df = _prefetch_raster_feature(
        engine, "fuel_moisture_runs", "run_time", grid, start, end, "fuel_moisture"
    )
    di_df = _prefetch_raster_feature(
        engine, "drought_index_runs", "valid_time", grid, start, end, "drought_index"
    )

    day_frames: list[pd.DataFrame] = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    while current < end:
        t0 = time.perf_counter()
        ref_time = current

        weather_df = _query_weather_features(engine, grid, ref_time)
        ts_df = _query_thunderstorm_features(engine, grid, ref_time)
        dslb = compute_days_since_last_burn(engine, grid, ref_time)
        labels_df = _query_ignition_labels(engine, grid, ref_time)

        frame = grid.copy()
        frame["ref_time"] = ref_time

        for sub_df, cols in [
            (weather_df, ["temperature_c", "relative_humidity", "wind_speed_kmh", "precip_last_7d_mm"]),
            (lulc_df, ["lulc_class", "lulc_flammability"]),
            (fm_df, ["fuel_moisture"]),
            (di_df, ["drought_index"]),
            (ts_df, ["thunderstorm_active"]),
        ]:
            if sub_df.empty or "lon_grid" not in sub_df.columns:
                for col in cols:
                    frame[col] = _BOOL_COLS_DEFAULT.get(col, float("nan"))
            else:
                frame = frame.merge(
                    sub_df[["lon_grid", "lat_grid"] + cols],
                    on=["lon_grid", "lat_grid"],
                    how="left",
                )
                for col in cols:
                    if col not in frame.columns:
                        frame[col] = _BOOL_COLS_DEFAULT.get(col, float("nan"))

        frame["thunderstorm_active"] = frame["thunderstorm_active"].fillna(False).astype(bool)
        frame["days_since_last_burn"] = dslb.values
        frame = frame.merge(
            labels_df[["lon_grid", "lat_grid", "ignition_label"]],
            on=["lon_grid", "lat_grid"],
            how="left",
        )
        frame["ignition_label"] = frame["ignition_label"].fillna(0).astype(int)

        day_frames.append(frame)
        LOGGER.info(
            "Day %s: %d cells, %d positives — %.3fs",
            current.date(), len(frame), int(frame["ignition_label"].sum()),
            time.perf_counter() - t0,
        )
        current += timedelta(days=1)

    if not day_frames:
        LOGGER.warning("WARNING [ignition-snapshot] No day frames extracted. Check bbox/date range.")
        empty = pd.DataFrame(
            columns=[
                "lon_grid", "lat_grid", "ref_time", "temperature_c", "relative_humidity",
                "wind_speed_kmh", "precip_last_7d_mm", "lulc_class", "lulc_flammability",
                "fuel_moisture", "drought_index", "thunderstorm_active", "days_since_last_burn",
                "ignition_label",
            ]
        )
        return {"train": empty, "eval": empty, "full": empty}

    full = pd.concat(day_frames, ignore_index=True)
    LOGGER.info(
        "Total: %d rows, %d positives (%.2f%%)",
        len(full), int(full["ignition_label"].sum()), 100.0 * full["ignition_label"].mean(),
    )

    # Chronological train/eval split.
    all_days = sorted(full["ref_time"].unique())
    n_train = max(1, int(len(all_days) * train_split_percentile))
    train_days = set(all_days[:n_train])
    train = full[full["ref_time"].isin(train_days)].reset_index(drop=True)
    eval_ = full[~full["ref_time"].isin(train_days)].reset_index(drop=True)

    return {"train": train, "eval": eval_, "full": full}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Export ignition probability training snapshot."
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), required=True,
    )
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD (exclusive)")
    parser.add_argument("--version", type=str, default="v1", help="Snapshot version tag")
    parser.add_argument(
        "--out", type=str, default="data/snapshots/ignition", help="Output directory"
    )
    parser.add_argument(
        "--train-split", type=float, default=0.8,
        help="Fraction of days to use for train split (chronological)",
    )
    args = parser.parse_args()

    bbox = tuple(args.bbox)
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    out_dir = Path(args.out) / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    splits = extract_snapshot(
        engine=engine, bbox=bbox, start=start, end=end,
        train_split_percentile=args.train_split,
    )

    for split_name, df in splits.items():
        out_path = str(out_dir / f"{split_name}.parquet")
        write_parquet_with_fallback(df, out_path)
        LOGGER.info("Wrote %s: %d rows → %s", split_name, len(df), out_path)


if __name__ == "__main__":
    main()
