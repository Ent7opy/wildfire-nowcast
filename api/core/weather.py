"""Shared weather point-lookup helpers.

Extracted from ``api.fires.scoring`` so that both the fire-scoring pipeline
and the risk-grid module can use the same logic without a cross-package
dependency on fire internals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from sqlalchemy import text

from api.db import get_engine

LOGGER = logging.getLogger(__name__)


def _to_numpy_datetime64(dt: datetime) -> np.datetime64:
    """Convert a datetime to numpy datetime64 with proper UTC handling.

    This helper centralizes timezone handling to avoid inconsistencies when
    converting timezone-aware datetimes to numpy datetime64.

    Args:
        dt: A timezone-aware or naive datetime. If naive, assumed to be UTC.

    Returns:
        numpy.datetime64 in millisecond precision UTC.
    """
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)

    # Using 'ms' precision to avoid nanosecond overflow issues with some xarray versions
    return np.datetime64(dt_utc.replace(tzinfo=None), "ms")


def get_weather_data_for_point(
    *,
    lat: float,
    lon: float,
    ref_time: datetime,
    time_tolerance_hours: float,
    precip_lookback_hours: float,
) -> dict[str, float] | None:
    """Query weather data for a specific point and time.

    Args:
        lat: Latitude of the point
        lon: Longitude of the point
        ref_time: Reference time for weather data
        time_tolerance_hours: Maximum time difference allowed for matching
        precip_lookback_hours: Hours to look back for precipitation accumulation

    Returns:
        Dict with weather variables or None if data unavailable:
        - rh2m: Relative humidity at 2m (%)
        - precip_recent_mm: Recent precipitation accumulation (mm)
        - wind_speed_ms: Wind speed (m/s)
    """
    stmt = text("""
        SELECT id, storage_path, run_time
        FROM weather_runs
        WHERE status = 'completed'
          AND run_time <= :ref_time
          AND run_time >= :ref_time - INTERVAL '1 hour' * :tolerance_hours
          AND COALESCE(bbox_min_lon, -180.0) <= :lon AND COALESCE(bbox_max_lon, 180.0) >= :lon
          AND COALESCE(bbox_min_lat, -90.0) <= :lat AND COALESCE(bbox_max_lat, 90.0) >= :lat
        ORDER BY run_time DESC, created_at DESC
        LIMIT 1
    """)

    with get_engine().connect() as conn:
        row = conn.execute(
            stmt,
            {
                "ref_time": ref_time,
                "tolerance_hours": time_tolerance_hours,
                "lat": lat,
                "lon": lon,
            },
        ).mappings().first()

    if not row:
        LOGGER.debug(
            "No weather run found for point (lat=%s, lon=%s) at time %s",
            lat, lon, ref_time
        )
        return None

    storage_path = Path(row["storage_path"])
    if not storage_path.is_absolute():
        storage_path = Path.cwd() / storage_path

    ds = None
    try:
        ds = xr.open_dataset(storage_path)
        ds_point = ds.sel(lat=lat, lon=lon, method="nearest")

        ref_time_64 = _to_numpy_datetime64(ref_time)
        if "time" in ds_point.coords:
            ds_point = ds_point.sel(time=ref_time_64, method="nearest")

        result: dict[str, float] = {}

        if "rh2m" in ds_point.data_vars:
            rh_val = float(ds_point["rh2m"].values)
            if not np.isnan(rh_val):
                result["rh2m"] = rh_val

        if "u10" in ds_point.data_vars and "v10" in ds_point.data_vars:
            u10_val = float(ds_point["u10"].values)
            v10_val = float(ds_point["v10"].values)
            if not np.isnan(u10_val) and not np.isnan(v10_val):
                result["wind_speed_ms"] = float(np.sqrt(u10_val**2 + v10_val**2))

        if "tp" in ds_point.data_vars and "time" in ds.coords:
            try:
                precip_start_64 = _to_numpy_datetime64(
                    ref_time - timedelta(hours=precip_lookback_hours)
                )
                ds_precip = ds_point.sel(time=slice(precip_start_64, ref_time_64))

                if "tp" in ds_precip.data_vars and len(ds_precip.time) > 0:
                    precip_sum = float(ds_precip["tp"].sum().values)
                    if not np.isnan(precip_sum):
                        # GFS outputs precipitation in meters; convert to mm
                        result["precip_recent_mm"] = precip_sum * 1000.0
            except Exception as e:
                LOGGER.debug(
                    "Failed to compute precipitation accumulation: %s", e
                )

        return result if result else None

    except Exception as e:
        LOGGER.warning(
            "Failed to load weather data from %s: %s",
            storage_path, e
        )
        return None
    finally:
        if ds is not None:
            ds.close()
