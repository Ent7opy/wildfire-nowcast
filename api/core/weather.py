"""Shared weather point-lookup helpers.

Primary path: query the ``weather_point_cache`` DB table (populated by
``ingest.weather_point_ingest``).  Fallback: open a NetCDF file from
``weather_runs.storage_path`` (covers JIT-produced per-AOI files).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from sqlalchemy import text

from api.db import get_engine
from ingest.weather_repository import snap_to_gfs_grid

LOGGER = logging.getLogger(__name__)

_RESOLUTION_NOTE = "GFS 0.25\u00b0 \u2014 nearest grid point (~25 km)"

_BIAS_CORRECTED_VARS = ("u10", "v10", "t2m", "rh2m")

# ---------------------------------------------------------------------------
# RH fire-risk classification
# ---------------------------------------------------------------------------

_RH_THRESHOLDS: list[tuple[float, str]] = [
    (15.0, "critical"),
    (25.0, "elevated"),
]


def classify_rh_fire_risk(rh_pct: float) -> str:
    """Return a fire-risk level based on relative humidity."""
    for threshold, level in _RH_THRESHOLDS:
        if rh_pct < threshold:
            return level
    return "normal"


# ---------------------------------------------------------------------------
# Shared internals
# ---------------------------------------------------------------------------

def _to_numpy_datetime64(dt: datetime) -> np.datetime64:
    """Convert a datetime to numpy datetime64 with proper UTC handling."""
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)
    return np.datetime64(dt_utc.replace(tzinfo=None), "ms")


def _ensure_utc(dt: datetime) -> datetime:
    """Normalise a datetime to UTC, treating naive values as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# DB point-cache queries (primary path)
# ---------------------------------------------------------------------------

def _query_point_cache(
    *,
    lat: float,
    lon: float,
    ref_time: datetime,
    time_tolerance_hours: float,
    forecast_hour: int = 0,
) -> dict[str, Any] | None:
    """Query ``weather_point_cache`` for a single GFS grid point + forecast hour.

    Returns a dict with keys ``u10, v10, t2m, rh2m, tp, run_time`` or ``None``
    if no matching row exists within the tolerance window.
    """
    lat_g, lon_g = snap_to_gfs_grid(lat, lon)

    stmt = text("""
        SELECT wpc.u10, wpc.v10, wpc.t2m, wpc.rh2m, wpc.tp,
               wr.run_time
        FROM weather_point_cache wpc
        JOIN weather_runs wr ON wr.id = wpc.run_id
        WHERE wpc.lat_grid = :lat_g AND wpc.lon_grid = :lon_g
          AND wpc.forecast_hour = :fh
          AND wr.status = 'completed'
          AND wr.run_time <= :ref_time
          AND wr.run_time >= :ref_time - INTERVAL '1 hour' * :tol
        ORDER BY wr.run_time DESC
        LIMIT 1
    """)

    try:
        with get_engine().connect() as conn:
            row = conn.execute(
                stmt,
                {
                    "lat_g": lat_g,
                    "lon_g": lon_g,
                    "fh": forecast_hour,
                    "ref_time": ref_time,
                    "tol": time_tolerance_hours,
                },
            ).mappings().first()

        if not row:
            return None

        # Validate expected columns exist (guard against schema mismatches
        # or mocked queries that return weather_runs columns instead).
        if "u10" not in row:
            return None

        return {
            "u10": row["u10"],
            "v10": row["v10"],
            "t2m": row["t2m"],
            "rh2m": row["rh2m"],
            "tp": row["tp"],
            "run_time": row["run_time"],
        }
    except Exception:
        # Point cache table may not exist yet (pre-migration) or query may
        # fail for other reasons.  Fall through to file-based path.
        LOGGER.debug("Point cache query failed; falling through to file path.", exc_info=True)
        return None


def _build_fields_from_cache_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a point-cache dict into the rich fields dict used by the context API."""
    ctx: dict[str, Any] = {}

    u10 = row.get("u10")
    v10 = row.get("v10")
    if u10 is not None and v10 is not None:
        ctx["wind_speed_ms"] = round(float(np.sqrt(u10**2 + v10**2)), 1)
        ctx["wind_direction_deg"] = round(
            (math.degrees(math.atan2(-u10, -v10)) + 360) % 360, 1
        )

    rh = row.get("rh2m")
    if rh is not None:
        ctx["relative_humidity_pct"] = round(rh, 1)
        ctx["rh_fire_risk"] = classify_rh_fire_risk(rh)

    t2m = row.get("t2m")
    if t2m is not None:
        ctx["temperature_c"] = round(t2m - 273.15, 1)

    tp = row.get("tp")
    if tp is not None:
        ctx["precip_mm_24h"] = round(tp * 1000.0, 1)

    return ctx


def _attach_provenance(ctx: dict[str, Any], run_time: datetime, ref_time: datetime) -> None:
    """Add source provenance and bias-correction metadata to a context dict in-place."""
    run_time_utc = _ensure_utc(run_time)
    ref_time_utc = _ensure_utc(ref_time)
    ctx["source_run_time"] = run_time_utc.isoformat()
    ctx["data_age_hours"] = round(
        (ref_time_utc - run_time_utc).total_seconds() / 3600, 1
    )
    ctx["resolution_note"] = _RESOLUTION_NOTE
    ctx["bias_correction"] = {
        "applied": True,
        "method": "affine (fitted against ERA5 reanalysis)",
        "variables": list(_BIAS_CORRECTED_VARS),
    }


# ---------------------------------------------------------------------------
# NetCDF file-based fallback (for JIT-produced per-AOI files)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _WeatherSnapshot:
    """Result of opening a weather run and selecting the nearest grid point."""

    ds: xr.Dataset
    ds_spatial: xr.Dataset
    ds_point: xr.Dataset
    ref_time_64: np.datetime64
    run_time: datetime
    storage_path: Path


def _open_weather_file_for_point(
    *,
    lat: float,
    lon: float,
    ref_time: datetime,
    time_tolerance_hours: float,
) -> _WeatherSnapshot | None:
    """Fallback: find a NetCDF-backed weather_runs row and open the file.

    This covers JIT-produced per-AOI files from spread forecast runs.
    Skips rows with empty storage_path (point-cache runs).
    """
    stmt = text("""
        SELECT id, storage_path, run_time
        FROM weather_runs
        WHERE status = 'completed'
          AND storage_path != ''
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
        return None

    storage_path = Path(row["storage_path"])
    if not storage_path.is_absolute():
        storage_path = Path.cwd() / storage_path

    if not storage_path.exists():
        LOGGER.debug("Weather file missing: %s", storage_path)
        return None

    ds = xr.open_dataset(storage_path)
    ds_spatial = ds.sel(lat=lat, lon=lon, method="nearest")

    ref_time_64 = _to_numpy_datetime64(ref_time)
    if "time" in ds_spatial.coords:
        ds_point = ds_spatial.sel(time=ref_time_64, method="nearest")
    else:
        ds_point = ds_spatial

    return _WeatherSnapshot(
        ds=ds,
        ds_spatial=ds_spatial,
        ds_point=ds_point,
        ref_time_64=ref_time_64,
        run_time=row["run_time"],
        storage_path=storage_path,
    )


def _extract_precip_mm(
    ds_point: xr.Dataset,
    ds: xr.Dataset,
    ref_time: datetime,
    ref_time_64: np.datetime64,
    lookback_hours: float,
) -> float | None:
    """Sum total precipitation over the lookback window, converting m -> mm."""
    if "tp" not in ds_point.data_vars or "time" not in ds.coords:
        return None
    try:
        precip_start_64 = _to_numpy_datetime64(
            ref_time - timedelta(hours=lookback_hours)
        )
        ds_precip = ds_point.sel(time=slice(precip_start_64, ref_time_64))
        if "tp" in ds_precip.data_vars and len(ds_precip.time) > 0:
            precip_sum = float(ds_precip["tp"].sum().values)
            if not np.isnan(precip_sum):
                return precip_sum * 1000.0
    except Exception as exc:
        LOGGER.debug("Failed to compute precipitation accumulation: %s", exc)
    return None


def _extract_weather_fields(
    ds_point_at_time: xr.Dataset,
    ds_spatial: xr.Dataset,
    target_time: datetime,
    target_time_64: np.datetime64,
    precip_lookback_hours: float,
) -> dict[str, Any]:
    """Extract all weather fields from a time-selected spatial point dataset."""
    ctx: dict[str, Any] = {}

    if "u10" in ds_point_at_time.data_vars and "v10" in ds_point_at_time.data_vars:
        u10 = float(ds_point_at_time["u10"].values)
        v10 = float(ds_point_at_time["v10"].values)
        if not np.isnan(u10) and not np.isnan(v10):
            ctx["wind_speed_ms"] = round(float(np.sqrt(u10**2 + v10**2)), 1)
            ctx["wind_direction_deg"] = round(
                (math.degrees(math.atan2(-u10, -v10)) + 360) % 360, 1
            )

    if "rh2m" in ds_point_at_time.data_vars:
        rh = float(ds_point_at_time["rh2m"].values)
        if not np.isnan(rh):
            ctx["relative_humidity_pct"] = round(rh, 1)
            ctx["rh_fire_risk"] = classify_rh_fire_risk(rh)

    if "t2m" in ds_point_at_time.data_vars:
        t2m = float(ds_point_at_time["t2m"].values)
        if not np.isnan(t2m):
            ctx["temperature_c"] = round(t2m - 273.15, 1)

    precip = _extract_precip_mm(
        ds_point_at_time, ds_spatial, target_time, target_time_64,
        precip_lookback_hours,
    )
    if precip is not None:
        ctx["precip_mm_24h"] = round(precip, 1)

    return ctx


# ---------------------------------------------------------------------------
# Public API — scoring (flat numeric dict)
# ---------------------------------------------------------------------------

def get_weather_data_for_point(
    *,
    lat: float,
    lon: float,
    ref_time: datetime,
    time_tolerance_hours: float,
    precip_lookback_hours: float,
) -> dict[str, float] | None:
    """Query weather data for a specific point and time.

    Tries the DB point cache first (fast, no file I/O).  Falls back to
    opening a NetCDF file if the cache has no data for this location.

    Returns:
        Dict with weather variables or None if data unavailable:
        - rh2m: Relative humidity at 2m (%)
        - precip_recent_mm: Recent precipitation accumulation (mm)
        - wind_speed_ms: Wind speed (m/s)
    """
    # ── 1. DB point cache (primary) ──────────────────────────────────────
    cached = _query_point_cache(
        lat=lat, lon=lon, ref_time=ref_time,
        time_tolerance_hours=time_tolerance_hours,
    )
    if cached is not None:
        result: dict[str, float] = {}
        if cached.get("rh2m") is not None:
            result["rh2m"] = cached["rh2m"]
        u10, v10 = cached.get("u10"), cached.get("v10")
        if u10 is not None and v10 is not None:
            result["wind_speed_ms"] = float(np.sqrt(u10**2 + v10**2))
        tp = cached.get("tp")
        if tp is not None:
            result["precip_recent_mm"] = tp * 1000.0
        return result if result else None

    # ── 2. File-based fallback (JIT per-AOI files) ───────────────────────
    try:
        snap = _open_weather_file_for_point(
            lat=lat, lon=lon, ref_time=ref_time,
            time_tolerance_hours=time_tolerance_hours,
        )
    except Exception as exc:
        LOGGER.warning("Failed to open weather file fallback: %s", exc)
        return None

    if snap is None:
        return None

    try:
        result = {}
        if "rh2m" in snap.ds_point.data_vars:
            rh_val = float(snap.ds_point["rh2m"].values)
            if not np.isnan(rh_val):
                result["rh2m"] = rh_val
        if "u10" in snap.ds_point.data_vars and "v10" in snap.ds_point.data_vars:
            u10_val = float(snap.ds_point["u10"].values)
            v10_val = float(snap.ds_point["v10"].values)
            if not np.isnan(u10_val) and not np.isnan(v10_val):
                result["wind_speed_ms"] = float(np.sqrt(u10_val**2 + v10_val**2))
        precip = _extract_precip_mm(
            snap.ds_point, snap.ds, ref_time, snap.ref_time_64,
            precip_lookback_hours,
        )
        if precip is not None:
            result["precip_recent_mm"] = precip
        return result if result else None
    except Exception as exc:
        LOGGER.warning("Failed to load weather data from %s: %s", snap.storage_path, exc)
        return None
    finally:
        snap.ds.close()


# ---------------------------------------------------------------------------
# Public API — fire detail panel (rich context dict)
# ---------------------------------------------------------------------------

def get_weather_context_for_point(
    *,
    lat: float,
    lon: float,
    ref_time: datetime,
    time_tolerance_hours: float = 6.0,
    precip_lookback_hours: float = 24.0,
) -> dict[str, Any] | None:
    """Return a weather-context block suitable for the fire detail response.

    Tries the DB point cache first.  Falls back to NetCDF file if needed.
    """
    # ── 1. DB point cache (primary) ──────────────────────────────────────
    cached = _query_point_cache(
        lat=lat, lon=lon, ref_time=ref_time,
        time_tolerance_hours=time_tolerance_hours,
    )
    if cached is not None:
        ctx = _build_fields_from_cache_row(cached)
        if not ctx:
            return None
        _attach_provenance(ctx, cached["run_time"], ref_time)
        return ctx

    # ── 2. File-based fallback ───────────────────────────────────────────
    try:
        snap = _open_weather_file_for_point(
            lat=lat, lon=lon, ref_time=ref_time,
            time_tolerance_hours=time_tolerance_hours,
        )
    except Exception as exc:
        LOGGER.warning("Failed to open weather data: %s", exc)
        return None

    if snap is None:
        return None

    try:
        ctx = _extract_weather_fields(
            snap.ds_point, snap.ds_spatial, ref_time, snap.ref_time_64,
            precip_lookback_hours,
        )
        if not ctx:
            return None
        _attach_provenance(ctx, snap.run_time, ref_time)
        return ctx
    except Exception as exc:
        LOGGER.warning("Failed to load weather context from %s: %s", snap.storage_path, exc)
        return None
    finally:
        snap.ds.close()


def get_weather_forecast_for_point(
    *,
    lat: float,
    lon: float,
    ref_time: datetime,
    forecast_offsets_hours: tuple[int, ...] = (6, 12),
    time_tolerance_hours: float = 6.0,
    precip_lookback_hours: float = 24.0,
) -> list[dict[str, Any]] | None:
    """Return near-term forecast steps for the fire detail response.

    Tries the DB point cache for each offset.  Falls back to NetCDF file.
    """
    ref_time_utc = _ensure_utc(ref_time)

    # ── 1. DB point cache (primary) ──────────────────────────────────────
    steps: list[dict[str, Any]] = []
    for offset_h in forecast_offsets_hours:
        cached = _query_point_cache(
            lat=lat, lon=lon, ref_time=ref_time,
            time_tolerance_hours=time_tolerance_hours,
            forecast_hour=offset_h,
        )
        if cached is not None:
            fields = _build_fields_from_cache_row(cached)
            if fields:
                target_time = ref_time_utc + timedelta(hours=offset_h)
                fields["forecast_hour"] = offset_h
                fields["valid_time"] = target_time.isoformat()
                steps.append(fields)

    if steps:
        return steps

    # ── 2. File-based fallback ───────────────────────────────────────────
    try:
        snap = _open_weather_file_for_point(
            lat=lat, lon=lon, ref_time=ref_time,
            time_tolerance_hours=time_tolerance_hours,
        )
    except Exception as exc:
        LOGGER.warning("Failed to open weather data for forecast: %s", exc)
        return None

    if snap is None:
        return None

    try:
        for offset_h in forecast_offsets_hours:
            target_time = ref_time_utc + timedelta(hours=offset_h)
            target_time_64 = _to_numpy_datetime64(target_time)

            if "time" in snap.ds_spatial.coords:
                ds_at_time = snap.ds_spatial.sel(time=target_time_64, method="nearest")
            else:
                break

            fields = _extract_weather_fields(
                ds_at_time, snap.ds_spatial, target_time, target_time_64,
                precip_lookback_hours,
            )
            if not fields:
                continue

            fields["forecast_hour"] = offset_h
            fields["valid_time"] = target_time.isoformat()
            steps.append(fields)

    except Exception as exc:
        LOGGER.warning("Failed to extract forecast steps from %s: %s", snap.storage_path, exc)
    finally:
        snap.ds.close()

    return steps if steps else None
