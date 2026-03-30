"""Shared weather point-lookup helpers.

Extracted from ``api.fires.scoring`` so that both the fire-scoring pipeline
and the risk-grid module can use the same logic without a cross-package
dependency on fire internals.
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

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RH fire-risk classification
# ---------------------------------------------------------------------------

_RH_THRESHOLDS: list[tuple[float, str]] = [
    (15.0, "critical"),
    (25.0, "elevated"),
]


def classify_rh_fire_risk(rh_pct: float) -> str:
    """Return a fire-risk level based on relative humidity.

    Thresholds follow standard fire-weather convention:
    - <15 %  → ``"critical"``
    - <25 %  → ``"elevated"``
    - ≥25 %  → ``"normal"``
    """
    for threshold, level in _RH_THRESHOLDS:
        if rh_pct < threshold:
            return level
    return "normal"


# ---------------------------------------------------------------------------
# Shared internals
# ---------------------------------------------------------------------------

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


def _ensure_utc(dt: datetime) -> datetime:
    """Normalise a datetime to UTC, treating naive values as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class _WeatherSnapshot:
    """Result of opening a weather run and selecting the nearest grid point."""

    ds: xr.Dataset
    ds_spatial: xr.Dataset   # spatially selected, NOT time-selected
    ds_point: xr.Dataset     # spatially + time-selected (nearest to ref_time)
    ref_time_64: np.datetime64
    run_time: datetime
    storage_path: Path


def _open_weather_for_point(
    *,
    lat: float,
    lon: float,
    ref_time: datetime,
    time_tolerance_hours: float,
) -> _WeatherSnapshot | None:
    """Find the best weather run, open the dataset, and select the nearest point.

    Returns ``None`` (with debug logging) when no qualifying run exists.
    The caller is responsible for closing ``snapshot.ds``.

    The snapshot exposes both ``ds_spatial`` (spatially selected, all time
    steps) and ``ds_point`` (additionally time-selected to the nearest step
    before or at *ref_time*).  Use ``ds_spatial`` when you need multiple
    forecast steps; use ``ds_point`` for the current-conditions shortcut.
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
            lat, lon, ref_time,
        )
        return None

    storage_path = Path(row["storage_path"])
    if not storage_path.is_absolute():
        storage_path = Path.cwd() / storage_path

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
    """Sum total precipitation over the lookback window, converting m → mm."""
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
    """Extract all weather fields from a time-selected spatial point dataset.

    Returns a dict with wind, humidity, temperature, precipitation, and
    rh_fire_risk.  Missing variables are omitted rather than set to null so
    callers can detect partial data.
    """
    ctx: dict[str, Any] = {}

    # --- wind ---
    if "u10" in ds_point_at_time.data_vars and "v10" in ds_point_at_time.data_vars:
        u10 = float(ds_point_at_time["u10"].values)
        v10 = float(ds_point_at_time["v10"].values)
        if not np.isnan(u10) and not np.isnan(v10):
            ctx["wind_speed_ms"] = round(float(np.sqrt(u10**2 + v10**2)), 1)
            # Meteorological convention: direction wind is coming *from*
            ctx["wind_direction_deg"] = round(
                (math.degrees(math.atan2(-u10, -v10)) + 360) % 360, 1
            )

    # --- humidity ---
    if "rh2m" in ds_point_at_time.data_vars:
        rh = float(ds_point_at_time["rh2m"].values)
        if not np.isnan(rh):
            ctx["relative_humidity_pct"] = round(rh, 1)
            ctx["rh_fire_risk"] = classify_rh_fire_risk(rh)

    # --- temperature ---
    if "t2m" in ds_point_at_time.data_vars:
        t2m = float(ds_point_at_time["t2m"].values)
        if not np.isnan(t2m):
            # GFS stores temperature in Kelvin
            ctx["temperature_c"] = round(t2m - 273.15, 1)

    # --- precipitation ---
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
    try:
        snap = _open_weather_for_point(
            lat=lat, lon=lon, ref_time=ref_time,
            time_tolerance_hours=time_tolerance_hours,
        )
    except Exception as exc:
        LOGGER.warning("Failed to open weather data: %s", exc)
        return None

    if snap is None:
        return None

    try:
        result: dict[str, float] = {}

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
        LOGGER.warning(
            "Failed to load weather data from %s: %s", snap.storage_path, exc,
        )
        return None
    finally:
        snap.ds.close()


# ---------------------------------------------------------------------------
# Public API — fire detail panel (rich context dict)
# ---------------------------------------------------------------------------

_RESOLUTION_NOTE = "GFS 0.25\u00b0 \u2014 nearest grid point (~25 km)"

# Variables that the GFS ingest pipeline bias-corrects (affine correction
# fitted against ERA5 reanalysis).  Listed here so the API response can
# transparently communicate which values have been post-processed.
_BIAS_CORRECTED_VARS = ("u10", "v10", "t2m", "rh2m")


def get_weather_context_for_point(
    *,
    lat: float,
    lon: float,
    ref_time: datetime,
    time_tolerance_hours: float = 6.0,
    precip_lookback_hours: float = 24.0,
) -> dict[str, Any] | None:
    """Return a weather-context block suitable for the fire detail response.

    Unlike :func:`get_weather_data_for_point` (which returns a flat dict of
    numeric values for scoring), this function returns the full context
    including wind direction, temperature, source provenance, data-age, bias
    correction metadata, and RH fire-risk classification.

    Returns ``None`` when no qualifying weather run covers the point.
    """
    try:
        snap = _open_weather_for_point(
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

        # --- provenance ---
        run_time_utc = _ensure_utc(snap.run_time)
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

        return ctx

    except Exception as exc:
        LOGGER.warning(
            "Failed to load weather context from %s: %s", snap.storage_path, exc,
        )
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

    Opens the same GFS run as :func:`get_weather_context_for_point` but
    selects additional time steps at *ref_time* + each offset in
    *forecast_offsets_hours*.  Each step carries the same fields as the
    current-conditions block plus ``forecast_hour`` and ``valid_time``.

    Returns ``None`` when no qualifying weather run covers the point.
    Returns a partial list when only some offsets fall within the stored
    forecast horizon (steps beyond the dataset are silently skipped).
    """
    try:
        snap = _open_weather_for_point(
            lat=lat, lon=lon, ref_time=ref_time,
            time_tolerance_hours=time_tolerance_hours,
        )
    except Exception as exc:
        LOGGER.warning("Failed to open weather data for forecast: %s", exc)
        return None

    if snap is None:
        return None

    ref_time_utc = _ensure_utc(ref_time)

    steps: list[dict[str, Any]] = []
    try:
        for offset_h in forecast_offsets_hours:
            target_time = ref_time_utc + timedelta(hours=offset_h)
            target_time_64 = _to_numpy_datetime64(target_time)

            # Select the nearest time step available; skip if no time dimension
            if "time" in snap.ds_spatial.coords:
                ds_at_time = snap.ds_spatial.sel(time=target_time_64, method="nearest")
            else:
                # Dataset has no time dimension — single-step file; only valid for +0
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
        LOGGER.warning(
            "Failed to extract forecast steps from %s: %s", snap.storage_path, exc,
        )
    finally:
        snap.ds.close()

    return steps if steps else None
