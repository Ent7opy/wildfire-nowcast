"""Fire detection scoring functions for composite likelihood calculation."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from sqlalchemy import text

from api.db import get_engine

LOGGER = logging.getLogger(__name__)


def mask_false_sources(
    detections: Iterable[dict],
    *,
    radius_m: float = 500.0,
) -> dict[int, bool]:
    """Identify fire detections near known industrial false-positive sources.

    Queries industrial_sources table and marks detections within radius_m as masked.
    Masked detections should be excluded from default views or assigned fire_likelihood=0.

    Spatial matching logic:
    - Uses ST_DWithin for efficient spatial query with geometry index
    - Default radius: 500m (typical thermal sensor spatial accuracy)
    - Industrial sources include power plants, refineries, steel mills, etc.

    Args:
        detections: Iterable of detection dicts with keys: id, lat, lon
        radius_m: Spatial masking radius in meters (default 500m)

    Returns:
        Dict mapping detection_id → masked (True if near industrial source)

    Notes:
        - Only returns True for masked detections; absent keys mean not masked
        - Relies on industrial_sources table populated via ingest pipeline
    """
    detection_list = list(detections)
    if not detection_list:
        return {}

    detection_ids = [d["id"] for d in detection_list]
    if not detection_ids:
        return {}

    # Query detections within radius_m of any industrial source
    stmt = text("""
        SELECT DISTINCT fd.id AS detection_id
        FROM fire_detections fd
        JOIN industrial_sources ind ON (
            ST_DWithin(fd.geom::geography, ind.geom::geography, :radius_m)
        )
        WHERE fd.id = ANY(:detection_ids)
    """)

    with get_engine().begin() as conn:
        result = conn.execute(
            stmt,
            {
                "detection_ids": detection_ids,
                "radius_m": float(radius_m),
            },
        )
        rows = result.mappings().all()

    # Map masked detections to True
    masked: dict[int, bool] = {}
    for row in rows:
        detection_id = int(row["detection_id"])
        masked[detection_id] = True

    # For detections not near industrial sources, explicitly mark as not masked
    for det_id in detection_ids:
        if det_id not in masked:
            masked[det_id] = False

    return masked


def compute_persistence_scores(
    detections: Iterable[dict],
    *,
    spatial_radius_m: float = 750.0,
    time_window_hours: tuple[float, float] = (24.0, 72.0),
) -> dict[int, float]:
    """Compute persistence scores for fire detections based on spatial-temporal clustering.

    Persistence scoring logic:
    - Groups detections within spatial_radius_m (default 750m) and time_window_hours window
    - Base score increases with cluster size (more detections nearby = higher persistence)
    - Multi-sensor bonus: +0.1 if cluster contains detections from ≥2 different sensors
    - Isolated detections (single detection in cluster) receive score ≤0.2

    Scoring formula:
    - Isolated (n=1): 0.2
    - Small cluster (n=2-3): 0.3-0.5
    - Medium cluster (n=4-9): 0.5-0.7
    - Large cluster (n≥10): 0.7-0.9
    - Multi-sensor bonus: +0.1 (capped at 1.0)

    Args:
        detections: Iterable of detection dicts with keys: id, lat, lon, acq_time, sensor
        spatial_radius_m: Spatial clustering radius in meters (default 750m)
        time_window_hours: Time window for clustering as (min_hours, max_hours) tuple
            Default (24, 72) means detections must be 24-72 hours apart

    Returns:
        Dict mapping detection_id → persistence_score in range [0, 1]

    Notes:
        - Uses ST_DWithin for efficient spatial clustering with geometry index
        - Time filtering ensures detections are within reasonable temporal proximity
        - Scores are computed relative to all detections in the database within
          the time window, not just the input batch
    """
    detection_list = list(detections)
    if not detection_list:
        return {}

    detection_ids = [d["id"] for d in detection_list]
    if not detection_ids:
        return {}

    min_hours, max_hours = time_window_hours
    if min_hours < 0 or max_hours <= min_hours:
        raise ValueError(
            f"Invalid time_window_hours: {time_window_hours}. "
            "Must be (min_hours, max_hours) with 0 ≤ min_hours < max_hours."
        )

    # Query clusters for each detection using ST_DWithin spatial clustering
    # and time window filtering. For each detection, find all nearby detections
    # within the spatial radius and time window.
    stmt = text("""
        WITH target_detections AS (
            SELECT id, geom, acq_time, sensor
            FROM fire_detections
            WHERE id = ANY(:detection_ids)
        )
        SELECT
            t.id AS detection_id,
            COUNT(DISTINCT n.id) AS cluster_size,
            COUNT(DISTINCT n.sensor) AS sensor_count,
            ARRAY_AGG(DISTINCT n.sensor) AS sensors
        FROM target_detections t
        JOIN fire_detections n ON (
            ST_DWithin(t.geom::geography, n.geom::geography, :radius_m)
            AND n.acq_time BETWEEN (t.acq_time - INTERVAL '1 hour' * :max_hours)
                                AND (t.acq_time - INTERVAL '1 hour' * :min_hours)
        )
        GROUP BY t.id
    """)

    with get_engine().begin() as conn:
        result = conn.execute(
            stmt,
            {
                "detection_ids": detection_ids,
                "radius_m": float(spatial_radius_m),
                "min_hours": float(min_hours),
                "max_hours": float(max_hours),
            },
        )
        rows = result.mappings().all()

    scores: dict[int, float] = {}
    for row in rows:
        detection_id = int(row["detection_id"])
        cluster_size = int(row["cluster_size"])
        sensor_count = int(row["sensor_count"])

        # Base score from cluster size
        if cluster_size == 1:
            base_score = 0.2
        elif cluster_size <= 3:
            base_score = 0.3 + (cluster_size - 2) * 0.1
        elif cluster_size <= 9:
            base_score = 0.5 + (cluster_size - 4) * 0.033
        else:
            base_score = min(0.9, 0.7 + (cluster_size - 10) * 0.02)

        # Multi-sensor bonus
        multi_sensor_bonus = 0.1 if sensor_count >= 2 else 0.0

        final_score = min(1.0, base_score + multi_sensor_bonus)
        scores[detection_id] = final_score

    # For detections not in clusters (no nearby detections), assign isolated score
    for det_id in detection_ids:
        if det_id not in scores:
            scores[det_id] = 0.2

    return scores


def compute_weather_plausibility_scores(
    detections: Iterable[dict],
    *,
    high_rh_threshold: float = 70.0,
    low_rh_bonus_threshold: float = 40.0,
    precip_lookback_hours: float = 72.0,
    heavy_precip_threshold_mm: float = 10.0,
    moderate_wind_threshold_ms: float = 3.0,
    time_tolerance_hours: float = 6.0,
) -> dict[int, float]:
    """Compute weather plausibility scores for fire detections.

    Weather plausibility scoring logic:
    - Penalizes detections in meteorologically unfavorable conditions
    - Boosts detections in fire-prone weather conditions
    - Uses weather data from ingested weather runs (GFS NetCDF files)

    Scoring rules:
    - Base score: 0.5 (neutral)
    - Penalties:
      - High RH (>70%): -0.3 (very wet conditions suppress fires)
      - Recent heavy precipitation (>10mm in 48-72h): -0.2 (wet fuel)
    - Bonuses:
      - Low RH (<40%): +0.2 (dry conditions favor fires)
      - Moderate/high wind (>3 m/s): +0.1 (wind spreads fires)
    - Score clamped to [0.1, 1.0] range

    Args:
        detections: Iterable of detection dicts with keys: id, lat, lon, acq_time
        high_rh_threshold: RH percentage above which to penalize (default 70%)
        low_rh_bonus_threshold: RH percentage below which to boost (default 40%)
        precip_lookback_hours: Hours to look back for precipitation history (default 72h)
        heavy_precip_threshold_mm: Precipitation threshold in mm for penalty (default 10mm)
        moderate_wind_threshold_ms: Wind speed threshold in m/s for bonus (default 3 m/s)
        time_tolerance_hours: Hours of tolerance for weather data matching (default 6h)

    Returns:
        Dict mapping detection_id → weather_plausibility_score in range [0.1, 1.0]

    Notes:
        - Falls back to neutral score (0.5) if weather data is unavailable
        - Uses nearest-neighbor interpolation for spatial and temporal matching
        - Weather variables: rh2m (relative humidity), tp (total precipitation), u10/v10 (wind)
    """
    detection_list = list(detections)
    if not detection_list:
        return {}

    detection_ids = [d["id"] for d in detection_list]
    if not detection_ids:
        return {}

    scores: dict[int, float] = {}

    # Group detections by time window to minimize weather file loading
    # For now, process each detection individually (can optimize later if needed)
    for det in detection_list:
        det_id = int(det["id"])
        lat = float(det["lat"])
        lon = float(det["lon"])
        acq_time = det["acq_time"]

        # Query weather data for this detection
        weather_data = _get_weather_data_for_point(
            lat=lat,
            lon=lon,
            ref_time=acq_time,
            time_tolerance_hours=time_tolerance_hours,
            precip_lookback_hours=precip_lookback_hours,
        )

        if weather_data is None:
            # No weather data available: use neutral score
            scores[det_id] = 0.5
            continue

        # Extract weather variables
        rh = weather_data.get("rh2m")
        precip_recent = weather_data.get("precip_recent_mm")
        wind_speed = weather_data.get("wind_speed_ms")

        # Base score: neutral
        score = 0.5

        # Apply penalties
        if rh is not None and rh > high_rh_threshold:
            score -= 0.3  # Very wet conditions suppress fires

        if precip_recent is not None and precip_recent > heavy_precip_threshold_mm:
            score -= 0.2  # Recent heavy rain wets fuel

        # Apply bonuses
        if rh is not None and rh < low_rh_bonus_threshold:
            score += 0.2  # Dry conditions favor fires

        if wind_speed is not None and wind_speed > moderate_wind_threshold_ms:
            score += 0.1  # Wind spreads fires

        # Clamp to [0.1, 1.0] range
        score = max(0.1, min(1.0, score))
        scores[det_id] = score

    return scores


def _get_weather_data_for_point(
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
    # Query weather runs that cover this point and time
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

    if not storage_path.exists():
        LOGGER.warning(
            "Weather run %s file missing at %s",
            row["id"], storage_path
        )
        return None

    try:
        # Load weather dataset
        ds = xr.open_dataset(storage_path)

        # Select nearest point spatially
        ds_point = ds.sel(lat=lat, lon=lon, method="nearest")

        # Select time closest to ref_time
        if "time" in ds_point.coords:
            # Convert ref_time to numpy datetime64 (naive UTC)
            ref_time_utc = ref_time.replace(tzinfo=None) if ref_time.tzinfo else ref_time
            ref_time_64 = np.datetime64(ref_time_utc, "ns")
            ds_point = ds_point.sel(time=ref_time_64, method="nearest")

        # Extract weather variables
        result: dict[str, float] = {}

        # Relative humidity at 2m
        if "rh2m" in ds_point.data_vars:
            rh_val = float(ds_point["rh2m"].values)
            if not np.isnan(rh_val):
                result["rh2m"] = rh_val

        # Wind speed (compute from u10 and v10 components)
        if "u10" in ds_point.data_vars and "v10" in ds_point.data_vars:
            u10_val = float(ds_point["u10"].values)
            v10_val = float(ds_point["v10"].values)
            if not np.isnan(u10_val) and not np.isnan(v10_val):
                wind_speed = np.sqrt(u10_val**2 + v10_val**2)
                result["wind_speed_ms"] = float(wind_speed)

        # Precipitation accumulation (if available)
        # Note: GFS may not always have precipitation data in all runs
        # If 'tp' (total precipitation) is available, accumulate recent values
        if "tp" in ds_point.data_vars and "time" in ds.coords:
            try:
                # Select time range for precipitation lookback
                precip_start = ref_time - timedelta(hours=precip_lookback_hours)
                precip_start_64 = np.datetime64(precip_start.replace(tzinfo=None), "ns")
                ref_time_64 = np.datetime64(ref_time.replace(tzinfo=None), "ns")

                ds_precip = ds.sel(lat=lat, lon=lon, method="nearest")
                ds_precip = ds_precip.sel(time=slice(precip_start_64, ref_time_64))

                if "tp" in ds_precip.data_vars and len(ds_precip.time) > 0:
                    # Sum precipitation over the lookback period
                    precip_sum = float(ds_precip["tp"].sum().values)
                    if not np.isnan(precip_sum):
                        # Convert from meters to mm (GFS typically outputs in meters)
                        result["precip_recent_mm"] = precip_sum * 1000.0
            except Exception as e:
                LOGGER.debug(
                    "Failed to compute precipitation accumulation: %s", e
                )

        ds.close()
        return result if result else None

    except Exception as e:
        LOGGER.warning(
            "Failed to load weather data from %s: %s",
            storage_path, e
        )
        return None
