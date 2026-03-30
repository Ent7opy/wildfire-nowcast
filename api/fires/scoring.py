"""Fire detection scoring functions for composite likelihood calculation."""

from __future__ import annotations

import logging
import os
from typing import Iterable

from sqlalchemy import text

from api.core.weather import get_weather_data_for_point
from api.db import get_engine

LOGGER = logging.getLogger(__name__)
_DEFAULT_POLICY_VERSION = os.getenv("INDUSTRIAL_MASK_POLICY_VERSION", "").strip()


def _active_industrial_policy(policy_version: str | None = None) -> dict | None:
    where = "active_to IS NULL OR active_to > NOW()"
    params: dict[str, object] = {}
    if policy_version:
        where = "policy_version = :policy_version"
        params["policy_version"] = str(policy_version).strip()
    stmt = text(
        f"""
        SELECT
            policy_version,
            strict_no_go,
            gold_buffer_m,
            silver_buffer_min_m,
            silver_buffer_max_m
        FROM industrial_mask_policies
        WHERE {where}
        ORDER BY active_from DESC, policy_version DESC
        LIMIT 1
        """
    )
    try:
        with get_engine().begin() as conn:
            row = conn.execute(stmt, params).mappings().first()
    except Exception:  # pragma: no cover - safe fallback for pre-migration state
        return None
    if row is None:
        return None
    payload = dict(row)
    required = {
        "policy_version",
        "strict_no_go",
        "gold_buffer_m",
        "silver_buffer_min_m",
        "silver_buffer_max_m",
    }
    if not required.issubset(payload.keys()):
        return None
    return payload


def _legacy_mask_false_sources(
    detection_ids: list[int],
    *,
    radius_m: float,
) -> dict[int, bool]:
    stmt = text("""
        SELECT DISTINCT fd.id AS detection_id
        FROM fire_detections fd
        JOIN industrial_sources ind ON (
            ST_DWithin(fd.geom::geography, ind.geom::geography, :radius_m)
        )
        WHERE fd.id = ANY(:detection_ids)
    """)

    with get_engine().begin() as conn:
        rows = conn.execute(
            stmt,
            {
                "detection_ids": detection_ids,
                "radius_m": float(radius_m),
            },
        ).mappings().all()
    masked = {int(row["detection_id"]): True for row in rows}
    for det_id in detection_ids:
        masked.setdefault(det_id, False)
    return masked


def _policy_mask_false_sources(
    detection_ids: list[int],
    *,
    policy: dict,
    write_audit: bool,
) -> dict[int, bool]:
    stmt = text(
        """
        WITH input_detections AS (
            SELECT id, geom
            FROM fire_detections
            WHERE id = ANY(:detection_ids)
        ),
        no_go AS (
            SELECT DISTINCT d.id AS detection_id
            FROM input_detections d
            JOIN industrial_no_go_zones z
              ON z.is_active
             AND z.policy_version = :policy_version
             AND z.geom && d.geom
             AND ST_Intersects(z.geom, d.geom)
        ),
        candidates AS (
            SELECT
                d.id AS detection_id,
                i.id AS industrial_source_id,
                i.authority_tier,
                ST_Distance(d.geom::geography, i.geom::geography) AS distance_m,
                CASE
                    WHEN i.authority_tier = 'gold' THEN :gold_buffer_m
                    WHEN i.authority_tier = 'silver' THEN LEAST(
                        :silver_buffer_max_m,
                        GREATEST(
                            :silver_buffer_min_m,
                            COALESCE(i.coordinate_precision_m::double precision, :silver_buffer_min_m)
                        )
                    )
                    ELSE 0.0
                END AS applied_buffer_m
            FROM input_detections d
            JOIN industrial_sources i
              ON COALESCE(i.is_active, TRUE)
             AND i.authority_tier IN ('gold', 'silver')
             AND (i.valid_from IS NULL OR i.valid_from <= NOW())
             AND (i.valid_to IS NULL OR i.valid_to >= NOW())
             AND ST_DWithin(
                d.geom::geography,
                i.geom::geography,
                CASE
                    WHEN i.authority_tier = 'gold' THEN :gold_buffer_m
                    WHEN i.authority_tier = 'silver' THEN LEAST(
                        :silver_buffer_max_m,
                        GREATEST(
                            :silver_buffer_min_m,
                            COALESCE(i.coordinate_precision_m::double precision, :silver_buffer_min_m)
                        )
                    )
                    ELSE 0.0
                END
             )
        ),
        ranked AS (
            SELECT
                c.*,
                BOOL_OR(c.authority_tier = 'gold') OVER (PARTITION BY c.detection_id) AS has_gold_candidate,
                ROW_NUMBER() OVER (
                    PARTITION BY c.detection_id
                    ORDER BY
                        CASE c.authority_tier WHEN 'gold' THEN 0 ELSE 1 END,
                        c.distance_m ASC,
                        c.industrial_source_id ASC
                ) AS rn
            FROM candidates c
        ),
        best AS (
            SELECT
                detection_id,
                industrial_source_id,
                authority_tier,
                distance_m,
                applied_buffer_m,
                has_gold_candidate
            FROM ranked
            WHERE rn = 1
        ),
        decisions AS (
            SELECT
                d.id AS detection_id,
                b.industrial_source_id,
                b.authority_tier,
                b.distance_m,
                b.applied_buffer_m,
                CASE
                    WHEN :strict_no_go AND ng.detection_id IS NOT NULL THEN FALSE
                    WHEN b.detection_id IS NULL THEN FALSE
                    WHEN b.authority_tier = 'gold' THEN TRUE
                    WHEN b.authority_tier = 'silver' AND COALESCE(b.has_gold_candidate, FALSE) THEN FALSE
                    WHEN b.authority_tier = 'silver' THEN TRUE
                    ELSE FALSE
                END AS masked,
                CASE
                    WHEN :strict_no_go AND ng.detection_id IS NOT NULL THEN 'no_go_zone'
                    WHEN b.detection_id IS NULL THEN 'no_nearby_source'
                    WHEN b.authority_tier = 'gold' THEN 'gold_match'
                    WHEN b.authority_tier = 'silver' AND COALESCE(b.has_gold_candidate, FALSE) THEN 'silver_suppressed_gold_overlap'
                    WHEN b.authority_tier = 'silver' THEN 'silver_fallback_match'
                    ELSE 'unmasked'
                END AS mask_reason
            FROM input_detections d
            LEFT JOIN no_go ng ON ng.detection_id = d.id
            LEFT JOIN best b ON b.detection_id = d.id
        )
        SELECT
            detection_id,
            industrial_source_id,
            authority_tier,
            distance_m,
            applied_buffer_m,
            masked,
            mask_reason
        FROM decisions
        """
    )

    params = {
        "detection_ids": detection_ids,
        "policy_version": str(policy["policy_version"]),
        "strict_no_go": bool(policy["strict_no_go"]),
        "gold_buffer_m": float(policy["gold_buffer_m"]),
        "silver_buffer_min_m": float(policy["silver_buffer_min_m"]),
        "silver_buffer_max_m": float(policy["silver_buffer_max_m"]),
    }
    with get_engine().begin() as conn:
        rows = conn.execute(stmt, params).mappings().all()
        if write_audit and rows:
            audit_stmt = text(
                """
                INSERT INTO industrial_mask_audit (
                    fire_detection_id,
                    industrial_source_id,
                    policy_version,
                    masked,
                    mask_reason,
                    matched_distance_m,
                    applied_buffer_m,
                    created_at
                ) VALUES (
                    :fire_detection_id,
                    :industrial_source_id,
                    :policy_version,
                    :masked,
                    :mask_reason,
                    :matched_distance_m,
                    :applied_buffer_m,
                    NOW()
                )
                """
            )
            conn.execute(
                audit_stmt,
                [
                    {
                        "fire_detection_id": int(row["detection_id"]),
                        "industrial_source_id": row["industrial_source_id"],
                        "policy_version": str(policy["policy_version"]),
                        "masked": bool(row["masked"]),
                        "mask_reason": str(row["mask_reason"]),
                        "matched_distance_m": row["distance_m"],
                        "applied_buffer_m": row["applied_buffer_m"],
                    }
                    for row in rows
                ],
            )

    masked = {int(row["detection_id"]): bool(row["masked"]) for row in rows}
    for det_id in detection_ids:
        masked.setdefault(det_id, False)
    return masked


def mask_false_sources(
    detections: Iterable[dict],
    *,
    radius_m: float = 500.0,
    policy_version: str | None = None,
    write_audit: bool = True,
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
        - If industrial_sources table is empty or missing, all detections pass through
          unmasked and a warning is logged
    """
    detection_list = list(detections)
    if not detection_list:
        return {}

    detection_ids = [d["id"] for d in detection_list]
    if not detection_ids:
        return {}

    # Check if industrial_sources table exists and has data
    try:
        check_stmt = text("""
            SELECT COUNT(*) as count
            FROM industrial_sources
        """)
        with get_engine().connect() as conn:
            result = conn.execute(check_stmt)
            row = result.mappings().first()
            source_count = row["count"] if row else 0

        if source_count == 0:
            LOGGER.warning(
                "Industrial sources table is empty; all detections pass through unmasked. "
                "Run ingest pipeline to populate industrial_sources table."
            )
            # Return all detections as unmasked
            return {det_id: False for det_id in detection_ids}
    except Exception as e:
        # Table may not exist or other DB error
        LOGGER.warning(
            "Failed to query industrial_sources table; all detections pass through unmasked. "
            "Error: %s",
            e,
        )
        return {det_id: False for det_id in detection_ids}

    effective_policy = _active_industrial_policy(policy_version or _DEFAULT_POLICY_VERSION)
    if effective_policy is None:
        return _legacy_mask_false_sources(detection_ids, radius_m=radius_m)

    try:
        return _policy_mask_false_sources(
            detection_ids,
            policy=effective_policy,
            write_audit=bool(write_audit),
        )
    except Exception as exc:  # pragma: no cover - safe fallback when policy tables are unavailable
        LOGGER.warning(
            "Policy masking failed; falling back to legacy industrial masking. Error: %s",
            exc,
        )
        return _legacy_mask_false_sources(detection_ids, radius_m=radius_m)


def compute_persistence_scores(
    detections: Iterable[dict],
    *,
    spatial_radius_m: float = 750.0,
    time_window_hours: tuple[float, float] = (0.0, 72.0),
    chunk_size: int = 5000,
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
            looking BACKWARD from target detection time. Default (0, 72) means
            all detections from the past 0-72 hours are considered
        chunk_size: Process detections in chunks to avoid memory issues with large batches
            (default 5000, max 10000). Each chunk uses a single database query.

    Returns:
        Dict mapping detection_id → persistence_score in range [0, 1]

    Notes:
        - Uses ST_DWithin for efficient spatial clustering with geometry index
        - Time filtering ensures detections are within reasonable temporal proximity
        - Scores are computed relative to all detections in the database within
          the time window, not just the input batch
        - Large batches are processed in chunks to prevent memory exhaustion
          and avoid overloading the database with massive IN clauses
    """
    # Clamp chunk size to reasonable bounds
    chunk_size = max(100, min(chunk_size, 10000))
    
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

    # Process detections in chunks for memory efficiency with large batches
    scores: dict[int, float] = {}
    total = len(detection_ids)
    
    # Query clusters for each detection using ST_DWithin spatial clustering
    # and time window filtering. Uses server-side cursor for memory efficiency.
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
                                AND t.acq_time
        )
        GROUP BY t.id
    """)

    # Process in chunks to avoid overwhelming the database and memory
    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_ids = detection_ids[chunk_start:chunk_end]
        
        with get_engine().begin() as conn:
            result = conn.execute(
                stmt,
                {
                    "detection_ids": chunk_ids,
                    "radius_m": float(spatial_radius_m),
                    "min_hours": float(min_hours),
                    "max_hours": float(max_hours),
                },
            )
            rows = result.mappings().all()

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
        for det_id in chunk_ids:
            if det_id not in scores:
                scores[det_id] = 0.2

    return scores


def compute_fire_likelihood(
    confidence_score: float,
    persistence_score: float | None,
    landcover_score: float | None,
    weather_score: float | None,
    false_source_masked: bool,
) -> float:
    """Compute composite fire likelihood score from component scores.

    Combines FIRMS confidence (weak prior), persistence filtering, land-cover plausibility,
    weather plausibility, and industrial false-source masking into a single fire likelihood score.

    Scoring logic:
    - If false_source_masked is True: return 0.0 (industrial false positive)
    - Otherwise: weighted combination of component scores
        - confidence_score: 0.2 weight (weak prior from FIRMS)
        - persistence_score: 0.3 weight (spatial-temporal clustering)
        - landcover_score: 0.25 weight (land-cover plausibility)
        - weather_score: 0.25 weight (meteorological plausibility)
    - Missing scores (None) are treated as neutral (0.5) for weighting

    Note on multi-sensor bonus:
        The original task (WN-FIRE-006) suggested a separate ~10% weight for multi-sensor
        detection bonus. This was intentionally omitted because:
        1. Multi-sensor detection is already captured within persistence_score via sensor_count
           in compute_persistence_scores() (see api/fires/scoring.py:~150-165)
        2. The 10% allocation was redistributed to landcover (12.5%) and weather (12.5%),
           increasing their weights from 0.2 to 0.25 each
        3. This simplifies the scoring model while maintaining signal strength from
           multi-sensor observations through the persistence component

    Args:
        confidence_score: Normalized FIRMS confidence in range [0, 1]
        persistence_score: Persistence score from spatial-temporal clustering (optional)
        landcover_score: Land-cover plausibility score (optional)
        weather_score: Weather plausibility score (optional)
        false_source_masked: True if detection is near industrial source

    Returns:
        Composite fire likelihood score in range [0, 1]

    Example:
        >>> compute_fire_likelihood(0.8, 0.9, 1.0, 0.7, False)
        0.855  # High likelihood: good confidence, strong persistence, forest, favorable weather
        >>> compute_fire_likelihood(0.9, 0.9, 0.1, 0.8, False)
        0.675  # Medium: good scores but unlikely land-cover (water/urban)
        >>> compute_fire_likelihood(0.9, 0.9, 0.9, 0.9, True)
        0.0  # Industrial false positive: masked regardless of other scores
    """
    # Industrial false sources get zero likelihood
    if false_source_masked:
        return 0.0

    # Define weights for each component
    # Confidence is a weak prior; persistence and plausibility scores are stronger
    weights = {
        "confidence": 0.2,
        "persistence": 0.3,
        "landcover": 0.25,
        "weather": 0.25,
    }

    # Use neutral score (0.5) for missing components
    scores = {
        "confidence": confidence_score,
        "persistence": persistence_score if persistence_score is not None else 0.5,
        "landcover": landcover_score if landcover_score is not None else 0.5,
        "weather": weather_score if weather_score is not None else 0.5,
    }

    # Compute weighted sum
    likelihood = sum(weights[k] * scores[k] for k in weights.keys())

    # Clamp to [0, 1] range (should already be in range, but defensive)
    return max(0.0, min(1.0, likelihood))


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
        weather_data = get_weather_data_for_point(
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
