"""Fire detection scoring functions for composite likelihood calculation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

from sqlalchemy import text

from api.db import get_engine


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
