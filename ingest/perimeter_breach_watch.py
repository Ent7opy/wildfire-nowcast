"""Perimeter breach (spot fire) watch.

Detects clusters of confirmed fire detections that lie outside a known fire
perimeter — a leading indicator of spot fires or breakovers.

Rules (from NOTIFICATION_CONTRACTS.md):
  - Lookback window : 6 hours
  - Detection filter: is_noise=False, false_source_masked=False, denoised_score >= 0.7
  - Candidate zone  : outside perimeter AND within ~50 km (0.5°) of boundary
  - Edge filter     : >= 500 m (0.004°) from boundary (suppress fringe noise)
  - Cluster         : detections within 1 km of each other
  - Alert threshold : cluster size >= 3

Severity is always "critical" (spot fires are high-priority events).
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from api.notifications import notify

LOGGER = logging.getLogger(__name__)

# Spatial thresholds (degrees, WGS-84 approximate)
_CANDIDATE_RADIUS_DEG = 0.5       # ~50 km
_MIN_DIST_DEG = 0.004             # ~500 m minimum distance outside perimeter
_CLUSTER_RADIUS_DEG = 0.01        # ~1 km grouping radius
_DENOISED_SCORE_MIN = 0.7
_LOOKBACK_HOURS = 6
_MIN_CLUSTER_SIZE = 3

# Perimeter age guard — perimeters older than this are stale; skip breach check.
# science_grade: consider per-source freshness SLAs.
_MAX_PERIMETER_AGE_HOURS: int = int(os.environ.get("PERIMETER_MAX_AGE_HOURS", "48"))

# Module-level breach state tracker.
# best-effort; resets on restart — science_grade: persist to DB.
_active_breaches: dict[str, dict] = {}
# Key: source_id
# Value: {"first_detected_at": datetime, "last_seen_at": datetime,
#         "cluster_lat": float, "cluster_lon": float}


def reset_breach_state(source_id: str) -> None:
    """Remove a single source_id from the active breach tracker.

    Intended for testing and manual operator resets.
    """
    _active_breaches.pop(source_id, None)


def get_active_breaches() -> dict:
    """Return a shallow copy of the current active breach state.

    Intended for inspection/debugging.
    """
    return dict(_active_breaches)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two (lat, lon) points."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _cluster_detections(
    detections: list[dict[str, Any]],
    radius_deg: float,
) -> list[list[dict[str, Any]]]:
    """Greedy single-linkage clustering by Euclidean degree distance.

    Returns a list of clusters, each cluster being a list of detection dicts
    with keys 'lat' and 'lon'.  Simple O(n²) implementation — detection counts
    per perimeter are expected to be small (< 200).
    """
    unassigned = list(detections)
    clusters: list[list[dict[str, Any]]] = []

    while unassigned:
        seed = unassigned.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            remaining: list[dict[str, Any]] = []
            for det in unassigned:
                # Check distance to any existing cluster member
                in_cluster = False
                for member in cluster:
                    dlat = det["lat"] - member["lat"]
                    dlon = det["lon"] - member["lon"]
                    if math.sqrt(dlat**2 + dlon**2) <= radius_deg:
                        in_cluster = True
                        break
                if in_cluster:
                    cluster.append(det)
                    changed = True
                else:
                    remaining.append(det)
            unassigned = remaining
        clusters.append(cluster)

    return clusters


def check_perimeter_breach(perimeter: dict[str, Any], session: Any) -> dict[str, Any] | None:
    """Check for spot fire clusters outside a single perimeter.

    Args:
        perimeter: Dict with keys id, source_id, fire_name, geom (WKT/GeoJSON),
                   source, acres.
        session:   SQLAlchemy session.

    Returns:
        Breach info dict if a qualifying cluster is found, else None.
    """
    source_id: str = perimeter["source_id"]
    fire_name: str = perimeter["fire_name"]

    try:
        now = datetime.now(timezone.utc)
        since = now - timedelta(hours=_LOOKBACK_HOURS)

        # Accept WKT string or a GeoJSON dict for the perimeter geometry.
        geom_input = perimeter["geom"]
        if isinstance(geom_input, dict):
            # GeoJSON → use ST_GeomFromGeoJSON
            geom_sql = "ST_GeomFromGeoJSON(:geom)"
            geom_param = {"geom": str(geom_input).replace("'", '"')}
        else:
            # Assume WKT
            geom_sql = "ST_GeomFromText(:geom, 4326)"
            geom_param = {"geom": geom_input}

        # Build parameter dict for the query
        params: dict[str, Any] = {
            "since": since,
            "score_min": _DENOISED_SCORE_MIN,
            "candidate_radius": _CANDIDATE_RADIUS_DEG,
            "min_dist": _MIN_DIST_DEG,
            **geom_param,
        }

        # Inject geom_sql into the query string (safe — not user input)
        sql = f"""
            SELECT
                fd.id,
                ST_Y(fd.geom) AS lat,
                ST_X(fd.geom) AS lon,
                fd.denoised_score,
                ST_Distance(fd.geom, {geom_sql}) AS dist_deg,
                MAX(fd.acq_time) OVER () AS max_acq_time
            FROM fire_detections fd
            WHERE
                fd.acq_time >= :since
                AND fd.is_noise = false
                AND fd.false_source_masked = false
                AND fd.denoised_score >= :score_min
                AND ST_DWithin(fd.geom, {geom_sql}, :candidate_radius)
                AND NOT ST_Within(fd.geom, {geom_sql})
                AND ST_Distance(fd.geom, {geom_sql}) >= :min_dist
        """

        from sqlalchemy import text  # local import to match ingest style

        rows = session.execute(text(sql), params).fetchall()

        if not rows:
            LOGGER.info(
                "perimeter_breach_watch: no candidate detections outside perimeter source_id=%s",
                source_id,
            )
            # No qualifying cluster found — resolve any active breach for this source.
            if source_id in _active_breaches:
                _active_breaches.pop(source_id)
                LOGGER.info(
                    "perimeter_breach_watch: breach resolved for source_id=%s", source_id
                )
            return None

        # Pull max_acq_time from the first row (window function is identical across rows).
        max_acq_time: datetime = rows[0].max_acq_time
        if max_acq_time.tzinfo is None:
            max_acq_time = max_acq_time.replace(tzinfo=timezone.utc)
        lag_min = int((now - max_acq_time).total_seconds() / 60)

        detections = [
            {"lat": row.lat, "lon": row.lon, "denoised_score": row.denoised_score, "dist_deg": row.dist_deg}
            for row in rows
        ]

        clusters = _cluster_detections(detections, _CLUSTER_RADIUS_DEG)

        for cluster in clusters:
            if len(cluster) < _MIN_CLUSTER_SIZE:
                continue

            # Compute cluster centroid
            lat = sum(d["lat"] for d in cluster) / len(cluster)
            lon = sum(d["lon"] for d in cluster) / len(cluster)
            max_score = max(d["denoised_score"] for d in cluster)
            # Distance: use the minimum dist_deg in the cluster → closest point
            min_dist_deg = min(d["dist_deg"] for d in cluster)
            # Convert degrees to km (approximate: 1° ≈ 111 km)
            dist_km = min_dist_deg * 111.0
            n_detections = len(cluster)

            # Data-freshness suffix for notification body.
            data_freshness_suffix = (
                f" (satellite data as of {max_acq_time.strftime('%H:%M UTC')},"
                f" approx. {lag_min}min ago)"
            )

            # Determine NEW vs ONGOING breach.
            if source_id not in _active_breaches:
                # First detection for this source.
                _active_breaches[source_id] = {
                    "first_detected_at": now,
                    "last_seen_at": now,
                    "cluster_lat": lat,
                    "cluster_lon": lon,
                }
                title = f"NEW: Spot fire detected outside {fire_name} perimeter"
                body = (
                    f"Cluster of {n_detections} confirmed detections found "
                    f"{dist_km:.1f}km outside perimeter boundary — "
                    f"possible spot fire or breakover."
                    f"{data_freshness_suffix}"
                )
                LOGGER.warning(
                    "perimeter_breach_watch: NEW spot fire cluster outside %s "
                    "source_id=%s n=%d dist_km=%.1f max_score=%.3f",
                    fire_name, source_id, n_detections, dist_km, max_score,
                )
            else:
                # Ongoing breach — update last_seen and report duration.
                entry = _active_breaches[source_id]
                first_detected_at: datetime = entry["first_detected_at"]
                duration_hours = (now - first_detected_at).total_seconds() / 3600.0
                entry["last_seen_at"] = now
                entry["cluster_lat"] = lat
                entry["cluster_lon"] = lon
                title = f"ONGOING: Spot fire persists outside {fire_name} perimeter"
                body = (
                    f"Cluster of {n_detections} confirmed detections found "
                    f"{dist_km:.1f}km outside perimeter boundary — "
                    f"possible spot fire or breakover. "
                    f"First detected {duration_hours:.1f}h ago, still active."
                    f"{data_freshness_suffix}"
                )
                LOGGER.warning(
                    "perimeter_breach_watch: ONGOING spot fire cluster outside %s "
                    "source_id=%s n=%d dist_km=%.1f max_score=%.3f duration_h=%.1f",
                    fire_name, source_id, n_detections, dist_km, max_score, duration_hours,
                )

            notify(
                f"perimeter_breach:{source_id}",
                title=title,
                body=body,
                severity="critical",
                denoised_score=max_score,
                aoi_id=None,
                fire_name=fire_name,
                detection_count=n_detections,
                cluster_lat=lat,
                cluster_lon=lon,
                distance_km=dist_km,
                data_as_of=max_acq_time.isoformat(),
                data_lag_minutes=lag_min,
            )

            return {
                "source_id": source_id,
                "fire_name": fire_name,
                "cluster_lat": lat,
                "cluster_lon": lon,
                "detection_count": n_detections,
                "max_denoised_score": max_score,
                "distance_km": dist_km,
                "data_as_of": max_acq_time,
            }

        # All clusters were below the threshold — treat as "no qualifying cluster".
        LOGGER.info(
            "perimeter_breach_watch: no qualifying cluster (all < %d) for source_id=%s",
            _MIN_CLUSTER_SIZE, source_id,
        )
        if source_id in _active_breaches:
            _active_breaches.pop(source_id)
            LOGGER.info(
                "perimeter_breach_watch: breach resolved for source_id=%s", source_id
            )
        return None

    except Exception:
        LOGGER.exception(
            "perimeter_breach_watch: error checking perimeter source_id=%s", source_id
        )
        return None


def run_perimeter_breach_checks(session: Any) -> list[dict[str, Any]]:
    """Check all active perimeters for spot fire clusters.

    Queries fire_perimeters where fire_end IS NULL or fire_end > now(), then
    calls check_perimeter_breach on fresh perimeters only.  Stale perimeters
    (created_at older than PERIMETER_MAX_AGE_HOURS) emit a warning notification
    instead.

    Returns:
        List of breach dicts for perimeters where a qualifying cluster was found.
    """
    try:
        from sqlalchemy import text  # local import to match ingest style

        now = datetime.now(timezone.utc)
        sql = """
            SELECT
                id,
                source_id,
                fire_name,
                ST_AsText(geom) AS geom,
                source,
                acres,
                created_at
            FROM fire_perimeters
            WHERE fire_end IS NULL OR fire_end > :now
        """
        rows = session.execute(text(sql), {"now": now}).fetchall()

        if not rows:
            LOGGER.info("perimeter_breach_watch: no active perimeters to check")
            return []

        LOGGER.info("perimeter_breach_watch: checking %d active perimeter(s)", len(rows))

        results: list[dict[str, Any]] = []
        for row in rows:
            source_id: str = row.source_id
            fire_name: str = row.fire_name
            created_at: datetime = row.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            age_hours = (now - created_at).total_seconds() / 3600.0

            if age_hours >= _MAX_PERIMETER_AGE_HOURS:
                # Stale perimeter — emit warning, skip breach check.
                LOGGER.warning(
                    "perimeter_breach_watch: stale perimeter source_id=%s age=%.0fh — skipping breach check",
                    source_id, age_hours,
                )
                notify(
                    f"perimeter_stale:{source_id}",
                    title=f"{fire_name} perimeter data is stale",
                    body=(
                        f"Perimeter last updated {age_hours:.0f}h ago — "
                        f"spot fire checks suspended until data refreshes."
                    ),
                    severity="warning",
                    aoi_id=None,
                    fire_name=fire_name,
                    perimeter_age_hours=age_hours,
                )
                continue

            perimeter = {
                "id": row.id,
                "source_id": source_id,
                "fire_name": fire_name,
                "geom": row.geom,  # WKT from ST_AsText
                "source": row.source,
                "acres": row.acres,
            }
            result = check_perimeter_breach(perimeter, session)
            if result is not None:
                results.append(result)

        LOGGER.info(
            "perimeter_breach_watch: %d breach(es) detected across %d perimeter(s)",
            len(results), len(rows),
        )
        return results

    except Exception:
        LOGGER.exception("perimeter_breach_watch: error fetching active perimeters")
        return []
