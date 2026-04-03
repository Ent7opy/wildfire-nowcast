"""Spread trajectory deviation watch.

Checks whether consecutive spread forecast runs show a significant change in
projected fire spread direction or speed.  Called as part of the operational
watch cycle; results in a notify() call when thresholds from the shared
notification contract are exceeded.

Spread deviation thresholds (NOTIFICATION_CONTRACTS.md):
    Direction rotation > 30°: Warning
    Direction rotation > 45°: Critical
    Speed increase  > 50%:   Warning
    Speed increase  > 100%:  Critical
"""

from __future__ import annotations

import logging
import math
from typing import Any
from uuid import UUID

from sqlalchemy import text

from api.notifications import notify

LOGGER = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
_DIR_WARN_DEG = 30.0
_DIR_CRIT_DEG = 45.0
_SPEED_WARN_PCT = 50.0
_SPEED_CRIT_PCT = 100.0

# km per degree of latitude (approximate)
_KM_PER_DEG = 111.0

# Default contour parameters
_CONTOUR_THRESHOLD = 0.5
_PREFERRED_HORIZON_H = 12


def _angular_difference(a: float, b: float) -> float:
    """Return the absolute shortest angular difference between two bearings (0-180°)."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _severity_for(direction_change: float, speed_change_pct: float) -> str | None:
    """Return the max severity triggered, or None if no threshold is exceeded."""
    sev: list[str] = []

    if direction_change > _DIR_CRIT_DEG:
        sev.append("critical")
    elif direction_change > _DIR_WARN_DEG:
        sev.append("warning")

    if speed_change_pct > _SPEED_CRIT_PCT:
        sev.append("critical")
    elif speed_change_pct > _SPEED_WARN_PCT:
        sev.append("warning")

    if not sev:
        return None
    return "critical" if "critical" in sev else "warning"


def check_spread_trajectory(aoi: dict[str, Any], session: Any) -> dict[str, Any] | None:
    """Check for significant trajectory shifts between the two most recent spread runs.

    Args:
        aoi:     Dict with at least ``id``, ``name``, and ``bbox`` (GeoJSON Polygon).
        session: SQLAlchemy connection/session obtained from the caller.

    Returns:
        A dict with trajectory metadata if a threshold is exceeded, else None.
    """
    aoi_id: UUID | str = aoi["id"]
    aoi_name: str = aoi["name"]

    try:
        bbox: dict[str, Any] = aoi["bbox"]
        coords = bbox["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
    except (KeyError, IndexError, TypeError) as exc:
        LOGGER.error(
            "spread_trajectory_watch: could not parse bbox for AOI %s (%s): %s",
            aoi_name, aoi_id, exc,
        )
        return None

    try:
        # ── Step 1: Fetch the two most recent COMPLETED runs that intersect the AOI ──
        runs_sql = text(
            """
            SELECT id, forecast_reference_time
            FROM spread_forecast_runs
            WHERE status = 'completed'
              AND ST_Intersects(
                    bbox,
                    ST_SetSRID(
                        ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat),
                        4326
                    )
                  )
            ORDER BY forecast_reference_time DESC
            LIMIT 2
            """
        )
        runs = session.execute(
            runs_sql,
            {
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
            },
        ).fetchall()

        if len(runs) < 2:
            LOGGER.info(
                "spread_trajectory_watch: only %d completed run(s) for AOI %s — "
                "cannot compute delta, skipping",
                len(runs), aoi_name,
            )
            return None

        # runs[0] is the most recent (current), runs[1] is the previous
        curr_run_id, curr_ref_time = runs[0]
        prev_run_id, prev_ref_time = runs[1]

        # ── Step 2: Fetch contour centroids ──────────────────────────────────────
        def _get_centroid(run_id: int) -> tuple[float, float] | None:
            """Return (lon, lat) centroid of the best-matching contour, or None."""
            # Try preferred horizon first; fall back to smallest available
            centroid_sql = text(
                """
                SELECT
                    ST_X(ST_Centroid(geom)) AS cx,
                    ST_Y(ST_Centroid(geom)) AS cy,
                    horizon_hours
                FROM spread_forecast_contours
                WHERE run_id = :run_id
                  AND threshold = :threshold
                ORDER BY
                    ABS(horizon_hours - :preferred_h),
                    horizon_hours
                LIMIT 1
                """
            )
            row = session.execute(
                centroid_sql,
                {
                    "run_id": run_id,
                    "threshold": _CONTOUR_THRESHOLD,
                    "preferred_h": _PREFERRED_HORIZON_H,
                },
            ).fetchone()
            if row is None:
                return None
            return (float(row[0]), float(row[1]))

        curr_centroid = _get_centroid(curr_run_id)
        prev_centroid = _get_centroid(prev_run_id)

        if curr_centroid is None or prev_centroid is None:
            LOGGER.info(
                "spread_trajectory_watch: missing contour centroid for AOI %s "
                "(curr_run=%s prev_run=%s) — skipping",
                aoi_name, curr_run_id, prev_run_id,
            )
            return None

        # ── Step 3: Compute bearings ──────────────────────────────────────────────
        prev_lon, prev_lat = prev_centroid
        curr_lon, curr_lat = curr_centroid

        # Bearing from previous centroid to current centroid
        delta_lon = curr_lon - prev_lon
        delta_lat = curr_lat - prev_lat
        current_bearing = math.degrees(math.atan2(delta_lon, delta_lat)) % 360.0

        # ── Step 4 & 5: Speed proxy ───────────────────────────────────────────────
        dist_sql = text(
            """
            SELECT ST_Distance(
                ST_SetSRID(ST_MakePoint(:prev_lon, :prev_lat), 4326),
                ST_SetSRID(ST_MakePoint(:curr_lon, :curr_lat), 4326)
            ) AS dist_deg
            """
        )
        dist_row = session.execute(
            dist_sql,
            {
                "prev_lon": prev_lon,
                "prev_lat": prev_lat,
                "curr_lon": curr_lon,
                "curr_lat": curr_lat,
            },
        ).fetchone()
        dist_deg: float = float(dist_row[0]) if dist_row else 0.0
        dist_km: float = dist_deg * _KM_PER_DEG

        hours_between = abs(
            (curr_ref_time - prev_ref_time).total_seconds() / 3600.0
        )
        current_speed_kmh = dist_km / hours_between if hours_between > 0 else 0.0

        # ── Need the previous run's heading to compute direction change ───────────
        # We need a third (even older) run to get the previous bearing.  Since the
        # contract only asks us to compare consecutive runs, the "previous heading"
        # is computed as the vector from the run-before-prev to prev.  Fetch it.
        older_sql = text(
            """
            SELECT id, forecast_reference_time
            FROM spread_forecast_runs
            WHERE status = 'completed'
              AND ST_Intersects(
                    bbox,
                    ST_SetSRID(
                        ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat),
                        4326
                    )
                  )
              AND forecast_reference_time < :prev_ref_time
            ORDER BY forecast_reference_time DESC
            LIMIT 1
            """
        )
        older_run_row = session.execute(
            older_sql,
            {
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "prev_ref_time": prev_ref_time,
            },
        ).fetchone()

        direction_change = 0.0
        speed_change_pct = 0.0
        previous_bearing: float | None = None

        if older_run_row is not None:
            older_run_id = older_run_row[0]
            older_ref_time = older_run_row[1]
            older_centroid = _get_centroid(older_run_id)

            if older_centroid is not None:
                older_lon, older_lat = older_centroid
                d_lon = prev_lon - older_lon
                d_lat = prev_lat - older_lat
                previous_bearing = (
                    math.degrees(math.atan2(d_lon, d_lat)) % 360.0
                )

                # Direction change
                direction_change = _angular_difference(previous_bearing, current_bearing)

                # Previous speed
                prev_dist_deg = math.sqrt(
                    (prev_lon - older_lon) ** 2 + (prev_lat - older_lat) ** 2
                )
                prev_dist_km = prev_dist_deg * _KM_PER_DEG
                prev_hours = abs(
                    (prev_ref_time - older_ref_time).total_seconds() / 3600.0
                )
                previous_speed_kmh = (
                    prev_dist_km / prev_hours if prev_hours > 0 else 0.0
                )

                if previous_speed_kmh > 0:
                    speed_change_pct = (
                        (current_speed_kmh - previous_speed_kmh) / previous_speed_kmh * 100.0
                    )
                else:
                    speed_change_pct = 0.0

        # ── Step 8: Apply thresholds ──────────────────────────────────────────────
        severity = _severity_for(direction_change, speed_change_pct)

        if severity is None:
            LOGGER.info(
                "spread_trajectory_watch: AOI %s — direction_change=%.1f° "
                "speed_change=%.1f%% — no threshold exceeded",
                aoi_name, direction_change, speed_change_pct,
            )
            return None

        # ── Step 9: Notify ────────────────────────────────────────────────────────
        LOGGER.warning(
            "spread_trajectory_watch: AOI %s — direction_change=%.1f° "
            "speed_change=%.1f%% — severity=%s",
            aoi_name, direction_change, speed_change_pct, severity,
        )
        notify(
            f"spread_trajectory_shift:{aoi_id}",
            title=f"Spread trajectory shifted for {aoi_name}",
            body=(
                f"Projected spread direction rotated {direction_change:.0f}° "
                f"and speed changed {speed_change_pct:+.0f}% vs previous model run."
            ),
            severity=severity,
            denoised_score=None,
            aoi_id=str(aoi_id),
            direction_change_deg=direction_change,
            speed_change_pct=speed_change_pct,
            current_bearing_deg=current_bearing,
        )

        return {
            "aoi_id": str(aoi_id),
            "aoi_name": aoi_name,
            "severity": severity,
            "direction_change_deg": direction_change,
            "speed_change_pct": speed_change_pct,
            "current_bearing_deg": current_bearing,
            "previous_bearing_deg": previous_bearing,
            "current_speed_kmh": current_speed_kmh,
        }

    except Exception as exc:
        LOGGER.error(
            "spread_trajectory_watch: unexpected error for AOI %s (%s): %s",
            aoi_name, aoi_id, exc,
            exc_info=True,
        )
        return None


def run_spread_trajectory_checks(
    aois: list[dict[str, Any]],
    session: Any,
) -> list[dict[str, Any]]:
    """Run spread trajectory checks for every watched AOI.

    Args:
        aois:    List of AOI dicts (same shape as produced by list_watched_aois_due).
        session: SQLAlchemy connection/session.

    Returns:
        List of non-None results from check_spread_trajectory.
    """
    results: list[dict[str, Any]] = []
    for aoi in aois:
        result = check_spread_trajectory(aoi, session)
        if result is not None:
            results.append(result)
    return results
