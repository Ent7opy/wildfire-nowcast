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
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text

from api.aoi_utils import _is_notifications_paused
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

# ICS operational planning horizons (hours).  Each horizon is checked
# independently; a missing contour logs a WARNING and skips that horizon.
_REQUIRED_HORIZONS: list[int] = [12, 24, 48, 72]

# ── Transition gate state ─────────────────────────────────────────────────────
# Module-level; best-effort (resets on restart).
# science_grade target: persist to DB so restarts don't re-fire old alerts.
# Key: "{aoi_id}:{horizon_hours}"
# Value: {"bearing": float, "run_id": int, "severity": str}
_last_alerted_state: dict[str, dict] = {}


def reset_trajectory_state(aoi_id: str) -> None:
    """Remove all transition-gate state for *aoi_id* across every horizon.

    Call this when an AOI's watch is disabled, or from tests to get a clean
    slate without reloading the module.
    """
    prefix = f"{aoi_id}:"
    keys_to_delete = [k for k in _last_alerted_state if k.startswith(prefix)]
    for k in keys_to_delete:
        del _last_alerted_state[k]


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


def check_spread_trajectory(
    aoi: dict[str, Any], session: Any
) -> list[dict[str, Any]] | None:
    """Check for significant trajectory shifts between the two most recent spread runs.

    Args:
        aoi:     Dict with at least ``id``, ``name``, and ``bbox`` (GeoJSON Polygon).
        session: SQLAlchemy connection/session obtained from the caller.

    Returns:
        ``None`` if fewer than 2 completed runs exist (cannot compute a delta).
        ``[]``   if runs exist but no horizon exceeded a threshold.
        A list of result dicts (one per horizon that triggered), otherwise.
        Each dict includes ``horizon_hours``.
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

        # ── Step 2: Fetch an older run for computing the previous heading ─────────
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

        older_run_id = older_run_row[0] if older_run_row is not None else None
        older_ref_time = older_run_row[1] if older_run_row is not None else None

        # ── Step 3: Helper — fetch the centroid for a specific horizon ────────────
        def _get_centroid_at(
            run_id: int, horizon_h: int
        ) -> tuple[float, float] | None:
            """Return (lon, lat) centroid for the exact horizon, or None."""
            centroid_sql = text(
                """
                SELECT
                    ST_X(ST_Centroid(geom)) AS cx,
                    ST_Y(ST_Centroid(geom)) AS cy
                FROM spread_forecast_contours
                WHERE run_id = :run_id
                  AND threshold = :threshold
                  AND horizon_hours = :horizon_h
                LIMIT 1
                """
            )
            row = session.execute(
                centroid_sql,
                {
                    "run_id": run_id,
                    "threshold": _CONTOUR_THRESHOLD,
                    "horizon_h": horizon_h,
                },
            ).fetchone()
            if row is None:
                return None
            return (float(row[0]), float(row[1]))

        # ── Step 4: Loop over required horizons ───────────────────────────────────
        results: list[dict[str, Any]] = []

        for h in _REQUIRED_HORIZONS:
            curr_centroid = _get_centroid_at(curr_run_id, h)
            if curr_centroid is None:
                LOGGER.warning(
                    "spread_trajectory_watch: no contour at %dh for AOI %s — "
                    "ICS %dh outlook unavailable",
                    h, aoi_name, h,
                )
                continue

            prev_centroid = _get_centroid_at(prev_run_id, h)
            if prev_centroid is None:
                LOGGER.warning(
                    "spread_trajectory_watch: no contour at %dh for AOI %s — "
                    "ICS %dh outlook unavailable",
                    h, aoi_name, h,
                )
                continue

            # ── Bearing and speed for curr→prev leg ──────────────────────────────
            prev_lon, prev_lat = prev_centroid
            curr_lon, curr_lat = curr_centroid

            delta_lon = curr_lon - prev_lon
            delta_lat = curr_lat - prev_lat
            current_bearing = (
                math.degrees(math.atan2(delta_lon, delta_lat)) % 360.0
            )

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
            current_speed_kmh = (
                dist_km / hours_between if hours_between > 0 else 0.0
            )

            # ── Direction change and previous speed (need older run) ──────────────
            direction_change = 0.0
            speed_change_pct = 0.0
            previous_bearing: float | None = None

            if older_run_id is not None:
                older_centroid = _get_centroid_at(older_run_id, h)

                if older_centroid is not None:
                    older_lon, older_lat = older_centroid

                    d_lon = prev_lon - older_lon
                    d_lat = prev_lat - older_lat
                    previous_bearing = (
                        math.degrees(math.atan2(d_lon, d_lat)) % 360.0
                    )

                    direction_change = _angular_difference(
                        previous_bearing, current_bearing
                    )

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
                            (current_speed_kmh - previous_speed_kmh)
                            / previous_speed_kmh
                            * 100.0
                        )

            # ── Apply thresholds ──────────────────────────────────────────────────
            severity = _severity_for(direction_change, speed_change_pct)

            if severity is None:
                LOGGER.info(
                    "spread_trajectory_watch: AOI %s %dh — direction_change=%.1f° "
                    "speed_change=%.1f%% — no threshold exceeded",
                    aoi_name, h, direction_change, speed_change_pct,
                )
                continue

            # ── Transition gate: suppress re-alerts on sustained shifts ───────────
            gate_key = f"{aoi_id}:{h}"
            prior = _last_alerted_state.get(gate_key)
            if prior is not None:
                additional_shift = _angular_difference(
                    prior["bearing"], current_bearing
                )
                if additional_shift < _DIR_WARN_DEG and severity == prior["severity"]:
                    LOGGER.info(
                        "spread_trajectory_watch: suppressing re-alert for AOI %s "
                        "%dh (additional_shift=%.1f°, same severity)",
                        aoi_name, h, additional_shift,
                    )
                    continue

            # ── Notify ────────────────────────────────────────────────────────────
            LOGGER.warning(
                "spread_trajectory_watch: AOI %s %dh — direction_change=%.1f° "
                "speed_change=%.1f%% — severity=%s",
                aoi_name, h, direction_change, speed_change_pct, severity,
            )
            dispatched = notify(
                f"spread_trajectory_shift:{aoi_id}:{h}",
                title=f"Spread trajectory shifted ({h}h outlook) for {aoi_name}",
                body=(
                    f"Projected spread direction rotated {direction_change:.0f}° "
                    f"and speed changed {speed_change_pct:+.0f}% vs previous model run."
                ),
                severity=severity,
                denoised_score=None,
                aoi_id=str(aoi_id),
                horizon_hours=h,
                direction_change_deg=direction_change,
                speed_change_pct=speed_change_pct,
                current_bearing_deg=current_bearing,
            )

            # Only advance the transition gate when the notification was actually
            # dispatched.  If notify() returned False (burst cap, rate limit, no
            # channel), the gate must NOT be updated — the next genuine shift must
            # still be allowed to fire.
            if dispatched:
                _last_alerted_state[gate_key] = {
                    "bearing": current_bearing,
                    "run_id": curr_run_id,
                    "severity": severity,
                }
            else:
                LOGGER.debug(
                    "spread_trajectory_watch: notify() suppressed for AOI %s %dh — "
                    "gate state NOT advanced",
                    aoi_name, h,
                )

            results.append(
                {
                    "aoi_id": str(aoi_id),
                    "aoi_name": aoi_name,
                    "horizon_hours": h,
                    "severity": severity,
                    "direction_change_deg": direction_change,
                    "speed_change_pct": speed_change_pct,
                    "current_bearing_deg": current_bearing,
                    "previous_bearing_deg": previous_bearing,
                    "current_speed_kmh": current_speed_kmh,
                }
            )

        return results

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
    *,
    _now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Run spread trajectory checks for every watched AOI.

    Args:
        aois:    List of AOI dicts (same shape as produced by list_watched_aois_due).
        session: SQLAlchemy connection/session.
        _now:    Override for current UTC time (testing only).

    Returns:
        Flat list of result dicts from all AOIs and all horizons that triggered.
    """
    results: list[dict[str, Any]] = []
    for aoi in aois:
        if _is_notifications_paused(aoi, _now=_now):
            LOGGER.info(
                "spread_trajectory_watch: notifications paused for AOI %s — skipping",
                aoi.get("name"),
            )
            continue
        horizon_results = check_spread_trajectory(aoi, session)
        if horizon_results is not None:
            results.extend(horizon_results)
    return results
