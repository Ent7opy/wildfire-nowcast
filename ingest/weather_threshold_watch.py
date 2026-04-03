"""Weather threshold watch.

Checks the most recent completed weather run intersecting an AOI bbox for
critical fire-weather conditions.  Relies on a pre-computed ``summary`` key
inside the ``weather_runs.metadata`` JSONB column — see the STAGE-GAP comment
below for the promotion requirement.

Fire weather thresholds (NOTIFICATION_CONTRACTS.md):
    RH < 25%:        Warning
    RH < 15%:        Critical
    Wind shift > 30°: Warning
    Wind shift > 45°: Critical

RH Threshold Crossing (Transition Gate):
    Alerts fire only when RH *crosses* a threshold boundary, not on every run
    where conditions are already bad.  This prevents alert fatigue when adverse
    conditions persist unchanged across many hourly runs.

    Rules (compared against prev_rh from the previous run's summary):
        - prev_rh > 25% AND curr_rh <= 25%  → newly entered Warning territory
        - prev_rh > 15% AND curr_rh <= 15%  → newly entered Critical territory
        - prev_rh <= 15% AND curr_rh < prev_rh  → already Critical, worsening
        - If no previous run exists, falls back to absolute threshold check.

    Wind shift is already delta-based (angular change between runs) and is
    unchanged by this gate.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text

from api.aoi_utils import _is_notifications_paused
from api.notifications import notify

LOGGER = logging.getLogger(__name__)

# STAGE-GAP WARNING (mvp_operational → science_grade): weather_threshold_watch
# requires weather_runs.metadata to include a 'summary' key with pre-computed
# bbox aggregates (rh2m_min, wind_bearing_deg). Without it, threshold checks
# are skipped. Target: add summary computation to weather ingest pipeline.

# ── Thresholds ────────────────────────────────────────────────────────────────
_RH_WARN_PCT = 25.0
_RH_CRIT_PCT = 15.0
_WIND_SHIFT_WARN_DEG = 30.0
_WIND_SHIFT_CRIT_DEG = 45.0


def _angular_difference(a: float, b: float) -> float:
    """Return the absolute shortest angular difference between two bearings (0-180°)."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def check_weather_thresholds(aoi: dict[str, Any], session: Any) -> dict[str, Any] | None:
    """Check the most recent weather run for critical fire-weather thresholds.

    Args:
        aoi:     Dict with at least ``id``, ``name``, and ``bbox`` (GeoJSON Polygon).
        session: SQLAlchemy connection/session obtained from the caller.

    Returns:
        A dict describing triggered conditions if thresholds are *newly crossed*
        or are worsening (see module-level docstring), else None.  Returns None
        when conditions are already bad but unchanged — use the transition gate
        to prevent alert fatigue on persistent adverse conditions.
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
            "weather_threshold_watch: could not parse bbox for AOI %s (%s): %s",
            aoi_name, aoi_id, exc,
        )
        return None

    try:
        # ── Step 1: Fetch the two most recent COMPLETED weather runs that intersect ──
        # We fetch 2 so we can compare wind bearing shift between them.
        runs_sql = text(
            """
            SELECT id, run_time, metadata
            FROM weather_runs
            WHERE status = 'completed'
              AND bbox_min_lon <= :max_lon
              AND bbox_max_lon >= :min_lon
              AND bbox_min_lat <= :max_lat
              AND bbox_max_lat >= :min_lat
            ORDER BY run_time DESC
            LIMIT 2
            """
        )
        runs = session.execute(
            runs_sql,
            {
                "min_lon": min_lon,
                "max_lon": max_lon,
                "min_lat": min_lat,
                "max_lat": max_lat,
            },
        ).fetchall()

        if not runs:
            LOGGER.info(
                "weather_threshold_watch: no completed weather runs found for AOI %s — skipping",
                aoi_name,
            )
            return None

        curr_row = runs[0]
        curr_run_id = curr_row[0]
        curr_metadata: dict[str, Any] = curr_row[2] or {}

        # ── Step 2: Require pre-computed summary ──────────────────────────────────
        summary = curr_metadata.get("summary")
        if summary is None:
            LOGGER.warning(
                "weather_threshold_watch: weather summary not available for run %s — "
                "weather threshold checks require summarized weather_runs metadata",
                curr_run_id,
            )
            return None

        rh2m_min: float | None = summary.get("rh2m_min")
        wind_bearing_deg: float | None = summary.get("wind_bearing_deg")

        if rh2m_min is None and wind_bearing_deg is None:
            LOGGER.info(
                "weather_threshold_watch: summary present but missing rh2m_min and "
                "wind_bearing_deg for AOI %s run %s — skipping",
                aoi_name, curr_run_id,
            )
            return None

        triggers: list[tuple[str, str]] = []  # (condition_name, severity)

        # ── Step 3a: RH threshold-crossing gate ──────────────────────────────────
        # Pull previous run summary once so we can use it for both RH and wind.
        prev_summary: dict[str, Any] | None = None
        if len(runs) >= 2:
            prev_metadata: dict[str, Any] = runs[1][2] or {}
            prev_summary = prev_metadata.get("summary")

        prev_rh: float | None = prev_summary.get("rh2m_min") if prev_summary else None
        rh_body_detail: str = ""

        if rh2m_min is not None:
            curr_rh = rh2m_min
            if prev_rh is not None:
                # Transition gate: only alert on threshold crossings or worsening.
                if prev_rh > _RH_WARN_PCT and curr_rh <= _RH_WARN_PCT:
                    # Newly entered Warning territory (may also cross Critical below).
                    triggers.append(("low_rh_warning", "warning"))
                    rh_body_detail = (
                        f"RH dropped below warning threshold "
                        f"(prev: {prev_rh:.0f}%, now: {curr_rh:.0f}%)"
                    )
                if prev_rh > _RH_CRIT_PCT and curr_rh <= _RH_CRIT_PCT:
                    # Newly crossed Critical threshold (supersedes Warning entry above).
                    triggers.append(("low_rh_critical", "critical"))
                    rh_body_detail = (
                        f"RH dropped below critical threshold "
                        f"(prev: {prev_rh:.0f}%, now: {curr_rh:.0f}%)"
                    )
                elif prev_rh <= _RH_CRIT_PCT and curr_rh < prev_rh:
                    # Already Critical and still dropping — re-alert.
                    triggers.append(("low_rh_critical", "critical"))
                    rh_body_detail = (
                        f"RH still dropping in critical range "
                        f"(prev: {prev_rh:.0f}%, now: {curr_rh:.0f}%)"
                    )
            else:
                # No previous run available — fall back to absolute threshold check.
                if curr_rh <= _RH_CRIT_PCT:
                    triggers.append(("low_rh_critical", "critical"))
                    rh_body_detail = f"RH {curr_rh:.0f}%"
                elif curr_rh <= _RH_WARN_PCT:
                    triggers.append(("low_rh_warning", "warning"))
                    rh_body_detail = f"RH {curr_rh:.0f}%"

        # ── Step 3b: Wind shift vs previous run ───────────────────────────────────
        wind_shift: float | None = None
        if wind_bearing_deg is not None and prev_summary is not None:
            prev_wind_bearing: float | None = prev_summary.get("wind_bearing_deg")
            if prev_wind_bearing is not None:
                wind_shift = _angular_difference(wind_bearing_deg, prev_wind_bearing)
                if wind_shift > _WIND_SHIFT_CRIT_DEG:
                    triggers.append(("wind_shift_critical", "critical"))
                elif wind_shift > _WIND_SHIFT_WARN_DEG:
                    triggers.append(("wind_shift_warning", "warning"))

        # ── Step 3c: Evaluate triggers ────────────────────────────────────────────
        if not triggers:
            LOGGER.info(
                "weather_threshold_watch: AOI %s — rh2m_min=%s wind_shift=%s — "
                "no threshold exceeded",
                aoi_name, rh2m_min, wind_shift,
            )
            return None

        severity = "critical" if any(s == "critical" for _, s in triggers) else "warning"
        condition_names = [c for c, _ in triggers]

        # ── Step 3d: Build human-readable body ────────────────────────────────────
        body_parts: list[str] = []
        if rh_body_detail:
            body_parts.append(rh_body_detail)
        if wind_shift is not None:
            body_parts.append(f"wind shift {wind_shift:.0f}°")
        body = (
            f"Critical fire weather detected in {aoi_name}: "
            + ", ".join(body_parts)
            + "."
        )

        # ── Step 3e: Notify ───────────────────────────────────────────────────────
        LOGGER.warning(
            "weather_threshold_watch: AOI %s — triggers=%s — severity=%s",
            aoi_name, condition_names, severity,
        )
        notify(
            f"weather_threshold:{aoi_id}",
            title=f"Critical fire weather in {aoi_name}",
            body=body,
            severity=severity,
            denoised_score=None,
            aoi_id=str(aoi_id),
            rh_pct=rh2m_min,
            wind_bearing=wind_bearing_deg,
            conditions=condition_names,
        )

        return {
            "aoi_id": str(aoi_id),
            "aoi_name": aoi_name,
            "severity": severity,
            "rh2m_min": rh2m_min,
            "wind_bearing_deg": wind_bearing_deg,
            "wind_shift_deg": wind_shift,
            "conditions": condition_names,
        }

    except Exception as exc:
        LOGGER.error(
            "weather_threshold_watch: unexpected error for AOI %s (%s): %s",
            aoi_name, aoi_id, exc,
            exc_info=True,
        )
        return None


def run_weather_threshold_checks(
    aois: list[dict[str, Any]],
    session: Any,
    *,
    _now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Run weather threshold checks for every watched AOI.

    Args:
        aois:    List of AOI dicts.
        session: SQLAlchemy connection/session.
        _now:    Override for current UTC time (testing only).

    Returns:
        List of non-None results from check_weather_thresholds.
    """
    results: list[dict[str, Any]] = []
    for aoi in aois:
        if _is_notifications_paused(aoi, _now=_now):
            LOGGER.info(
                "weather_threshold_watch: notifications paused for AOI %s — skipping",
                aoi.get("name"),
            )
            continue
        result = check_weather_thresholds(aoi, session)
        if result is not None:
            results.append(result)
    return results
