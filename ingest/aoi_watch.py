"""AOI watchlist scheduler.

Queries watched AOIs that are due for a forecast check, submits JIT forecast
jobs to the API, and fires notifications when spread probability exceeds the
configured threshold.

Designed to be called as an orchestrator job (run_aoi_watch_cycle) on a
short recurring interval (default 5 minutes). Each call is idempotent: it
only processes AOIs that are actually due based on their individual
watch_interval_minutes setting.

Alert rate-limiting is enforced at two levels:
  1. DB: watch_last_alerted_at — persists across restarts.
  2. In-process: api.notifications._is_rate_limited — suppresses within-process
     duplicates using the global NOTIFICATION_RATE_LIMIT_SECONDS window.

Environment variables (optional):
  AOI_WATCH_API_BASE_URL   Base URL for the API (default: http://localhost:8000)
  AOI_WATCH_JIT_TIMEOUT_S  Max seconds to wait for a JIT job (default: 300)
  AOI_WATCH_POLL_INTERVAL_S  JIT polling interval in seconds (default: 5)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import httpx

from sqlalchemy import text as sa_text

from api.aoi_utils import _is_notifications_paused
from api.aois.repo import list_watched_aois_due, update_aoi_watch_status
from api.db import get_engine
from api.notifications import notify

LOGGER = logging.getLogger(__name__)

_DEFAULT_API_BASE_URL = "http://localhost:8000"
_DEFAULT_JIT_TIMEOUT_S = 300.0
_DEFAULT_POLL_INTERVAL_S = 5.0

# Terminal JIT job statuses.
_JIT_TERMINAL = {"completed", "failed"}


def _api_base_url() -> str:
    return os.getenv("AOI_WATCH_API_BASE_URL", _DEFAULT_API_BASE_URL).rstrip("/")


def _jit_timeout() -> float:
    return float(os.getenv("AOI_WATCH_JIT_TIMEOUT_S", str(_DEFAULT_JIT_TIMEOUT_S)))


def _poll_interval() -> float:
    return float(os.getenv("AOI_WATCH_POLL_INTERVAL_S", str(_DEFAULT_POLL_INTERVAL_S)))


def _submit_jit_forecast(
    client: httpx.Client,
    bbox_geojson: dict[str, Any],
    api_base: str,
) -> str | None:
    """Submit a JIT forecast for the AOI bbox. Returns job_id or None on error."""
    # Extract bbox from GeoJSON envelope: [min_lon, min_lat, max_lon, max_lat]
    try:
        coords = bbox_geojson["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bbox = [min(lons), min(lats), max(lons), max(lats)]
    except (KeyError, IndexError, TypeError) as exc:
        LOGGER.warning("aoi_watch: could not extract bbox from AOI geometry: %s", exc)
        return None

    try:
        resp = client.post(
            f"{api_base}/forecast/jit",
            json={"bbox": bbox},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["job_id"]
    except Exception as exc:
        LOGGER.warning("aoi_watch: JIT forecast submission failed: %s", exc)
        return None


def _poll_jit_job(
    client: httpx.Client,
    job_id: str,
    api_base: str,
) -> dict[str, Any] | None:
    """Poll a JIT job until terminal. Returns the final job dict or None on timeout/error."""
    timeout_s = _jit_timeout()
    deadline = time.monotonic() + timeout_s
    interval = _poll_interval()

    while time.monotonic() < deadline:
        try:
            resp = client.get(f"{api_base}/forecast/jit/{job_id}", timeout=15.0)
            resp.raise_for_status()
            job = resp.json()
        except Exception as exc:
            LOGGER.warning("aoi_watch: job poll failed job_id=%s: %s", job_id, exc)
            time.sleep(interval)
            continue

        if job.get("status") in _JIT_TERMINAL:
            return job

        time.sleep(interval)

    LOGGER.warning("aoi_watch: JIT job %s timed out after %.0fs", job_id, timeout_s)
    return None


def _should_alert(
    aoi: dict[str, Any],
    max_spread_prob: float,
    now: datetime,
) -> bool:
    """Return True if an alert should fire for this AOI.

    Respects DB-level rate limiting: no duplicate alert within watch_interval_minutes.
    """
    threshold = aoi.get("watch_alert_threshold")
    if threshold is None or max_spread_prob < threshold:
        return False

    last_alerted_at: datetime | None = aoi.get("watch_last_alerted_at")
    interval_minutes: int | None = aoi.get("watch_interval_minutes")
    if last_alerted_at is not None and interval_minutes is not None:
        elapsed_minutes = (now - last_alerted_at).total_seconds() / 60.0
        if elapsed_minutes < interval_minutes:
            return False

    return True


# FIRMS near-real-time data latency floor (acquisition to DB availability)
_FIRMS_LATENCY_FLOOR_MINUTES: int = 60

# Minimum denoised_score for a detection to qualify as confirmed.
_IGNITION_MIN_SCORE: float = 0.7
# Minimum cluster size (number of qualifying detections within proximity).
_IGNITION_MIN_CLUSTER_SIZE: int = 2
# Proximity radius for clustering (metres).
_IGNITION_CLUSTER_RADIUS_M: float = 1000.0
# Lookback window when watch_last_checked_at is NULL.
_IGNITION_DEFAULT_LOOKBACK_HOURS: int = 2


def check_new_ignition(
    aoi: dict[str, Any],
    engine=None,
    _now: datetime | None = None,
) -> dict[str, Any] | None:
    """Check for a new confirmed ignition cluster inside an AOI.

    Queries fire_detections for qualifying detections (is_noise=false,
    false_source_masked=false, denoised_score >= 0.7) within the AOI geometry
    since watch_last_checked_at (or 2 h ago if null).  Groups detections into
    proximity clusters (>= 2 within 1 km of each other) that are either
    unassociated with a fire event or whose event started within the last 2 h
    (i.e. a genuinely new ignition, not an existing tracked fire).

    If a qualifying cluster is found, fires a "new_ignition:{aoi_id}"
    notification and returns a dict with cluster details.

    Returns None when no qualifying cluster is found or on any error.

    Args:
        aoi:    AOI record dict (from list_watched_aois_due).
        engine: SQLAlchemy engine (defaults to get_engine()). Injectable for testing.
        _now:   Override for current UTC time (testing only).
    """
    aoi_id: UUID = aoi["id"]
    aoi_name: str = aoi["name"]

    try:
        now = _now if _now is not None else datetime.now(timezone.utc)
        last_checked_at: datetime | None = aoi.get("watch_last_checked_at")
        if last_checked_at is None:
            since = now - timedelta(hours=_IGNITION_DEFAULT_LOOKBACK_HOURS)
        else:
            # Ensure timezone-aware.
            if last_checked_at.tzinfo is None:
                since = last_checked_at.replace(tzinfo=timezone.utc)
            else:
                since = last_checked_at

        # Serialise AOI geometry to GeoJSON string for PostGIS.
        geometry = aoi.get("geometry")
        if not geometry:
            LOGGER.warning(
                "aoi_watch: AOI %s (%s) has no geometry — skipping ignition check",
                aoi_name,
                aoi_id,
            )
            return None
        geojson_str = json.dumps(geometry)

        db_engine = engine or get_engine()
        with db_engine.begin() as conn:
            # Fetch all qualifying detections within the AOI geometry that are
            # new enough (acquired since the lookback window).
            rows = conn.execute(
                sa_text(
                    """
                    SELECT
                        fd.id,
                        fd.lat,
                        fd.lon,
                        fd.acq_time,
                        fd.denoised_score,
                        fd.event_id,
                        fe.started_at AS event_started_at
                    FROM fire_detections fd
                    LEFT JOIN fire_events fe ON fe.id = fd.event_id
                    WHERE fd.is_noise = false
                      AND fd.false_source_masked = false
                      AND fd.denoised_score >= :min_score
                      AND fd.acq_time >= :since
                      AND ST_Within(
                            fd.geom,
                            ST_SetSRID(ST_GeomFromGeoJSON(:geojson), 4326)
                          )
                    """
                ),
                {
                    "min_score": _IGNITION_MIN_SCORE,
                    "since": since,
                    "geojson": geojson_str,
                },
            ).mappings().all()

        detections = [dict(r) for r in rows]

        # Compute satellite acquisition timestamp and data-currency lag.
        # _FIRMS_LATENCY_FLOOR_MINUTES documents the minimum expected lag.
        max_acq_time: datetime | None = max(
            (d["acq_time"] for d in detections if d.get("acq_time") is not None),
            default=None,
        )
        lag_minutes: int | None = None
        if max_acq_time is not None:
            # Ensure timezone-aware before arithmetic.
            acq_aware = (
                max_acq_time.replace(tzinfo=timezone.utc)
                if max_acq_time.tzinfo is None
                else max_acq_time
            )
            lag_minutes = int((now - acq_aware).total_seconds() / 60)

        LOGGER.info(
            "aoi_watch: ignition check for AOI %s (%s): %d qualifying detection(s) since %s",
            aoi_name,
            aoi_id,
            len(detections),
            since.isoformat(),
        )

        if not detections:
            return None

        # Filter to detections that represent NEW ignitions: event_id is None, or
        # the associated event started within the last 2 hours.
        new_cutoff = now - timedelta(hours=2)
        new_detections = [
            d
            for d in detections
            if d["event_id"] is None
            or (
                d["event_started_at"] is not None
                and (
                    d["event_started_at"].replace(tzinfo=timezone.utc)
                    if d["event_started_at"].tzinfo is None
                    else d["event_started_at"]
                )
                >= new_cutoff
            )
        ]

        if not new_detections:
            LOGGER.info(
                "aoi_watch: AOI %s — all qualifying detections belong to existing events (> 2 h old)",
                aoi_name,
            )
            return None

        # Cluster by proximity: find the largest cluster where at least two
        # detections are within _IGNITION_CLUSTER_RADIUS_M of each other.
        # Simple O(n²) sweep — detection counts per AOI check are small.
        best_cluster: list[dict[str, Any]] = []
        for anchor in new_detections:
            cos_lat = math.cos(math.radians(float(anchor["lat"])))
            cluster: list[dict[str, Any]] = []
            for detection in new_detections:
                dlat_m = (float(detection["lat"]) - float(anchor["lat"])) * 111_000
                dlon_m = (float(detection["lon"]) - float(anchor["lon"])) * 111_000 * cos_lat
                dist_m = math.sqrt(dlat_m**2 + dlon_m**2)
                if dist_m <= _IGNITION_CLUSTER_RADIUS_M:
                    cluster.append(detection)
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        if len(best_cluster) < _IGNITION_MIN_CLUSTER_SIZE:
            LOGGER.info(
                "aoi_watch: AOI %s — largest proximity cluster has %d detection(s) (< %d required)",
                aoi_name,
                len(best_cluster),
                _IGNITION_MIN_CLUSTER_SIZE,
            )
            return None

        # Compute cluster centroid and max score.
        centroid_lat = sum(float(d["lat"]) for d in best_cluster) / len(best_cluster)
        centroid_lon = sum(float(d["lon"]) for d in best_cluster) / len(best_cluster)
        max_score = max(float(d["denoised_score"]) for d in best_cluster)
        detection_count = len(best_cluster)

        LOGGER.warning(
            "aoi_watch: NEW IGNITION detected in AOI %s — %d detection(s), max_score=%.3f, "
            "centroid=(%.4f, %.4f)",
            aoi_name,
            detection_count,
            max_score,
            centroid_lat,
            centroid_lon,
        )

        _data_suffix = (
            f" Based on satellite data acquired at "
            f"{max_acq_time.strftime('%H:%M UTC')} (~{lag_minutes}min ago)."
            if max_acq_time is not None and lag_minutes is not None
            else ""
        )

        if _is_notifications_paused(aoi, _now=now):
            LOGGER.info(
                "aoi_watch: notifications paused for AOI %s until %s — skipping alerts",
                aoi_name,
                aoi.get("watch_notifications_paused_until"),
            )
        else:
            notify(
                f"new_ignition:{aoi_id}",
                title=f"New confirmed ignition in {aoi_name}",
                body=(
                    f"{detection_count} confirmed detection(s) clustered within "
                    f"{_IGNITION_CLUSTER_RADIUS_M / 1000:.0f} km in AOI '{aoi_name}'. "
                    f"Max denoiser confidence: {max_score:.0%}."
                    + _data_suffix
                ),
                severity="critical",
                aoi_id=str(aoi_id),
                denoised_score=round(max_score, 4),
                detection_count=detection_count,
                lat=round(centroid_lat, 6),
                lon=round(centroid_lon, 6),
                data_as_of=max_acq_time.isoformat() if max_acq_time is not None else None,
                data_lag_minutes=lag_minutes,
            )

        return {
            "aoi_id": str(aoi_id),
            "aoi_name": aoi_name,
            "detection_count": detection_count,
            "max_denoised_score": max_score,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "data_as_of": max_acq_time,
            "data_lag_minutes": lag_minutes,
        }

    except Exception:
        LOGGER.exception(
            "aoi_watch: error during ignition check for AOI %s (%s) — returning None",
            aoi_name,
            aoi_id,
        )
        return None


def run_aoi_watch_cycle(api_base_url: str | None = None) -> int:
    """Run one AOI watchlist check cycle.

    Queries all watched AOIs that are due, submits JIT forecasts, checks
    thresholds, and fires notifications.

    Returns the number of AOIs processed.
    """
    api_base = (api_base_url or _api_base_url()).rstrip("/")
    now = datetime.now(timezone.utc)

    due_aois = list_watched_aois_due(now)
    if not due_aois:
        LOGGER.debug("aoi_watch: no AOIs due for check")
        return 0

    LOGGER.info("aoi_watch: %d AOI(s) due for forecast check", len(due_aois))

    processed = 0
    with httpx.Client() as client:
        for aoi in due_aois:
            aoi_id: UUID = aoi["id"]
            aoi_name: str = aoi["name"]

            LOGGER.info("aoi_watch: checking AOI %s (%s)", aoi_name, aoi_id)

            job_id = _submit_jit_forecast(client, aoi["bbox"], api_base)
            if job_id is None:
                # Submission failed — update last_checked_at so we don't spam
                update_aoi_watch_status(
                    aoi_id=aoi_id,
                    last_checked_at=datetime.now(timezone.utc),
                    last_spread_prob=None,
                )
                LOGGER.warning("aoi_watch: skipping AOI %s — forecast submission failed", aoi_name)
                continue

            job = _poll_jit_job(client, job_id, api_base)
            check_time = datetime.now(timezone.utc)

            if job is None or job.get("status") != "completed":
                error = job.get("error") if job else "timeout"
                LOGGER.warning(
                    "aoi_watch: forecast failed for AOI %s job_id=%s: %s",
                    aoi_name, job_id, error,
                )
                update_aoi_watch_status(
                    aoi_id=aoi_id,
                    last_checked_at=check_time,
                    last_spread_prob=None,
                )
                continue

            max_spread_prob: float | None = job.get("result", {}).get("max_spread_prob")
            alerted_at: datetime | None = None

            if max_spread_prob is not None and _should_alert(aoi, max_spread_prob, check_time):
                threshold = aoi["watch_alert_threshold"]
                if _is_notifications_paused(aoi, _now=now):
                    LOGGER.info(
                        "aoi_watch: notifications paused for AOI %s until %s — skipping alerts",
                        aoi_name,
                        aoi.get("watch_notifications_paused_until"),
                    )
                else:
                    notify(
                        event_type=f"aoi_watch_alert:{aoi_id}",
                        title=f"Spread alert: {aoi_name}",
                        body=(
                            f"AOI '{aoi_name}' has reached spread probability "
                            f"{max_spread_prob:.0%} (threshold: {threshold:.0%})."
                        ),
                        severity="warning",
                        aoi_id=str(aoi_id),
                        aoi_name=aoi_name,
                        max_spread_prob=f"{max_spread_prob:.3f}",
                        threshold=f"{threshold:.3f}",
                        job_id=job_id,
                    )
                    alerted_at = check_time
                    LOGGER.info(
                        "aoi_watch: alert fired for AOI %s max_spread_prob=%.3f threshold=%.3f",
                        aoi_name, max_spread_prob, threshold,
                    )
            elif max_spread_prob is not None:
                LOGGER.info(
                    "aoi_watch: AOI %s max_spread_prob=%.3f below threshold=%.3f — no alert",
                    aoi_name, max_spread_prob, aoi.get("watch_alert_threshold"),
                )

            # Run new-ignition check on every cycle pass, independently of spread alerts.
            check_new_ignition(aoi)

            update_aoi_watch_status(
                aoi_id=aoi_id,
                last_checked_at=check_time,
                last_spread_prob=max_spread_prob,
                last_alerted_at=alerted_at,
            )
            processed += 1

    LOGGER.info("aoi_watch: cycle complete — processed %d AOI(s)", processed)

    # ── Batch watch checks that require a DB session ──────────────────────────
    # AOI-specific checks only run when AOIs were due; perimeter breach checks
    # query all active fire_perimeters globally and always run every cycle.
    if due_aois:
        _run_aoi_batch_checks(due_aois)
    _run_perimeter_checks()

    return processed


def _run_aoi_batch_checks(aois: list[dict[str, Any]]) -> None:
    """Run weather threshold and spread trajectory checks for the given AOIs.

    Opens a single read-only DB connection shared across both check functions.
    Errors in any individual check are logged but do not abort others.
    """
    from ingest.weather_threshold_watch import run_weather_threshold_checks  # noqa: PLC0415
    from ingest.spread_trajectory_watch import run_spread_trajectory_checks  # noqa: PLC0415

    engine = get_engine()
    try:
        with engine.connect() as conn:
            try:
                results = run_weather_threshold_checks(aois, conn)
                LOGGER.info(
                    "aoi_watch: weather threshold checks — %d trigger(s)", len(results)
                )
            except Exception:
                LOGGER.exception("aoi_watch: weather threshold checks failed")

            try:
                results = run_spread_trajectory_checks(aois, conn)
                LOGGER.info(
                    "aoi_watch: spread trajectory checks — %d trigger(s)", len(results)
                )
            except Exception:
                LOGGER.exception("aoi_watch: spread trajectory checks failed")
    except Exception:
        LOGGER.exception("aoi_watch: could not open DB connection for AOI batch checks")


def _run_perimeter_checks() -> None:
    """Run perimeter breach checks against all active fire perimeters.

    Runs every watch cycle regardless of whether any AOIs were due — spot fire
    detection is not AOI-specific and must not be gated on AOI scheduling.
    """
    from ingest.perimeter_breach_watch import run_perimeter_breach_checks  # noqa: PLC0415

    engine = get_engine()
    try:
        with engine.connect() as conn:
            results = run_perimeter_breach_checks(conn)
            LOGGER.info(
                "aoi_watch: perimeter breach checks — %d breach(es)", len(results)
            )
    except Exception:
        LOGGER.exception("aoi_watch: perimeter breach checks failed")
