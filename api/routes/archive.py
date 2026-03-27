"""API routes for historical archive mode: data availability check and ingest trigger."""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import no_cache
from pydantic import BaseModel
from sqlalchemy import text

from api.db import get_engine

# Add repo root to path so the RQ worker can import ingest module
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

archive_router = APIRouter(tags=["archive"])

TIMEFRAME_HOURS: dict[str, tuple[int, int]] = {
    "morning":   (6,  11),
    "afternoon": (12, 17),
    "evening":   (18, 23),
    "night":     (0,  5),
}

MAX_FIRMS_LOOKBACK_DAYS = 10
MAX_ARCHIVE_RANGE_DAYS = int(os.getenv("MAX_ARCHIVE_RANGE_DAYS", "7"))
INTER_DAY_DELAY_SECONDS = 2
RANGE_REDIS_TTL_SECONDS = 7 * 86400  # 7 days


def _redis_url() -> str:
    """Return the Redis connection URL from environment variables."""
    return f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"


def _timeframe_window(date_str: str, timeframe: str) -> tuple[datetime, datetime]:
    """Return UTC start/end datetimes for a date + timeframe.

    Timeframe hours are treated as UTC hours, consistent with satellite acquisition
    timestamps (acq_time in fire_detections is always UTC) and the frontend's
    computeArchiveTimeRange which also anchors to UTC.
    """
    hours = TIMEFRAME_HOURS.get(timeframe)
    if hours is None:
        raise ValueError(f"Unknown timeframe: {timeframe!r}")
    start_h, end_h = hours
    d = date.fromisoformat(date_str)
    start_dt = datetime(d.year, d.month, d.day, start_h, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(d.year, d.month, d.day, end_h, 59, 59, tzinfo=timezone.utc)
    return start_dt, end_dt


def _full_day_window(date_str: str) -> tuple[datetime, datetime]:
    """Return UTC start/end datetimes spanning the full calendar day."""
    d = date.fromisoformat(date_str)
    start_dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    return start_dt, end_dt


class ArchiveRangeIngestRequest(BaseModel):
    start_date: str  # 'YYYY-MM-DD'
    end_date: str    # 'YYYY-MM-DD'


class ArchiveRangeDayStatus(BaseModel):
    date: str
    status: str  # queued | started | finished | failed
    error: str | None = None


class ArchiveRangeIngestResponse(BaseModel):
    range_job_id: str
    dates: list[str]
    estimated_minutes: int
    warning: str | None = None


class ArchiveRangeStatusResponse(BaseModel):
    range_job_id: str
    days: list[ArchiveRangeDayStatus]
    overall_status: str  # queued | in_progress | completed | partial_failure | not_found
    completed_count: int
    total_count: int


def _compute_range_overall_status(day_statuses: list[dict]) -> str:
    """Derive an overall status string from the list of per-day status dicts."""
    statuses = {d["status"] for d in day_statuses}
    if not statuses or statuses <= {"queued"}:
        return "queued"
    if "started" in statuses or "queued" in statuses:
        return "in_progress"
    if statuses <= {"finished"}:
        return "completed"
    # Mix of finished + failed, no pending/started
    return "partial_failure"


def _run_archive_ingest_range(range_job_id: str, dates: list[str]) -> None:
    """RQ worker task: ingest multiple archive dates sequentially.

    Stores per-day status as a JSON blob in Redis under
    ``archive_range:{range_job_id}`` so the status endpoint can report
    fine-grained progress.  Processing continues even if individual days fail
    so the caller can see which days succeeded.
    """
    import json as _json

    from redis import Redis as _Redis

    redis_conn = _Redis.from_url(_redis_url())
    key = f"archive_range:{range_job_id}"

    # Load the pre-initialised status map once; maintain it in memory for the
    # rest of the loop to avoid a redundant Redis round-trip per day.
    raw = redis_conn.get(key)
    status_map: dict[str, dict] = (
        _json.loads(raw)
        if raw
        else {d: {"status": "queued", "error": None} for d in dates}
    )

    for i, date_str in enumerate(dates):
        status_map[date_str] = {"status": "started", "error": None}
        redis_conn.setex(key, RANGE_REDIS_TTL_SECONDS, _json.dumps(status_map))

        logger.info(
            "Archive range ingest: day %d/%d: %s (range_job_id=%s)",
            i + 1, len(dates), date_str, range_job_id,
        )
        try:
            _run_archive_ingest(date_str, "full")
            status_map[date_str] = {"status": "finished", "error": None}
        except Exception as exc:  # noqa: BLE001
            logger.error("Archive range ingest failed for %s: %s", date_str, exc)
            status_map[date_str] = {"status": "failed", "error": str(exc)}

        redis_conn.setex(key, RANGE_REDIS_TTL_SECONDS, _json.dumps(status_map))

        # Brief pause between days to stay within FIRMS rate limits
        if i < len(dates) - 1:
            time.sleep(INTER_DAY_DELAY_SECONDS)


class ArchiveAvailabilityResponse(BaseModel):
    has_data: bool
    detection_count: int


class ArchiveIngestRequest(BaseModel):
    date: str       # 'YYYY-MM-DD'
    timeframe: Literal["morning", "afternoon", "evening", "night"]


class ArchiveIngestResponse(BaseModel):
    job_id: str
    estimated_minutes: int


class ArchiveIngestStatusResponse(BaseModel):
    status: str   # queued | started | finished | failed | deferred | stopped | unknown
    error: str | None = None


def _run_archive_ingest(date_str: str, timeframe: str) -> None:
    """RQ worker task: run FIRMS ingest + eventize for the requested date.

    Step 1 — FIRMS ingest: fetches exactly 1 day via the FIRMS DATE parameter,
    bypassing the live-ingest watermark so historical detections are not silently
    dropped and the live watermark state is left untouched.

    Step 2 — Eventize: groups the newly inserted fire_detections into fire_events.
    This step is required because the map queries fire_events, not fire_detections.
    Without it the ingest would succeed but the map would remain empty.
    """
    from ingest.firms_ingest import run_firms_ingest

    logger.info("Archive ingest step 1/2 (FIRMS fetch): date=%s timeframe=%s", date_str, timeframe)
    exit_code = run_firms_ingest(day_range=1, area=None, sources=None, archive_date=date_str)
    if exit_code != 0:
        logger.error("Archive FIRMS ingest exited with code %d", exit_code)
        raise RuntimeError(f"FIRMS ingest failed with exit code {exit_code}")

    logger.info("Archive ingest step 2/2 (eventize): date=%s", date_str)
    try:
        from ml.denoiser.eventize import eventize_detections, EventizeParams
        from api.db import get_engine as _get_engine
        start_dt, end_dt = _full_day_window(date_str)
        stats = eventize_detections(
            _get_engine(),
            start_time=start_dt,
            end_time=end_dt,
            params=EventizeParams(),
        )
        logger.info("Archive eventize complete for %s: %s", date_str, stats)
    except Exception as exc:
        logger.error("Eventize step failed for %s: %s", date_str, exc)
        raise RuntimeError(f"Eventize failed for {date_str}: {exc}") from exc


@archive_router.get("/fires/archive/availability", response_model=ArchiveAvailabilityResponse)
async def check_archive_availability(date: str, timeframe: str) -> ArchiveAvailabilityResponse:
    """Check whether fire detection data exists for the given date + timeframe window."""
    if timeframe not in TIMEFRAME_HOURS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid timeframe '{timeframe}'. Must be one of: {list(TIMEFRAME_HOURS)}"
        )
    try:
        start_dt, end_dt = _timeframe_window(date, timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    with get_engine().begin() as conn:
        # Query fire_events (the table the map uses), not fire_detections.
        # Uses the same overlap predicate as list_fire_events_bbox_time so "has_data"
        # accurately reflects whether the map will show anything.
        row = conn.execute(
            text(
                "SELECT COUNT(*) FROM fire_events "
                "WHERE start_time <= :end_time AND end_time >= :start_time"
            ),
            {"start_time": start_dt, "end_time": end_dt},
        ).fetchone()

    count = int(row[0]) if row else 0
    return ArchiveAvailabilityResponse(has_data=count > 0, detection_count=count)


@archive_router.post(
    "/fires/archive/ingest",
    response_model=ArchiveIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(no_cache)],
)
async def trigger_archive_ingest(body: ArchiveIngestRequest) -> ArchiveIngestResponse:
    """Trigger a background FIRMS re-ingest for a historical date/timeframe."""
    if body.timeframe not in TIMEFRAME_HOURS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid timeframe '{body.timeframe}'.",
        )

    try:
        requested_date = date.fromisoformat(body.date)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid date '{body.date}'. Expected YYYY-MM-DD.",
        )

    days_ago = (date.today() - requested_date).days
    if days_ago < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot ingest future dates.",
        )
    if days_ago >= MAX_FIRMS_LOOKBACK_DAYS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Date {body.date} is {days_ago} days ago. "
                f"FIRMS NRT API only supports up to {MAX_FIRMS_LOOKBACK_DAYS} days back. "
                "Historical data older than this is not available via the online API."
            ),
        )

    try:
        from redis import Redis
        from rq import Queue

        redis_conn = Redis.from_url(_redis_url())
        q = Queue(connection=redis_conn, default_timeout=600)
        job = q.enqueue(_run_archive_ingest, body.date, body.timeframe)
        job_id = str(job.id)
    except Exception as exc:
        logger.error("Failed to enqueue archive ingest job: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker queue unavailable. Please try again later.",
        )

    # Always fetches exactly 1 day (archive_date mode), so cost is constant regardless
    # of how far back the date is: ~5 min covers 2 VIIRS downloads + denoiser cold-start.
    estimated_minutes = 5

    return ArchiveIngestResponse(job_id=job_id, estimated_minutes=estimated_minutes)


@archive_router.get("/fires/archive/ingest/{job_id}", response_model=ArchiveIngestStatusResponse)
async def get_archive_ingest_status(job_id: str) -> ArchiveIngestStatusResponse:
    """Return the current status of an archive ingest job."""
    try:
        from redis import Redis
        from rq.job import Job

        redis_conn = Redis.from_url(_redis_url())
        job = Job.fetch(job_id, connection=redis_conn)
        job_status = job.get_status()
        error: str | None = None
        if job.is_failed and job.exc_info:
            # exc_info is a formatted traceback string; take the last line for a concise message
            lines = [ln.strip() for ln in str(job.exc_info).splitlines() if ln.strip()]
            error = lines[-1] if lines else "Job failed with no error info"
        return ArchiveIngestStatusResponse(status=str(job_status.value), error=error)
    except Exception as exc:
        logger.warning("Could not fetch job status for %s: %s", job_id, exc)
        return ArchiveIngestStatusResponse(status="unknown", error=None)


@archive_router.post(
    "/fires/archive/ingest-range",
    response_model=ArchiveRangeIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(no_cache)],
)
async def trigger_archive_ingest_range(body: ArchiveRangeIngestRequest) -> ArchiveRangeIngestResponse:
    """Trigger background FIRMS re-ingest for a contiguous date range.

    Each day is processed sequentially by a single RQ job.  Per-day progress
    is tracked in Redis and available via ``GET /fires/archive/ingest-range/{range_job_id}/status``.
    Ranges exceeding ``MAX_ARCHIVE_RANGE_DAYS`` (default 7, env-overridable) are rejected.
    Ranges larger than 5 days include a warning about temporary DB size impact.
    """
    try:
        start_d = date.fromisoformat(body.start_date)
        end_d = date.fromisoformat(body.end_date)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid date format: {exc}. Expected YYYY-MM-DD.",
        )

    if end_d < start_d:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="end_date must be on or after start_date.",
        )

    today = date.today()
    for d, label in [(start_d, "start_date"), (end_d, "end_date")]:
        days_ago = (today - d).days
        if days_ago < 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{label} {d} is in the future. Cannot ingest future dates.",
            )
        if days_ago >= MAX_FIRMS_LOOKBACK_DAYS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"{label} {d} is {days_ago} days ago. "
                    f"FIRMS NRT API only supports up to {MAX_FIRMS_LOOKBACK_DAYS} days back."
                ),
            )

    num_days = (end_d - start_d).days + 1
    if num_days > MAX_ARCHIVE_RANGE_DAYS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Range of {num_days} days exceeds the maximum of {MAX_ARCHIVE_RANGE_DAYS}. "
                "Reduce the date range or increase MAX_ARCHIVE_RANGE_DAYS."
            ),
        )

    dates = [(start_d + timedelta(days=i)).isoformat() for i in range(num_days)]

    warning: str | None = None
    if num_days > 5:
        warning = (
            f"Ingesting {num_days} days will write a significant volume of archive data. "
            "Archive rows carry a 3-day TTL and are cleaned up automatically, "
            "but large ranges may temporarily increase DB size."
        )

    try:
        import json as _json
        import uuid as _uuid

        from redis import Redis
        from rq import Queue

        redis_conn = Redis.from_url(_redis_url())

        # Pre-generate the range_job_id so we can initialise Redis state before the
        # worker starts (the status endpoint needs to see "queued" immediately).
        range_job_id = str(_uuid.uuid4())
        status_map = {d: {"status": "queued", "error": None} for d in dates}
        redis_conn.setex(
            f"archive_range:{range_job_id}",
            RANGE_REDIS_TTL_SECONDS,
            _json.dumps(status_map),
        )

        q = Queue(connection=redis_conn, default_timeout=num_days * 600)  # 10 min per day
        q.enqueue(_run_archive_ingest_range, range_job_id, dates)

    except Exception as exc:
        logger.error("Failed to enqueue archive range ingest: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker queue unavailable. Please try again later.",
        )

    estimated_minutes = num_days * 5  # ~5 min per day (FIRMS fetch + eventize)

    return ArchiveRangeIngestResponse(
        range_job_id=range_job_id,
        dates=dates,
        estimated_minutes=estimated_minutes,
        warning=warning,
    )


@archive_router.get(
    "/fires/archive/ingest-range/{range_job_id}/status",
    response_model=ArchiveRangeStatusResponse,
)
async def get_archive_range_status(range_job_id: str) -> ArchiveRangeStatusResponse:
    """Return per-day completion status for a range ingest job."""
    try:
        import json as _json

        from redis import Redis

        redis_conn = Redis.from_url(_redis_url())
        raw = redis_conn.get(f"archive_range:{range_job_id}")

        if raw is None:
            return ArchiveRangeStatusResponse(
                range_job_id=range_job_id,
                days=[],
                overall_status="not_found",
                completed_count=0,
                total_count=0,
            )

        status_map: dict[str, dict] = _json.loads(raw)
        days = [
            ArchiveRangeDayStatus(date=d, status=v["status"], error=v.get("error"))
            for d, v in sorted(status_map.items())
        ]
        overall = _compute_range_overall_status([{"status": v["status"]} for v in status_map.values()])
        completed = sum(1 for v in status_map.values() if v["status"] == "finished")

        return ArchiveRangeStatusResponse(
            range_job_id=range_job_id,
            days=days,
            overall_status=overall,
            completed_count=completed,
            total_count=len(days),
        )
    except Exception as exc:
        logger.warning("Could not fetch range status for %s: %s", range_job_id, exc)
        return ArchiveRangeStatusResponse(
            range_job_id=range_job_id,
            days=[],
            overall_status="not_found",
            completed_count=0,
            total_count=0,
        )
