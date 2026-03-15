"""API routes for historical archive mode: data availability check and ingest trigger."""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, status
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


def _timeframe_window(date_str: str, timeframe: str) -> tuple[datetime, datetime]:
    """Return UTC-aware start/end datetimes for a date + timeframe."""
    hours = TIMEFRAME_HOURS.get(timeframe)
    if hours is None:
        raise ValueError(f"Unknown timeframe: {timeframe!r}")
    start_h, end_h = hours
    d = date.fromisoformat(date_str)
    # Treat the hours as local-time on the requested date; use UTC (naive) for DB queries
    start_dt = datetime(d.year, d.month, d.day, start_h, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(d.year, d.month, d.day, end_h, 59, 59, tzinfo=timezone.utc)
    return start_dt, end_dt


class ArchiveAvailabilityResponse(BaseModel):
    has_data: bool
    detection_count: int


class ArchiveIngestRequest(BaseModel):
    date: str       # 'YYYY-MM-DD'
    timeframe: Literal["morning", "afternoon", "evening", "night"]


class ArchiveIngestResponse(BaseModel):
    job_id: str
    estimated_minutes: int


def _run_archive_ingest(date_str: str, timeframe: str) -> None:
    """RQ worker task: run FIRMS ingest for the requested date window."""
    from ingest.firms_ingest import run_firms_ingest

    d = date.fromisoformat(date_str)
    days_ago = (date.today() - d).days
    # Fetch a window that covers the full day (not just the timeframe) to maximise coverage.
    # Clamp to [1, MAX_FIRMS_LOOKBACK_DAYS].
    day_range = max(1, min(days_ago + 1, MAX_FIRMS_LOOKBACK_DAYS))
    logger.info("Archive ingest: date=%s timeframe=%s day_range=%d", date_str, timeframe, day_range)
    exit_code = run_firms_ingest(day_range=day_range, area=None, sources=None)
    if exit_code != 0:
        logger.error("Archive FIRMS ingest exited with code %d", exit_code)


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
        row = conn.execute(
            text(
                "SELECT COUNT(*) FROM fire_detections "
                "WHERE acq_time >= :start_time AND acq_time <= :end_time"
            ),
            {"start_time": start_dt, "end_time": end_dt},
        ).fetchone()

    count = int(row[0]) if row else 0
    return ArchiveAvailabilityResponse(has_data=count > 0, detection_count=count)


@archive_router.post(
    "/fires/archive/ingest",
    response_model=ArchiveIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
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
        import os

        redis_url = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
        redis_conn = Redis.from_url(redis_url)
        q = Queue(connection=redis_conn, default_timeout=600)
        job = q.enqueue(_run_archive_ingest, body.date, body.timeframe)
        job_id = str(job.id)
    except Exception as exc:
        logger.error("Failed to enqueue archive ingest job: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker queue unavailable. Please try again later.",
        )

    # Rough estimate: ~2 min for a 1-day ingest, scales with day_range
    day_range = max(1, min(days_ago + 1, MAX_FIRMS_LOOKBACK_DAYS))
    estimated_minutes = max(2, day_range)

    return ArchiveIngestResponse(job_id=job_id, estimated_minutes=estimated_minutes)
