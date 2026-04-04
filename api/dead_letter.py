"""Dead-letter queue depth and age monitoring.

Provides metrics for RQ dead-letter queues so external monitors and the
health endpoint can alert on accumulating failed jobs.

Also provides a generic ``move_to_dead_letter_ingest`` failure callback
for ingest/archive RQ jobs that lack their own dead-letter handling.
"""

import logging
import time
from typing import Any

from rq import Queue
from rq.job import Job

from api.cache import get_redis

logger = logging.getLogger(__name__)

# Canonical dead-letter queue names.  Forecast has one today; ingest gets one
# via this module so that failed archive/ingest jobs are observable too.
DEAD_LETTER_QUEUES = ("failed_forecast", "failed_ingest")


def get_dead_letter_queue(name: str) -> Queue:
    """Return an RQ Queue instance for the named dead-letter queue."""
    return Queue(name, connection=get_redis())


def dead_letter_metrics(queue_name: str) -> dict[str, Any]:
    """Return depth and oldest-job age for a single dead-letter queue.

    Returns a dict with:
      - ``depth``: number of jobs currently in the queue
      - ``oldest_job_age_seconds``: age in seconds of the oldest job, or None
        if the queue is empty
    """
    q = get_dead_letter_queue(queue_name)
    job_ids = q.job_ids  # lightweight: reads list from Redis, no job fetch

    depth = len(job_ids)
    oldest_age: float | None = None

    if depth > 0:
        # Only fetch the oldest job (first in the list) to compute age.
        try:
            oldest_job = Job.fetch(job_ids[0], connection=q.connection)
            enqueued_at = oldest_job.enqueued_at
            if enqueued_at is not None:
                oldest_age = time.time() - enqueued_at.timestamp()
        except Exception:
            # Job may have expired from Redis between listing and fetch.
            logger.debug("Could not fetch oldest job %s from %s", job_ids[0], queue_name)

    return {
        "depth": depth,
        "oldest_job_age_seconds": round(oldest_age, 1) if oldest_age is not None else None,
    }


def move_to_dead_letter_ingest(job, connection, type, value, traceback):
    """RQ on_failure callback: park failed ingest/archive jobs in the ``failed_ingest`` DLQ.

    Mirrors :func:`api.forecast.worker.move_to_dead_letter` but for
    ingest-family jobs.  Logging only — no DB status update because archive
    jobs track status via Redis keys, not a DB table.
    """
    error_name = type.__name__ if type else "Unknown"
    error_msg = str(value) if value else ""
    logger.error(
        "Ingest job %s failed: %s: %s — parking in failed_ingest dead-letter queue",
        job.id,
        error_name,
        error_msg,
    )
    try:
        dlq = get_dead_letter_queue("failed_ingest")
        dlq.enqueue_job(job)
    except Exception as e:
        logger.error("Failed to move ingest job %s to dead-letter queue: %s", job.id, e)


def all_dead_letter_metrics() -> dict[str, dict[str, Any]]:
    """Return metrics for every canonical dead-letter queue.

    Keyed by queue name, each value is the dict returned by
    :func:`dead_letter_metrics`.
    """
    result: dict[str, dict[str, Any]] = {}
    for name in DEAD_LETTER_QUEUES:
        try:
            result[name] = dead_letter_metrics(name)
        except Exception:
            logger.exception("Error reading dead-letter metrics for %s", name)
            result[name] = {"depth": None, "oldest_job_age_seconds": None, "error": "unavailable"}
    return result
