"""Tests for dead-letter queue metrics and the health endpoint."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.dead_letter import (
    DEAD_LETTER_QUEUES,
    all_dead_letter_metrics,
    dead_letter_metrics,
    move_to_dead_letter_ingest,
)
from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests: dead_letter_metrics
# ---------------------------------------------------------------------------


def test_dead_letter_metrics_empty_queue():
    mock_queue = MagicMock()
    mock_queue.job_ids = []
    mock_queue.connection = MagicMock()

    with patch("api.dead_letter.get_dead_letter_queue", return_value=mock_queue):
        result = dead_letter_metrics("failed_forecast")

    assert result["depth"] == 0
    assert result["oldest_job_age_seconds"] is None


def test_dead_letter_metrics_with_jobs():
    enqueued_at = datetime(2026, 4, 4, 10, 0, 0, tzinfo=timezone.utc)
    mock_job = MagicMock()
    mock_job.enqueued_at = enqueued_at

    mock_queue = MagicMock()
    mock_queue.job_ids = ["job-1", "job-2", "job-3"]
    mock_queue.connection = MagicMock()

    with (
        patch("api.dead_letter.get_dead_letter_queue", return_value=mock_queue),
        patch("api.dead_letter.Job.fetch", return_value=mock_job),
    ):
        result = dead_letter_metrics("failed_forecast")

    assert result["depth"] == 3
    assert result["oldest_job_age_seconds"] is not None
    assert result["oldest_job_age_seconds"] > 0


def test_dead_letter_metrics_handles_fetch_error():
    """If the oldest job vanished from Redis, depth is still reported."""
    mock_queue = MagicMock()
    mock_queue.job_ids = ["gone-job"]
    mock_queue.connection = MagicMock()

    with (
        patch("api.dead_letter.get_dead_letter_queue", return_value=mock_queue),
        patch("api.dead_letter.Job.fetch", side_effect=Exception("no such job")),
    ):
        result = dead_letter_metrics("failed_forecast")

    assert result["depth"] == 1
    assert result["oldest_job_age_seconds"] is None


def test_all_dead_letter_metrics_covers_all_queues():
    mock_queue = MagicMock()
    mock_queue.job_ids = []
    mock_queue.connection = MagicMock()

    with patch("api.dead_letter.get_dead_letter_queue", return_value=mock_queue):
        result = all_dead_letter_metrics()

    for name in DEAD_LETTER_QUEUES:
        assert name in result
        assert result[name]["depth"] == 0


def test_all_dead_letter_metrics_handles_per_queue_error():
    """One queue erroring does not block the others."""

    def _fail_on_forecast(queue_name):
        if queue_name == "failed_forecast":
            raise RuntimeError("Redis down")
        q = MagicMock()
        q.job_ids = []
        q.connection = MagicMock()
        return q

    with patch("api.dead_letter.get_dead_letter_queue", side_effect=_fail_on_forecast):
        result = all_dead_letter_metrics()

    assert "error" in result["failed_forecast"]
    assert result["failed_ingest"]["depth"] == 0


# ---------------------------------------------------------------------------
# Unit test: move_to_dead_letter_ingest callback
# ---------------------------------------------------------------------------


def test_move_to_dead_letter_ingest_parks_job():
    job = SimpleNamespace(id="rq-ingest-job-1")
    mock_dlq = MagicMock()

    with patch("api.dead_letter.get_dead_letter_queue", return_value=mock_dlq):
        move_to_dead_letter_ingest(job, MagicMock(), RuntimeError, RuntimeError("boom"), None)

    mock_dlq.enqueue_job.assert_called_once_with(job)


def test_move_to_dead_letter_ingest_handles_enqueue_failure():
    """If the DLQ itself is unavailable, the callback must not raise."""
    job = SimpleNamespace(id="rq-ingest-job-2")
    mock_dlq = MagicMock()
    mock_dlq.enqueue_job.side_effect = Exception("Redis gone")

    with patch("api.dead_letter.get_dead_letter_queue", return_value=mock_dlq):
        # Should not raise
        move_to_dead_letter_ingest(job, MagicMock(), RuntimeError, RuntimeError("boom"), None)


# ---------------------------------------------------------------------------
# Integration tests: health endpoint
# ---------------------------------------------------------------------------


def test_dead_letter_health_endpoint_returns_queue_metrics(monkeypatch):
    metrics = {
        "failed_forecast": {"depth": 2, "oldest_job_age_seconds": 3600.0},
        "failed_ingest": {"depth": 0, "oldest_job_age_seconds": None},
    }
    monkeypatch.setattr("api.routes.internal.all_dead_letter_metrics", lambda: metrics)

    response = client.get("/internal/health/dead-letter-queues")
    assert response.status_code == 200
    body = response.json()
    assert "as_of" in body
    assert body["queues"] == metrics


def test_dead_letter_health_endpoint_empty_queues(monkeypatch):
    metrics = {
        "failed_forecast": {"depth": 0, "oldest_job_age_seconds": None},
        "failed_ingest": {"depth": 0, "oldest_job_age_seconds": None},
    }
    monkeypatch.setattr("api.routes.internal.all_dead_letter_metrics", lambda: metrics)

    response = client.get("/internal/health/dead-letter-queues")
    assert response.status_code == 200
    body = response.json()
    assert all(q["depth"] == 0 for q in body["queues"].values())


def test_consolidated_dashboard_includes_dead_letter_section(monkeypatch):
    """The /internal/health/dashboard response must include dead_letter_queues."""
    dlq_metrics = {
        "failed_forecast": {"depth": 1, "oldest_job_age_seconds": 120.0},
        "failed_ingest": {"depth": 0, "oldest_job_age_seconds": None},
    }
    monkeypatch.setattr("api.routes.internal.all_dead_letter_metrics", lambda: dlq_metrics)
    monkeypatch.setattr("api.routes.internal.read_orchestrator_dashboard", lambda: None)
    monkeypatch.setattr(
        "api.routes.internal.build_data_status_snapshot",
        lambda include_internal=False: {"overall_state": "healthy", "forecast_inputs_ready": True, "sources": {}},
    )
    monkeypatch.setattr(
        "api.routes.internal.build_db_size_snapshot",
        lambda **_: {
            "as_of": "2026-04-04T00:00:00+00:00",
            "database": {"size_bytes": 100, "size_pretty": "100 B"},
            "tables": {},
            "retention_policy": {},
            "cleanup": {"last_run_at": None, "last_outcome": None, "next_run_at": None, "interval_minutes": None, "source": "error"},
        },
    )

    response = client.get("/internal/health/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert "dead_letter_queues" in body
    assert body["dead_letter_queues"] == dlq_metrics
