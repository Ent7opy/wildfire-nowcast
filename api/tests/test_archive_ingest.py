"""Tests for archive ingest routes and worker function.

Unit tests cover:
  - UTC-aligned timeframe window helpers (the new UTC-anchored logic)
  - Request validation: future dates and dates beyond MAX_FIRMS_LOOKBACK_DAYS
  - Job-status endpoint: graceful "unknown" fallback when Redis is unavailable
  - Watermark is not advanced in archive mode (verified via firms_ingest behaviour)

Integration tests (marked @pytest.mark.integration) require a live database and
the FIRMS/eventize pipeline to be importable.
"""
from __future__ import annotations

import pytest
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from api.routes.archive import (
    _timeframe_window,
    _full_day_window,
    TIMEFRAME_HOURS,
    MAX_FIRMS_LOOKBACK_DAYS,
)


class TestTimeframeWindow:
    """_timeframe_window must produce UTC-anchored start/end pairs."""

    def test_morning_utc_hours(self):
        start, end = _timeframe_window("2026-01-15", "morning")
        assert start == datetime(2026, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 1, 15, 11, 59, 59, tzinfo=timezone.utc)

    def test_afternoon_utc_hours(self):
        start, end = _timeframe_window("2026-01-15", "afternoon")
        assert start == datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 1, 15, 17, 59, 59, tzinfo=timezone.utc)

    def test_evening_utc_hours(self):
        start, end = _timeframe_window("2026-01-15", "evening")
        assert start == datetime(2026, 1, 15, 18, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 1, 15, 23, 59, 59, tzinfo=timezone.utc)

    def test_night_utc_hours(self):
        start, end = _timeframe_window("2026-01-15", "night")
        assert start == datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 1, 15, 5, 59, 59, tzinfo=timezone.utc)

    def test_all_datetimes_are_utc(self):
        for tf in TIMEFRAME_HOURS:
            start, end = _timeframe_window("2026-06-01", tf)
            assert start.tzinfo is timezone.utc, f"{tf} start not UTC"
            assert end.tzinfo is timezone.utc, f"{tf} end not UTC"

    def test_unknown_timeframe_raises(self):
        with pytest.raises(ValueError, match="Unknown timeframe"):
            _timeframe_window("2026-01-01", "lunchtime")


class TestFullDayWindow:
    """_full_day_window must span 00:00:00 to 23:59:59 UTC."""

    def test_spans_full_calendar_day(self):
        start, end = _full_day_window("2026-03-10")
        assert start == datetime(2026, 3, 10, 0, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 3, 10, 23, 59, 59, tzinfo=timezone.utc)

    def test_both_utc(self):
        start, end = _full_day_window("2025-12-31")
        assert start.tzinfo is timezone.utc
        assert end.tzinfo is timezone.utc


# ---------------------------------------------------------------------------
# HTTP endpoint validation (no Redis / DB required)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """FastAPI test client with Redis enqueue mocked out."""
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _error_text(resp) -> str:
    """Extract the human-readable error string from the app's error envelope."""
    body = resp.json()
    # App uses custom envelope: {"code": "422", "message": "...", "details": ...}
    # Fall back to FastAPI standard {"detail": "..."} for robustness.
    return str(body.get("message") or body.get("detail") or body).lower()


class TestTriggerArchiveIngest:
    """trigger_archive_ingest must reject invalid dates before touching Redis."""

    def test_rejects_future_date(self, client):
        future = (date.today() + timedelta(days=1)).isoformat()
        resp = client.post("/fires/archive/ingest", json={"date": future, "timeframe": "morning"})
        assert resp.status_code == 422
        assert "future" in _error_text(resp)

    def test_rejects_date_at_lookback_boundary(self, client):
        too_old = (date.today() - timedelta(days=MAX_FIRMS_LOOKBACK_DAYS)).isoformat()
        resp = client.post("/fires/archive/ingest", json={"date": too_old, "timeframe": "morning"})
        assert resp.status_code == 400
        assert resp.json()["error"] == "ArchiveRangeError"
        assert str(MAX_FIRMS_LOOKBACK_DAYS) in _error_text(resp)

    def test_rejects_invalid_date_format(self, client):
        resp = client.post("/fires/archive/ingest", json={"date": "not-a-date", "timeframe": "morning"})
        assert resp.status_code == 422

    def test_rejects_unknown_timeframe(self, client):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        resp = client.post("/fires/archive/ingest", json={"date": yesterday, "timeframe": "lunchtime"})
        # Pydantic Literal validation rejects invalid timeframe
        assert resp.status_code == 422

    @patch("rq.Queue")
    @patch("api.routes.archive.get_redis")
    def test_accepts_valid_recent_date(self, mock_get_redis, mock_queue_cls, client):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        mock_job = MagicMock()
        mock_job.id = "test-job-id-123"
        mock_queue_cls.return_value.enqueue.return_value = mock_job
        mock_get_redis.return_value = MagicMock()

        resp = client.post("/fires/archive/ingest", json={"date": yesterday, "timeframe": "morning"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["job_id"] == "test-job-id-123"
        assert body["estimated_minutes"] == 5


class TestGetArchiveIngestStatus:
    """get_archive_ingest_status must fall back to 'unknown' when Redis is unavailable."""

    def test_returns_unknown_when_redis_unavailable(self, client):
        with patch("api.routes.archive.get_redis", side_effect=ConnectionError("Redis down")):
            resp = client.get("/fires/archive/ingest/nonexistent-job-id")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unknown"

    @patch("rq.job.Job")
    @patch("api.routes.archive.get_redis")
    def test_returns_finished_status(self, mock_get_redis, mock_job_cls, client):
        mock_job = MagicMock()
        mock_job.get_status.return_value = MagicMock(value="finished")
        mock_job.is_failed = False
        mock_job_cls.fetch.return_value = mock_job
        mock_get_redis.return_value = MagicMock()

        resp = client.get("/fires/archive/ingest/some-job-id")
        assert resp.status_code == 200
        assert resp.json()["status"] == "finished"
        assert resp.json()["error"] is None

    @patch("rq.job.Job")
    @patch("api.routes.archive.get_redis")
    def test_extracts_last_line_of_traceback_on_failure(self, mock_get_redis, mock_job_cls, client):
        mock_job = MagicMock()
        mock_job.get_status.return_value = MagicMock(value="failed")
        mock_job.is_failed = True
        mock_job.exc_info = "Traceback (most recent call last):\n  File ...\nRuntimeError: FIRMS ingest failed"
        mock_job_cls.fetch.return_value = mock_job
        mock_get_redis.return_value = MagicMock()

        resp = client.get("/fires/archive/ingest/some-job-id")
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"
        assert "FIRMS ingest failed" in resp.json()["error"]


# ---------------------------------------------------------------------------
# Archive mode watermark isolation
# ---------------------------------------------------------------------------

class TestArchiveModeWatermarkIsolation:
    """Verify that archive mode does not advance the live ingest watermark."""

    def test_archive_mode_does_not_advance_watermark(self, monkeypatch):
        """run_firms_ingest in archive mode must leave watermark untouched."""
        from ingest.firms_ingest import run_firms_ingest
        import ingest.repository as repo

        advance_calls: list = []
        monkeypatch.setattr(repo, "advance_ingest_watermark", lambda **kw: advance_calls.append(kw))

        # Stub out all network / DB calls so this runs without infrastructure
        monkeypatch.setattr("ingest.firms_ingest.fetch_csv_rows", lambda *a, **kw: [])
        monkeypatch.setattr("ingest.firms_ingest.parse_detection_rows", lambda *a, **kw: ([], MagicMock()))
        monkeypatch.setattr(repo, "create_ingest_batch", lambda **kw: 999)
        monkeypatch.setattr(repo, "get_ingest_watermark", lambda **kw: None)
        monkeypatch.setattr(repo, "insert_detections", lambda *a, **kw: 0)
        monkeypatch.setattr(repo, "finalize_ingest_batch", lambda *a, **kw: None)
        monkeypatch.setattr(repo, "count_detections_for_batch", lambda *a, **kw: 0)

        run_firms_ingest(day_range=1, area=None, sources=None, archive_date="2026-01-10")

        assert advance_calls == [], (
            "advance_ingest_watermark must NOT be called in archive mode, "
            f"but was called with: {advance_calls}"
        )
