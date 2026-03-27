"""Tests for multi-day archive range ingest routes and helpers."""
from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.routes.archive import (
    MAX_ARCHIVE_RANGE_DAYS,
    MAX_FIRMS_LOOKBACK_DAYS,
    _compute_range_overall_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _error_text(resp) -> str:
    body = resp.json()
    return str(body.get("message") or body.get("detail") or body).lower()


def _recent_date(days_ago: int = 1) -> str:
    return (date.today() - timedelta(days=days_ago)).isoformat()


# ---------------------------------------------------------------------------
# _compute_range_overall_status
# ---------------------------------------------------------------------------


class TestComputeRangeOverallStatus:
    def test_all_queued(self):
        assert _compute_range_overall_status([{"status": "queued"}, {"status": "queued"}]) == "queued"

    def test_empty(self):
        assert _compute_range_overall_status([]) == "queued"

    def test_one_started(self):
        result = _compute_range_overall_status([{"status": "started"}, {"status": "queued"}])
        assert result == "in_progress"

    def test_mix_queued_finished(self):
        result = _compute_range_overall_status([{"status": "finished"}, {"status": "queued"}])
        assert result == "in_progress"

    def test_all_finished(self):
        result = _compute_range_overall_status([{"status": "finished"}, {"status": "finished"}])
        assert result == "completed"

    def test_partial_failure(self):
        result = _compute_range_overall_status([{"status": "finished"}, {"status": "failed"}])
        assert result == "partial_failure"

    def test_all_failed(self):
        result = _compute_range_overall_status([{"status": "failed"}, {"status": "failed"}])
        assert result == "partial_failure"


# ---------------------------------------------------------------------------
# POST /fires/archive/ingest-range — validation
# ---------------------------------------------------------------------------


class TestTriggerArchiveIngestRange:
    def test_rejects_end_before_start(self, client):
        resp = client.post(
            "/fires/archive/ingest-range",
            json={"start_date": _recent_date(2), "end_date": _recent_date(3)},
        )
        assert resp.status_code == 422
        assert "end_date" in _error_text(resp)

    def test_rejects_future_start(self, client):
        future = (date.today() + timedelta(days=1)).isoformat()
        resp = client.post(
            "/fires/archive/ingest-range",
            json={"start_date": future, "end_date": future},
        )
        assert resp.status_code == 422
        assert "future" in _error_text(resp)

    def test_rejects_start_beyond_lookback(self, client):
        too_old = (date.today() - timedelta(days=MAX_FIRMS_LOOKBACK_DAYS)).isoformat()
        yesterday = _recent_date(1)
        resp = client.post(
            "/fires/archive/ingest-range",
            json={"start_date": too_old, "end_date": yesterday},
        )
        assert resp.status_code == 422
        assert str(MAX_FIRMS_LOOKBACK_DAYS) in _error_text(resp)

    def test_rejects_end_beyond_lookback(self, client):
        # end_date is within lookback but start is even older — still rejected by end check
        # Actually both are validated. Use a case where start is valid but end is too old.
        # With MAX_FIRMS_LOOKBACK_DAYS=10: start=9 days ago (ok), end=10 days ago < start → rejected first.
        # Let's test: start=10 days ago directly.
        too_old = (date.today() - timedelta(days=MAX_FIRMS_LOOKBACK_DAYS)).isoformat()
        resp = client.post(
            "/fires/archive/ingest-range",
            json={"start_date": too_old, "end_date": too_old},
        )
        assert resp.status_code == 422

    def test_rejects_range_too_large(self, client):
        # MAX_ARCHIVE_RANGE_DAYS+1 days — but must stay within FIRMS lookback.
        # Only possible if MAX_ARCHIVE_RANGE_DAYS < MAX_FIRMS_LOOKBACK_DAYS - 1 (default: 7 < 9)
        if MAX_ARCHIVE_RANGE_DAYS >= MAX_FIRMS_LOOKBACK_DAYS - 1:
            pytest.skip("Cannot construct oversized range within FIRMS lookback window")
        end_days_ago = 1
        start_days_ago = end_days_ago + MAX_ARCHIVE_RANGE_DAYS  # one day more than max
        resp = client.post(
            "/fires/archive/ingest-range",
            json={"start_date": _recent_date(start_days_ago), "end_date": _recent_date(end_days_ago)},
        )
        assert resp.status_code == 422
        assert str(MAX_ARCHIVE_RANGE_DAYS) in _error_text(resp)

    def test_rejects_invalid_date_format(self, client):
        resp = client.post(
            "/fires/archive/ingest-range",
            json={"start_date": "not-a-date", "end_date": _recent_date(1)},
        )
        assert resp.status_code == 422

    @patch("rq.Queue")
    @patch("redis.Redis")
    def test_accepts_single_day_range(self, mock_redis_cls, mock_queue_cls, client):
        mock_redis_cls.from_url.return_value = MagicMock()
        mock_queue_cls.return_value.enqueue.return_value = MagicMock()

        yesterday = _recent_date(1)
        resp = client.post(
            "/fires/archive/ingest-range",
            json={"start_date": yesterday, "end_date": yesterday},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "range_job_id" in body
        assert body["dates"] == [yesterday]
        assert body["estimated_minutes"] == 5
        assert body["warning"] is None

    @patch("rq.Queue")
    @patch("redis.Redis")
    def test_returns_warning_for_large_range(self, mock_redis_cls, mock_queue_cls, client):
        if MAX_ARCHIVE_RANGE_DAYS < 6:
            pytest.skip("MAX_ARCHIVE_RANGE_DAYS too small to test warning threshold")
        mock_redis_cls.from_url.return_value = MagicMock()
        mock_queue_cls.return_value.enqueue.return_value = MagicMock()

        end_days_ago = 1
        start_days_ago = 6  # 6-day range → triggers warning
        if start_days_ago > MAX_ARCHIVE_RANGE_DAYS:
            pytest.skip("Cannot construct 6-day range within limits")
        resp = client.post(
            "/fires/archive/ingest-range",
            json={"start_date": _recent_date(start_days_ago), "end_date": _recent_date(end_days_ago)},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["warning"] is not None
        assert "db" in body["warning"].lower() or "archive" in body["warning"].lower()

    @patch("rq.Queue")
    @patch("redis.Redis")
    def test_dates_list_covers_full_range(self, mock_redis_cls, mock_queue_cls, client):
        mock_redis_cls.from_url.return_value = MagicMock()
        mock_queue_cls.return_value.enqueue.return_value = MagicMock()

        start = _recent_date(3)
        end = _recent_date(1)
        resp = client.post(
            "/fires/archive/ingest-range",
            json={"start_date": start, "end_date": end},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert len(body["dates"]) == 3
        assert body["dates"][0] == start
        assert body["dates"][-1] == end


# ---------------------------------------------------------------------------
# GET /fires/archive/ingest-range/{range_job_id}/status
# ---------------------------------------------------------------------------


class TestGetArchiveRangeStatus:
    def test_returns_not_found_when_redis_empty(self, client):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        with patch("redis.Redis") as mock_cls:
            mock_cls.from_url.return_value = mock_redis
            resp = client.get("/fires/archive/ingest-range/nonexistent-uuid/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["overall_status"] == "not_found"
        assert body["days"] == []

    def test_returns_queued_status(self, client):
        status_map = {
            "2026-03-20": {"status": "queued", "error": None},
            "2026-03-21": {"status": "queued", "error": None},
        }
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(status_map).encode()
        with patch("redis.Redis") as mock_cls:
            mock_cls.from_url.return_value = mock_redis
            resp = client.get("/fires/archive/ingest-range/some-range-id/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["overall_status"] == "queued"
        assert body["total_count"] == 2
        assert body["completed_count"] == 0

    def test_returns_completed_status(self, client):
        status_map = {
            "2026-03-20": {"status": "finished", "error": None},
            "2026-03-21": {"status": "finished", "error": None},
        }
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(status_map).encode()
        with patch("redis.Redis") as mock_cls:
            mock_cls.from_url.return_value = mock_redis
            resp = client.get("/fires/archive/ingest-range/some-range-id/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["overall_status"] == "completed"
        assert body["completed_count"] == 2
        assert body["total_count"] == 2

    def test_returns_partial_failure_status(self, client):
        status_map = {
            "2026-03-20": {"status": "finished", "error": None},
            "2026-03-21": {"status": "failed", "error": "FIRMS ingest failed"},
        }
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(status_map).encode()
        with patch("redis.Redis") as mock_cls:
            mock_cls.from_url.return_value = mock_redis
            resp = client.get("/fires/archive/ingest-range/some-range-id/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["overall_status"] == "partial_failure"
        assert body["completed_count"] == 1
        # Failed day error is surfaced in the days list
        failed = next(d for d in body["days"] if d["date"] == "2026-03-21")
        assert failed["error"] == "FIRMS ingest failed"

    def test_days_sorted_by_date(self, client):
        status_map = {
            "2026-03-22": {"status": "finished", "error": None},
            "2026-03-20": {"status": "finished", "error": None},
            "2026-03-21": {"status": "queued", "error": None},
        }
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(status_map).encode()
        with patch("redis.Redis") as mock_cls:
            mock_cls.from_url.return_value = mock_redis
            resp = client.get("/fires/archive/ingest-range/some-range-id/status")
        assert resp.status_code == 200
        body = resp.json()
        dates = [d["date"] for d in body["days"]]
        assert dates == sorted(dates)

    def test_returns_not_found_on_redis_error(self, client):
        with patch("redis.Redis") as mock_cls:
            mock_cls.from_url.side_effect = ConnectionError("Redis down")
            resp = client.get("/fires/archive/ingest-range/any-id/status")
        assert resp.status_code == 200
        assert resp.json()["overall_status"] == "not_found"
