"""Tests for the AOI watchlist scheduler."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

from ingest.aoi_watch import _should_alert, run_aoi_watch_cycle

_NOW = datetime(2026, 3, 27, 12, 0, 0, tzinfo=timezone.utc)


def _make_aoi(**overrides) -> dict:
    base = {
        "id": uuid4(),
        "name": "Test AOI",
        "bbox": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
        "watch_enabled": True,
        "watch_interval_minutes": 30,
        "watch_alert_threshold": 0.5,
        "watch_last_checked_at": None,
        "watch_last_alerted_at": None,
        "watch_last_spread_prob": None,
    }
    base.update(overrides)
    return base


# ── _should_alert ─────────────────────────────────────────────────────────────


def test_should_alert_above_threshold():
    aoi = _make_aoi(watch_alert_threshold=0.5)
    assert _should_alert(aoi, 0.75, _NOW) is True


def test_should_alert_at_threshold():
    aoi = _make_aoi(watch_alert_threshold=0.5)
    assert _should_alert(aoi, 0.5, _NOW) is True


def test_should_alert_below_threshold():
    aoi = _make_aoi(watch_alert_threshold=0.5)
    assert _should_alert(aoi, 0.3, _NOW) is False


def test_should_alert_no_threshold():
    aoi = _make_aoi(watch_alert_threshold=None)
    assert _should_alert(aoi, 0.9, _NOW) is False


def test_should_alert_rate_limited_within_interval():
    """No alert when last_alerted_at is within watch_interval_minutes."""
    last_alerted = _NOW - timedelta(minutes=10)  # 10 min ago, interval is 30 min
    aoi = _make_aoi(
        watch_alert_threshold=0.5,
        watch_interval_minutes=30,
        watch_last_alerted_at=last_alerted,
    )
    assert _should_alert(aoi, 0.9, _NOW) is False


def test_should_alert_rate_limit_expired():
    """Alert allowed when last_alerted_at is beyond watch_interval_minutes."""
    last_alerted = _NOW - timedelta(minutes=35)  # 35 min ago, interval is 30 min
    aoi = _make_aoi(
        watch_alert_threshold=0.5,
        watch_interval_minutes=30,
        watch_last_alerted_at=last_alerted,
    )
    assert _should_alert(aoi, 0.9, _NOW) is True


def test_should_alert_never_alerted():
    """Alert allowed when never alerted before."""
    aoi = _make_aoi(watch_alert_threshold=0.5, watch_last_alerted_at=None)
    assert _should_alert(aoi, 0.9, _NOW) is True


# ── run_aoi_watch_cycle ────────────────────────────────────────────────────────


def test_cycle_no_due_aois():
    """Cycle returns 0 when no AOIs are due."""
    with patch("ingest.aoi_watch.list_watched_aois_due", return_value=[]):
        result = run_aoi_watch_cycle(api_base_url="http://localhost:8000")
    assert result == 0


def test_cycle_jit_submission_failure():
    """Cycle handles JIT submission failure gracefully — updates last_checked_at."""
    aoi = _make_aoi()

    mock_update = MagicMock()
    with (
        patch("ingest.aoi_watch.list_watched_aois_due", return_value=[aoi]),
        patch("ingest.aoi_watch._submit_jit_forecast", return_value=None),
        patch("ingest.aoi_watch.update_aoi_watch_status", mock_update),
    ):
        result = run_aoi_watch_cycle(api_base_url="http://localhost:8000")

    assert result == 0
    # Status should still be updated (to record the attempt)
    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args.kwargs
    assert call_kwargs["last_spread_prob"] is None


def test_cycle_jit_timeout():
    """Cycle handles JIT job timeout — records None spread prob."""
    aoi = _make_aoi()
    mock_update = MagicMock()

    with (
        patch("ingest.aoi_watch.list_watched_aois_due", return_value=[aoi]),
        patch("ingest.aoi_watch._submit_jit_forecast", return_value="fake-job-id"),
        patch("ingest.aoi_watch._poll_jit_job", return_value=None),  # timeout
        patch("ingest.aoi_watch.update_aoi_watch_status", mock_update),
    ):
        result = run_aoi_watch_cycle(api_base_url="http://localhost:8000")

    assert result == 0
    mock_update.assert_called_once()
    assert mock_update.call_args.kwargs["last_spread_prob"] is None


def test_cycle_forecast_below_threshold():
    """Cycle processes AOI but does NOT fire alert when below threshold."""
    aoi = _make_aoi(watch_alert_threshold=0.7)
    mock_update = MagicMock()
    mock_notify = MagicMock()

    job_result = {"status": "completed", "result": {"max_spread_prob": 0.4}}

    with (
        patch("ingest.aoi_watch.list_watched_aois_due", return_value=[aoi]),
        patch("ingest.aoi_watch._submit_jit_forecast", return_value="job-123"),
        patch("ingest.aoi_watch._poll_jit_job", return_value=job_result),
        patch("ingest.aoi_watch.update_aoi_watch_status", mock_update),
        patch("ingest.aoi_watch.notify", mock_notify),
    ):
        result = run_aoi_watch_cycle(api_base_url="http://localhost:8000")

    assert result == 1
    mock_notify.assert_not_called()
    call_kwargs = mock_update.call_args.kwargs
    assert call_kwargs["last_spread_prob"] == 0.4
    assert call_kwargs["last_alerted_at"] is None


def test_cycle_forecast_above_threshold_fires_alert():
    """Cycle fires notification when spread prob exceeds threshold."""
    aoi = _make_aoi(watch_alert_threshold=0.5)
    mock_update = MagicMock()
    mock_notify = MagicMock()

    job_result = {"status": "completed", "result": {"max_spread_prob": 0.8}}

    with (
        patch("ingest.aoi_watch.list_watched_aois_due", return_value=[aoi]),
        patch("ingest.aoi_watch._submit_jit_forecast", return_value="job-456"),
        patch("ingest.aoi_watch._poll_jit_job", return_value=job_result),
        patch("ingest.aoi_watch.update_aoi_watch_status", mock_update),
        patch("ingest.aoi_watch.notify", mock_notify),
    ):
        result = run_aoi_watch_cycle(api_base_url="http://localhost:8000")

    assert result == 1
    mock_notify.assert_called_once()

    call_kwargs = mock_notify.call_args.kwargs
    assert call_kwargs["event_type"].startswith("aoi_watch_alert:")
    assert call_kwargs.get("severity") == "warning"

    call_kwargs = mock_update.call_args.kwargs
    assert call_kwargs["last_spread_prob"] == 0.8
    assert call_kwargs["last_alerted_at"] is not None


def test_cycle_rate_limit_prevents_duplicate_alert():
    """No alert when AOI was alerted within its watch interval."""
    last_alerted = datetime.now(timezone.utc) - timedelta(minutes=5)
    aoi = _make_aoi(
        watch_alert_threshold=0.5,
        watch_interval_minutes=30,
        watch_last_alerted_at=last_alerted,
    )
    mock_update = MagicMock()
    mock_notify = MagicMock()

    job_result = {"status": "completed", "result": {"max_spread_prob": 0.9}}

    with (
        patch("ingest.aoi_watch.list_watched_aois_due", return_value=[aoi]),
        patch("ingest.aoi_watch._submit_jit_forecast", return_value="job-789"),
        patch("ingest.aoi_watch._poll_jit_job", return_value=job_result),
        patch("ingest.aoi_watch.update_aoi_watch_status", mock_update),
        patch("ingest.aoi_watch.notify", mock_notify),
    ):
        result = run_aoi_watch_cycle(api_base_url="http://localhost:8000")

    assert result == 1
    mock_notify.assert_not_called()  # suppressed by rate limiting
    # last_alerted_at should NOT be updated since we did not alert
    assert mock_update.call_args.kwargs["last_alerted_at"] is None


def test_cycle_multiple_aois_independent():
    """Multiple AOIs are each checked independently."""
    aoi_ok = _make_aoi(name="OK AOI", watch_alert_threshold=0.8)
    aoi_alert = _make_aoi(name="Alert AOI", watch_alert_threshold=0.3)
    mock_update = MagicMock()
    mock_notify = MagicMock()

    def fake_poll(client, job_id, api_base):
        # Both succeed but with different probabilities
        return {"status": "completed", "result": {"max_spread_prob": 0.5}}

    with (
        patch("ingest.aoi_watch.list_watched_aois_due", return_value=[aoi_ok, aoi_alert]),
        patch("ingest.aoi_watch._submit_jit_forecast", return_value="some-job-id"),
        patch("ingest.aoi_watch._poll_jit_job", side_effect=fake_poll),
        patch("ingest.aoi_watch.update_aoi_watch_status", mock_update),
        patch("ingest.aoi_watch.notify", mock_notify),
    ):
        result = run_aoi_watch_cycle(api_base_url="http://localhost:8000")

    assert result == 2
    assert mock_update.call_count == 2
    # Only the alert AOI (threshold=0.3, prob=0.5) should fire
    assert mock_notify.call_count == 1
    alerted_event = mock_notify.call_args.kwargs["event_type"]
    assert alerted_event.startswith("aoi_watch_alert:")
