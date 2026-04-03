"""Tests for per-AOI notification pause (operator mute) feature."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

_NOW = datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)


def _make_aoi(**overrides) -> dict:
    base = {
        "id": uuid4(),
        "name": "Test AOI",
        "bbox": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
        "watch_enabled": True,
        "watch_interval_minutes": 30,
        "watch_alert_threshold": 0.5,
        "watch_last_checked_at": None,
        "watch_last_alerted_at": None,
        "watch_last_spread_prob": None,
        "watch_notifications_paused_until": None,
    }
    base.update(overrides)
    return base


# ── check_new_ignition: paused AOI skips notify ───────────────────────────────

def test_paused_aoi_skips_notify_in_new_ignition():
    """When notifications are paused, check_new_ignition runs the DB query but
    does NOT call notify()."""
    from ingest.aoi_watch import check_new_ignition

    aoi = _make_aoi(
        watch_notifications_paused_until=_NOW + timedelta(hours=2),
    )

    # Two qualifying detections within the cluster radius, both new
    mock_rows = [
        {
            "id": uuid4(),
            "lat": 37.0,
            "lon": -120.0,
            "acq_time": _NOW - timedelta(minutes=30),
            "denoised_score": 0.85,
            "event_id": None,
            "event_started_at": None,
        },
        {
            "id": uuid4(),
            "lat": 37.001,
            "lon": -120.001,
            "acq_time": _NOW - timedelta(minutes=25),
            "denoised_score": 0.90,
            "event_id": None,
            "event_started_at": None,
        },
    ]

    mock_conn = MagicMock()
    mock_conn.execute.return_value.mappings.return_value.all.return_value = mock_rows
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    with patch("ingest.aoi_watch.notify") as mock_notify, \
         patch("ingest.aoi_watch.get_engine", return_value=mock_engine):
        result = check_new_ignition(aoi, engine=mock_engine, _now=_NOW)

    # A cluster was found but notify must not have been called
    assert result is not None
    mock_notify.assert_not_called()


def test_expired_pause_allows_notify():
    """When the pause timestamp is in the past (expired), notify IS called."""
    from datetime import datetime, timezone

    from ingest.aoi_watch import check_new_ignition

    # Use real now() so the expired timestamp is genuinely in the past.
    real_now = datetime.now(timezone.utc)
    aoi = _make_aoi(
        watch_notifications_paused_until=real_now - timedelta(hours=1),
    )

    mock_rows = [
        {
            "id": uuid4(),
            "lat": 37.0,
            "lon": -120.0,
            "acq_time": _NOW - timedelta(minutes=30),
            "denoised_score": 0.85,
            "event_id": None,
            "event_started_at": None,
        },
        {
            "id": uuid4(),
            "lat": 37.001,
            "lon": -120.001,
            "acq_time": _NOW - timedelta(minutes=25),
            "denoised_score": 0.90,
            "event_id": None,
            "event_started_at": None,
        },
    ]

    mock_conn = MagicMock()
    mock_conn.execute.return_value.mappings.return_value.all.return_value = mock_rows
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    with patch("ingest.aoi_watch.notify") as mock_notify, \
         patch("ingest.aoi_watch.get_engine", return_value=mock_engine):
        result = check_new_ignition(aoi, engine=mock_engine, _now=_NOW)

    assert result is not None
    mock_notify.assert_called_once()


def test_no_pause_field_allows_notify():
    """AOI with no paused_until key treats notifications as active."""
    from ingest.aoi_watch import check_new_ignition

    aoi = _make_aoi()
    # Verify the field is absent, not just None
    aoi.pop("watch_notifications_paused_until", None)

    mock_rows = [
        {
            "id": uuid4(),
            "lat": 37.0,
            "lon": -120.0,
            "acq_time": _NOW - timedelta(minutes=30),
            "denoised_score": 0.85,
            "event_id": None,
            "event_started_at": None,
        },
        {
            "id": uuid4(),
            "lat": 37.001,
            "lon": -120.001,
            "acq_time": _NOW - timedelta(minutes=25),
            "denoised_score": 0.90,
            "event_id": None,
            "event_started_at": None,
        },
    ]

    mock_conn = MagicMock()
    mock_conn.execute.return_value.mappings.return_value.all.return_value = mock_rows
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    with patch("ingest.aoi_watch.notify") as mock_notify, \
         patch("ingest.aoi_watch.get_engine", return_value=mock_engine):
        result = check_new_ignition(aoi, engine=mock_engine, _now=_NOW)

    assert result is not None
    mock_notify.assert_called_once()


# ── run_spread_trajectory_checks: paused AOI skips check_spread_trajectory ────

def test_spread_trajectory_skips_paused_aoi():
    """run_spread_trajectory_checks skips check_spread_trajectory for paused AOIs."""
    from ingest.spread_trajectory_watch import run_spread_trajectory_checks

    paused_aoi = _make_aoi(
        watch_notifications_paused_until=_NOW + timedelta(hours=3),
    )
    session = MagicMock()

    with patch("ingest.spread_trajectory_watch.check_spread_trajectory") as mock_check:
        results = run_spread_trajectory_checks([paused_aoi], session)

    mock_check.assert_not_called()
    assert results == []


def test_spread_trajectory_runs_for_active_aoi():
    """run_spread_trajectory_checks calls check_spread_trajectory when not paused."""
    from ingest.spread_trajectory_watch import run_spread_trajectory_checks

    active_aoi = _make_aoi()
    session = MagicMock()

    with patch("ingest.spread_trajectory_watch.check_spread_trajectory", return_value=[]) as mock_check:
        results = run_spread_trajectory_checks([active_aoi], session)

    mock_check.assert_called_once_with(active_aoi, session)


# ── run_weather_threshold_checks: paused AOI skips check_weather_thresholds ───

def test_weather_threshold_skips_paused_aoi():
    """run_weather_threshold_checks skips check_weather_thresholds for paused AOIs."""
    from ingest.weather_threshold_watch import run_weather_threshold_checks

    paused_aoi = _make_aoi(
        watch_notifications_paused_until=_NOW + timedelta(hours=1),
    )
    session = MagicMock()

    with patch("ingest.weather_threshold_watch.check_weather_thresholds") as mock_check:
        results = run_weather_threshold_checks([paused_aoi], session)

    mock_check.assert_not_called()
    assert results == []


def test_weather_threshold_runs_for_active_aoi():
    """run_weather_threshold_checks calls check_weather_thresholds when not paused."""
    from ingest.weather_threshold_watch import run_weather_threshold_checks

    active_aoi = _make_aoi()
    session = MagicMock()

    with patch("ingest.weather_threshold_watch.check_weather_thresholds", return_value=None) as mock_check:
        results = run_weather_threshold_checks([active_aoi], session)

    mock_check.assert_called_once_with(active_aoi, session)
