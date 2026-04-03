"""Tests for per-AOI notification pause/resume endpoints and helper."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient

import api.routes.aois as aois_routes
from api.aoi_utils import _is_notifications_paused
from api.main import app

client = TestClient(app)

_NOW_STR = "2026-04-03T12:00:00+00:00"

_BASE_AOI = {
    "id": None,  # overridden per test
    "name": "Test AOI",
    "description": None,
    "tags": None,
    "owner_id": None,
    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
    "bbox": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
    "area_km2": 100.0,
    "vertex_count": 5,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "watch_enabled": True,
    "watch_interval_minutes": 30,
    "watch_alert_threshold": 0.5,
    "watch_last_checked_at": None,
    "watch_last_alerted_at": None,
    "watch_last_spread_prob": None,
    "watch_notifications_paused_until": None,
}


def _make_aoi(**overrides):
    aoi = dict(_BASE_AOI)
    aoi["id"] = uuid4()
    aoi.update(overrides)
    return aoi


# ── POST /aois/{aoi_id}/pause-notifications ───────────────────────────────────

def test_pause_sets_paused_until(monkeypatch):
    """POST pause with 4h → response contains paused_until ~4h from now."""
    aoi = _make_aoi()
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=aoi))
    mock_set = MagicMock()
    monkeypatch.setattr(aois_routes.repo, "set_aoi_notifications_paused_until", mock_set)

    response = client.post(
        f"/aois/{aoi['id']}/pause-notifications",
        json={"duration_hours": 4.0, "reason": "Known prescribed burn"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["aoi_id"] == str(aoi["id"])
    assert data["reason"] == "Known prescribed burn"

    # paused_until should be ~4h from now (allow ±60s slop)
    paused_until = datetime.fromisoformat(data["paused_until"])
    expected = datetime.now(timezone.utc) + timedelta(hours=4)
    assert abs((paused_until - expected).total_seconds()) < 60

    # Repo function must have been called with a future timestamp
    mock_set.assert_called_once()
    call_args = mock_set.call_args[0]
    assert call_args[1] > datetime.now(timezone.utc)


def test_pause_rejects_duration_over_168h(monkeypatch):
    """duration=200 exceeds the 168h maximum → 422 Unprocessable Entity."""
    aoi = _make_aoi()
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=aoi))

    response = client.post(
        f"/aois/{aoi['id']}/pause-notifications",
        json={"duration_hours": 200.0},
    )

    assert response.status_code == 422


def test_pause_rejects_duration_below_minimum(monkeypatch):
    """duration=0.1 is below the 0.5h minimum → 422 Unprocessable Entity."""
    aoi = _make_aoi()
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=aoi))

    response = client.post(
        f"/aois/{aoi['id']}/pause-notifications",
        json={"duration_hours": 0.1},
    )

    assert response.status_code == 422


def test_pause_404_for_unknown_aoi(monkeypatch):
    """Unknown aoi_id → 404."""
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=None))

    response = client.post(
        f"/aois/{uuid4()}/pause-notifications",
        json={"duration_hours": 2.0},
    )

    assert response.status_code == 404


# ── POST /aois/{aoi_id}/resume-notifications ─────────────────────────────────

def test_resume_clears_paused_until(monkeypatch):
    """POST resume → repo is called with None to clear the pause."""
    aoi = _make_aoi(
        watch_notifications_paused_until=(
            datetime.now(timezone.utc) + timedelta(hours=2)
        ).isoformat()
    )
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=aoi))
    mock_set = MagicMock()
    monkeypatch.setattr(aois_routes.repo, "set_aoi_notifications_paused_until", mock_set)

    response = client.post(f"/aois/{aoi['id']}/resume-notifications")

    assert response.status_code == 200
    data = response.json()
    assert data["aoi_id"] == str(aoi["id"])
    assert "resumed_at" in data

    # Repo must be called with None to clear the pause
    mock_set.assert_called_once()
    call_args = mock_set.call_args[0]
    assert call_args[1] is None


def test_resume_404_for_unknown_aoi(monkeypatch):
    """Unknown aoi_id → 404."""
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=None))

    response = client.post(f"/aois/{uuid4()}/resume-notifications")

    assert response.status_code == 404


# ── _is_notifications_paused helper ──────────────────────────────────────────

def test_is_notifications_paused_future():
    """paused_until in the future → True."""
    aoi = {"watch_notifications_paused_until": datetime.now(timezone.utc) + timedelta(hours=2)}
    assert _is_notifications_paused(aoi) is True


def test_is_notifications_paused_past():
    """paused_until in the past → False (pause expired)."""
    aoi = {"watch_notifications_paused_until": datetime.now(timezone.utc) - timedelta(hours=1)}
    assert _is_notifications_paused(aoi) is False


def test_is_notifications_paused_none():
    """paused_until = None → False (notifications active)."""
    aoi = {"watch_notifications_paused_until": None}
    assert _is_notifications_paused(aoi) is False


def test_is_notifications_paused_string_future():
    """paused_until as ISO string in the future → True."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    aoi = {"watch_notifications_paused_until": future}
    assert _is_notifications_paused(aoi) is True


def test_is_notifications_paused_string_past():
    """paused_until as ISO string in the past → False."""
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    aoi = {"watch_notifications_paused_until": past}
    assert _is_notifications_paused(aoi) is False


def test_is_notifications_paused_missing_key():
    """AOI with no paused_until key → False (treat as active)."""
    aoi = {}
    assert _is_notifications_paused(aoi) is False
