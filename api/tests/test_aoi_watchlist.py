"""Tests for AOI watchlist endpoints."""

from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient

import api.routes.aois as aois_routes
from api.main import app

client = TestClient(app)

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
    "watch_enabled": False,
    "watch_interval_minutes": None,
    "watch_alert_threshold": None,
    "watch_last_checked_at": None,
    "watch_last_alerted_at": None,
    "watch_last_spread_prob": None,
}


def _make_aoi(**overrides):
    aoi = dict(_BASE_AOI)
    aoi["id"] = uuid4()
    aoi.update(overrides)
    return aoi


# ── PUT /aois/{id}/watch ──────────────────────────────────────────────────────


def test_configure_watch_enable(monkeypatch):
    """Enabling watch with valid config returns 200."""
    aoi_id = uuid4()
    watched_aoi = _make_aoi(
        id=aoi_id,
        watch_enabled=True,
        watch_interval_minutes=30,
        watch_alert_threshold=0.5,
    )

    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=_make_aoi(id=aoi_id)))
    monkeypatch.setattr(aois_routes.repo, "set_aoi_watch", MagicMock(return_value=watched_aoi))

    response = client.put(
        f"/aois/{aoi_id}/watch",
        json={"enabled": True, "interval_minutes": 30, "alert_threshold": 0.5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["watch_enabled"] is True
    assert data["watch_interval_minutes"] == 30
    assert data["watch_alert_threshold"] == 0.5


def test_configure_watch_disable(monkeypatch):
    """Disabling watch clears interval and threshold."""
    aoi_id = uuid4()
    disabled_aoi = _make_aoi(id=aoi_id, watch_enabled=False)

    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=_make_aoi(id=aoi_id)))
    monkeypatch.setattr(aois_routes.repo, "set_aoi_watch", MagicMock(return_value=disabled_aoi))

    response = client.put(f"/aois/{aoi_id}/watch", json={"enabled": False})
    assert response.status_code == 200
    assert response.json()["watch_enabled"] is False


def test_configure_watch_enable_missing_interval(monkeypatch):
    """Enabling watch without interval_minutes is rejected."""
    aoi_id = uuid4()
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=_make_aoi(id=aoi_id)))

    response = client.put(
        f"/aois/{aoi_id}/watch",
        json={"enabled": True, "alert_threshold": 0.5},
    )
    assert response.status_code == 422


def test_configure_watch_enable_missing_threshold(monkeypatch):
    """Enabling watch without alert_threshold is rejected."""
    aoi_id = uuid4()
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=_make_aoi(id=aoi_id)))

    response = client.put(
        f"/aois/{aoi_id}/watch",
        json={"enabled": True, "interval_minutes": 30},
    )
    assert response.status_code == 422


def test_configure_watch_invalid_interval(monkeypatch):
    """interval_minutes below minimum is rejected by Pydantic validator."""
    aoi_id = uuid4()
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=_make_aoi(id=aoi_id)))

    response = client.put(
        f"/aois/{aoi_id}/watch",
        json={"enabled": True, "interval_minutes": 1, "alert_threshold": 0.5},
    )
    assert response.status_code == 422


def test_configure_watch_invalid_threshold(monkeypatch):
    """alert_threshold outside (0, 1] is rejected."""
    aoi_id = uuid4()
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=_make_aoi(id=aoi_id)))

    response = client.put(
        f"/aois/{aoi_id}/watch",
        json={"enabled": True, "interval_minutes": 30, "alert_threshold": 1.5},
    )
    assert response.status_code == 422


def test_configure_watch_not_found(monkeypatch):
    """Returns 404 when AOI does not exist."""
    monkeypatch.setattr(aois_routes.repo, "set_aoi_watch", MagicMock(return_value=None))

    response = client.put(
        f"/aois/{uuid4()}/watch",
        json={"enabled": True, "interval_minutes": 30, "alert_threshold": 0.5},
    )
    assert response.status_code == 404


# ── GET /aois/watchlist ───────────────────────────────────────────────────────


def test_get_watchlist_empty(monkeypatch):
    """Returns empty watchlist when no AOIs are watched."""
    monkeypatch.setattr(aois_routes.repo, "list_watched_aois", MagicMock(return_value=[]))

    response = client.get("/aois/watchlist")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["items"] == []


def test_get_watchlist_with_items(monkeypatch):
    """Returns watched AOIs with alert_active derived from spread_prob vs threshold."""
    aois = [
        _make_aoi(
            id=uuid4(),
            name="Alert AOI",
            watch_enabled=True,
            watch_interval_minutes=30,
            watch_alert_threshold=0.5,
            watch_last_spread_prob=0.75,  # exceeds threshold → alert_active=True
        ),
        _make_aoi(
            id=uuid4(),
            name="Calm AOI",
            watch_enabled=True,
            watch_interval_minutes=60,
            watch_alert_threshold=0.7,
            watch_last_spread_prob=0.3,  # below threshold → alert_active=False
        ),
    ]
    monkeypatch.setattr(aois_routes.repo, "list_watched_aois", MagicMock(return_value=aois))

    response = client.get("/aois/watchlist")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2

    alert_items = [i for i in data["items"] if i["name"] == "Alert AOI"]
    calm_items = [i for i in data["items"] if i["name"] == "Calm AOI"]

    assert len(alert_items) == 1
    assert alert_items[0]["alert_active"] is True

    assert len(calm_items) == 1
    assert calm_items[0]["alert_active"] is False


def test_get_watchlist_no_data_yet(monkeypatch):
    """alert_active is False when last_spread_prob is None (not yet checked)."""
    aoi = _make_aoi(
        id=uuid4(),
        watch_enabled=True,
        watch_interval_minutes=30,
        watch_alert_threshold=0.5,
        watch_last_spread_prob=None,
    )
    monkeypatch.setattr(aois_routes.repo, "list_watched_aois", MagicMock(return_value=[aoi]))

    response = client.get("/aois/watchlist")
    assert response.status_code == 200
    assert response.json()["items"][0]["alert_active"] is False


# ── Rate limiting — alert_active correctly reflects threshold boundary ────────


def test_alert_active_at_exact_threshold(monkeypatch):
    """alert_active is True when spread_prob == threshold (>= comparison)."""
    aoi = _make_aoi(
        id=uuid4(),
        watch_enabled=True,
        watch_interval_minutes=30,
        watch_alert_threshold=0.5,
        watch_last_spread_prob=0.5,  # exactly at threshold
    )
    monkeypatch.setattr(aois_routes.repo, "list_watched_aois", MagicMock(return_value=[aoi]))

    response = client.get("/aois/watchlist")
    assert response.json()["items"][0]["alert_active"] is True


def test_alert_active_just_below_threshold(monkeypatch):
    """alert_active is False when spread_prob is just below threshold."""
    aoi = _make_aoi(
        id=uuid4(),
        watch_enabled=True,
        watch_interval_minutes=30,
        watch_alert_threshold=0.5,
        watch_last_spread_prob=0.499,
    )
    monkeypatch.setattr(aois_routes.repo, "list_watched_aois", MagicMock(return_value=[aoi]))

    response = client.get("/aois/watchlist")
    assert response.json()["items"][0]["alert_active"] is False
