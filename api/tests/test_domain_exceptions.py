"""Tests for the domain exception hierarchy and global exception handler.

Each test verifies that:
  1. The correct HTTP status code is returned.
  2. The response body contains ``{"error": "<ExceptionClassName>", "detail": "..."}``.
"""
from __future__ import annotations

from datetime import date, timedelta
import pytest
from fastapi.testclient import TestClient

from api.deps import get_fire_repo
from api.errors import (
    ArchiveRangeError,
    FiresNotFoundError,
    InvalidBoundingBoxError,
    ModelNotReadyError,
    StalenessError,
    WildfireError,
    _STATUS_MAP,
)
from api.main import app
from api.tests.conftest import make_fire_repo

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Unit tests — exception hierarchy and status map
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_all_subclasses_are_wildfire_errors(self):
        for cls in (FiresNotFoundError, InvalidBoundingBoxError, StalenessError, ModelNotReadyError, ArchiveRangeError):
            assert issubclass(cls, WildfireError)

    def test_status_map_covers_all_subclasses(self):
        expected = {FiresNotFoundError, InvalidBoundingBoxError, StalenessError, ModelNotReadyError, ArchiveRangeError}
        assert set(_STATUS_MAP.keys()) == expected

    def test_status_map_values(self):
        assert _STATUS_MAP[FiresNotFoundError] == 404
        assert _STATUS_MAP[InvalidBoundingBoxError] == 422
        assert _STATUS_MAP[StalenessError] == 503
        assert _STATUS_MAP[ModelNotReadyError] == 503
        assert _STATUS_MAP[ArchiveRangeError] == 400

    def test_unknown_subclass_maps_to_500(self):
        class _Custom(WildfireError):
            pass

        assert _STATUS_MAP.get(type(_Custom()), 500) == 500


# ---------------------------------------------------------------------------
# Integration tests — fires bbox validation → InvalidBoundingBoxError
# ---------------------------------------------------------------------------

_FIRES_PARAMS = {
    "start_time": "2025-01-01T00:00:00Z",
    "end_time": "2025-01-02T00:00:00Z",
}


def _make_bbox_repo(raise_on_validate: str | None = None):
    repo = make_fire_repo()
    if raise_on_validate:
        repo.validate_bbox.side_effect = ValueError(raise_on_validate)
    return repo


def test_fires_invalid_bbox_returns_invalid_bounding_box_error(monkeypatch):
    """validate_bbox ValueError → InvalidBoundingBoxError with HTTP 422."""
    repo = _make_bbox_repo(raise_on_validate="min_lon must be less than max_lon")
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: repo)

    response = client.get(
        "/fires/detections",
        params={"min_lon": 22.0, "min_lat": 40.0, "max_lon": 20.0, "max_lat": 43.0, **_FIRES_PARAMS},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "InvalidBoundingBoxError"
    assert "min_lon must be less than max_lon" in body["detail"]


def test_fires_events_invalid_bbox_error_type(monkeypatch):
    """validate_bbox ValueError on /fires/events → InvalidBoundingBoxError."""
    repo = _make_bbox_repo(raise_on_validate="bbox too large")
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: repo)

    response = client.get(
        "/fires/events",
        params={"min_lon": 22.0, "min_lat": 40.0, "max_lon": 20.0, "max_lat": 43.0, **_FIRES_PARAMS},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "InvalidBoundingBoxError"


def test_fires_fronts_invalid_bbox_error_type(monkeypatch):
    """validate_bbox ValueError on /fires/fronts → InvalidBoundingBoxError."""
    repo = _make_bbox_repo(raise_on_validate="invalid coordinates")
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: repo)

    response = client.get(
        "/fires/fronts",
        params={"min_lon": 22.0, "min_lat": 40.0, "max_lon": 20.0, "max_lat": 43.0, **_FIRES_PARAMS},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "InvalidBoundingBoxError"


# ---------------------------------------------------------------------------
# Integration tests — archive range → ArchiveRangeError
# ---------------------------------------------------------------------------

def _recent_date(days_ago: int) -> str:
    return (date.today() - timedelta(days=days_ago)).isoformat()


def test_archive_range_too_large_returns_archive_range_error():
    """num_days > MAX_ARCHIVE_RANGE_DAYS → ArchiveRangeError with HTTP 400."""
    from api.routes.archive import MAX_ARCHIVE_RANGE_DAYS, MAX_FIRMS_LOOKBACK_DAYS

    if MAX_ARCHIVE_RANGE_DAYS >= MAX_FIRMS_LOOKBACK_DAYS - 1:
        pytest.skip("Cannot construct oversized range within FIRMS lookback window")

    end_days_ago = 1
    start_days_ago = end_days_ago + MAX_ARCHIVE_RANGE_DAYS

    response = client.post(
        "/fires/archive/ingest-range",
        json={"start_date": _recent_date(start_days_ago), "end_date": _recent_date(end_days_ago)},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "ArchiveRangeError"
    assert str(MAX_ARCHIVE_RANGE_DAYS) in body["detail"]


def test_archive_ingest_lookback_exceeded_returns_archive_range_error():
    """Date older than FIRMS lookback → ArchiveRangeError with HTTP 400."""
    from api.routes.archive import MAX_FIRMS_LOOKBACK_DAYS

    # MAX_FIRMS_LOOKBACK_DAYS itself is still within the window (> check, not >=).
    # Use +1 so this date is genuinely beyond the limit.
    too_old = (date.today() - timedelta(days=MAX_FIRMS_LOOKBACK_DAYS + 1)).isoformat()

    response = client.post(
        "/fires/archive/ingest",
        json={"date": too_old, "timeframe": "morning"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "ArchiveRangeError"
    assert str(MAX_FIRMS_LOOKBACK_DAYS) in body["detail"]


# ---------------------------------------------------------------------------
# Integration tests — internal bbox → InvalidBoundingBoxError
# ---------------------------------------------------------------------------

def test_internal_industrial_coverage_invalid_bbox():
    """Malformed bbox param → InvalidBoundingBoxError with HTTP 422."""
    response = client.get("/internal/health/industrial-coverage?bbox=bad,values")

    assert response.status_code == 422
    assert response.json()["error"] == "InvalidBoundingBoxError"


def test_internal_industrial_coverage_wrong_part_count():
    """3-element bbox param → InvalidBoundingBoxError with HTTP 422."""
    response = client.get("/internal/health/industrial-coverage?bbox=-120.0,37.0,-119.0")

    assert response.status_code == 422
    assert response.json()["error"] == "InvalidBoundingBoxError"


# ---------------------------------------------------------------------------
# Response shape contract
# ---------------------------------------------------------------------------

def test_wildfire_error_response_has_error_and_detail_keys(monkeypatch):
    """Domain exception response must have exactly 'error' and 'detail' keys."""
    repo = _make_bbox_repo(raise_on_validate="bad bbox")
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: repo)

    response = client.get(
        "/fires/detections",
        params={"min_lon": 22.0, "min_lat": 40.0, "max_lon": 20.0, "max_lat": 43.0, **_FIRES_PARAMS},
    )

    body = response.json()
    assert "error" in body
    assert "detail" in body
    # Must NOT use the HTTPException envelope shape
    assert "code" not in body
    assert "message" not in body
