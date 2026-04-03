"""Tests for check_new_ignition in ingest.aoi_watch."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4


from ingest.aoi_watch import check_new_ignition

_NOW = datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
_TWO_H_AGO = _NOW - timedelta(hours=2)
_THREE_H_AGO = _NOW - timedelta(hours=3)

# A minimal GeoJSON polygon that wraps roughly (0°,0°) to (1°,1°).
_AOI_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
}


def _make_aoi(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": uuid4(),
        "name": "Test AOI",
        "geometry": _AOI_GEOMETRY,
        "watch_enabled": True,
        "watch_interval_minutes": 30,
        "watch_alert_threshold": 0.5,
        "watch_last_checked_at": None,
        "watch_last_alerted_at": None,
        "watch_last_spread_prob": None,
    }
    base.update(overrides)
    return base


def _make_detection(
    *,
    lat: float = 0.5,
    lon: float = 0.5,
    denoised_score: float = 0.85,
    is_noise: bool = False,
    false_source_masked: bool = False,
    event_id: str | None = None,
    event_started_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "lat": lat,
        "lon": lon,
        "denoised_score": denoised_score,
        "is_noise": is_noise,
        "false_source_masked": false_source_masked,
        "event_id": event_id,
        "event_started_at": event_started_at,
    }


class _MockMappings:
    """Mimics SQLAlchemy .mappings().all() result."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> "_MockMappings":
        return self

    def all(self) -> list[dict[str, Any]]:
        return self._rows


class _MockConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def execute(self, stmt, params=None) -> _MockMappings:
        return _MockMappings(self._rows)


class _MockEngine:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    @contextmanager
    def begin(self):
        yield _MockConn(self._rows)


# ── qualifying cluster → notify called ────────────────────────────────────────


def test_qualifying_cluster_fires_notify() -> None:
    """Two qualifying detections close together trigger a new_ignition notification."""
    aoi = _make_aoi()
    aoi_id = str(aoi["id"])

    detections = [
        _make_detection(lat=0.5, lon=0.5, denoised_score=0.90),
        _make_detection(lat=0.501, lon=0.501, denoised_score=0.80),  # ~140 m apart
    ]
    engine = _MockEngine(detections)

    mock_notify = MagicMock()
    with patch("ingest.aoi_watch.notify", mock_notify):
        result = check_new_ignition(aoi, engine=engine, _now=_NOW)

    assert result is not None, "Expected a cluster result dict"
    mock_notify.assert_called_once()

    call_kwargs = mock_notify.call_args.kwargs if mock_notify.call_args.kwargs else {}
    call_args = mock_notify.call_args.args if mock_notify.call_args.args else ()

    # event_type is the first positional arg.
    event_type = call_args[0] if call_args else call_kwargs.get("event_type", "")
    assert event_type == f"new_ignition:{aoi_id}"

    assert call_kwargs.get("severity") == "critical"
    assert float(call_kwargs.get("denoised_score", 0)) > 0
    assert call_kwargs.get("aoi_id") == aoi_id
    assert int(call_kwargs.get("detection_count", 0)) >= 2


def test_qualifying_cluster_returns_cluster_info() -> None:
    """Return dict contains the expected keys and valid values."""
    aoi = _make_aoi()
    detections = [
        _make_detection(lat=0.5, lon=0.5, denoised_score=0.95),
        _make_detection(lat=0.501, lon=0.501, denoised_score=0.75),
    ]
    engine = _MockEngine(detections)

    with patch("ingest.aoi_watch.notify"):
        result = check_new_ignition(aoi, engine=engine)

    assert result is not None
    assert result["aoi_id"] == str(aoi["id"])
    assert result["detection_count"] >= 2
    assert result["max_denoised_score"] > 0
    assert isinstance(result["centroid_lat"], float)
    assert isinstance(result["centroid_lon"], float)


# ── no detections → notify not called ─────────────────────────────────────────


def test_no_detections_returns_none_and_no_notify() -> None:
    """When the DB returns no detections, check_new_ignition returns None without notifying."""
    aoi = _make_aoi()
    engine = _MockEngine([])

    mock_notify = MagicMock()
    with patch("ingest.aoi_watch.notify", mock_notify):
        result = check_new_ignition(aoi, engine=engine, _now=_NOW)

    assert result is None
    mock_notify.assert_not_called()


# ── detections are noise → notify not called ──────────────────────────────────


def test_noise_detections_excluded_by_query() -> None:
    """Detections with is_noise=True are excluded at the SQL query level.

    The DB query filters them out before we ever see them.  We simulate this
    by returning an empty result (as the real DB would).
    """
    aoi = _make_aoi()
    # Simulate DB returning no rows because the WHERE clause excluded noise.
    engine = _MockEngine([])

    mock_notify = MagicMock()
    with patch("ingest.aoi_watch.notify", mock_notify):
        result = check_new_ignition(aoi, engine=engine, _now=_NOW)

    assert result is None
    mock_notify.assert_not_called()


def test_noise_detections_single_qualifying_below_cluster_threshold() -> None:
    """Only one qualifying detection — not enough for a cluster of ≥ 2."""
    aoi = _make_aoi()
    detections = [
        _make_detection(lat=0.5, lon=0.5, denoised_score=0.90, is_noise=False),
    ]
    engine = _MockEngine(detections)

    mock_notify = MagicMock()
    with patch("ingest.aoi_watch.notify", mock_notify):
        result = check_new_ignition(aoi, engine=engine, _now=_NOW)

    assert result is None
    mock_notify.assert_not_called()


# ── old fire event → notify not called ────────────────────────────────────────


def test_detections_with_old_event_suppressed() -> None:
    """Detections linked to a fire_event that started > 2 h ago are treated as existing fires."""
    aoi = _make_aoi()
    old_event_start = _NOW - timedelta(hours=5)  # > 2 h ago

    detections = [
        _make_detection(
            lat=0.5, lon=0.5, denoised_score=0.90,
            event_id=str(uuid4()), event_started_at=old_event_start,
        ),
        _make_detection(
            lat=0.501, lon=0.501, denoised_score=0.85,
            event_id=str(uuid4()), event_started_at=old_event_start,
        ),
    ]
    engine = _MockEngine(detections)

    mock_notify = MagicMock()
    with patch("ingest.aoi_watch.notify", mock_notify):
        result = check_new_ignition(aoi, engine=engine, _now=_NOW)

    assert result is None
    mock_notify.assert_not_called()


def test_detections_with_recent_event_qualifies() -> None:
    """Detections linked to an event that started < 2 h ago are treated as new ignitions."""
    aoi = _make_aoi()
    recent_event_start = _NOW - timedelta(hours=1)  # < 2 h ago — qualifies

    detections = [
        _make_detection(
            lat=0.5, lon=0.5, denoised_score=0.90,
            event_id=str(uuid4()), event_started_at=recent_event_start,
        ),
        _make_detection(
            lat=0.501, lon=0.501, denoised_score=0.85,
            event_id=str(uuid4()), event_started_at=recent_event_start,
        ),
    ]
    engine = _MockEngine(detections)

    mock_notify = MagicMock()
    with patch("ingest.aoi_watch.notify", mock_notify):
        result = check_new_ignition(aoi, engine=engine, _now=_NOW)

    assert result is not None
    mock_notify.assert_called_once()


# ── cluster too sparse → notify not called ────────────────────────────────────


def test_detections_too_far_apart_no_cluster() -> None:
    """Two detections more than 1 km apart do not form a qualifying cluster."""
    aoi = _make_aoi()
    # ~0.05° lat ≈ 5.5 km apart — well beyond the 1 km cluster radius.
    detections = [
        _make_detection(lat=0.1, lon=0.1, denoised_score=0.90),
        _make_detection(lat=0.15, lon=0.1, denoised_score=0.85),
    ]
    engine = _MockEngine(detections)

    mock_notify = MagicMock()
    with patch("ingest.aoi_watch.notify", mock_notify):
        result = check_new_ignition(aoi, engine=engine, _now=_NOW)

    assert result is None
    mock_notify.assert_not_called()


# ── missing geometry → graceful return None ───────────────────────────────────


def test_aoi_without_geometry_returns_none() -> None:
    """AOI with no geometry key returns None without raising."""
    aoi = _make_aoi(geometry=None)
    engine = _MockEngine([])

    mock_notify = MagicMock()
    with patch("ingest.aoi_watch.notify", mock_notify):
        result = check_new_ignition(aoi, engine=engine, _now=_NOW)

    assert result is None
    mock_notify.assert_not_called()


# ── db exception → graceful return None ───────────────────────────────────────


def test_db_exception_returns_none() -> None:
    """DB errors are caught and None is returned without raising."""
    aoi = _make_aoi()

    class _BrokenEngine:
        @contextmanager
        def begin(self):
            raise RuntimeError("connection refused")
            yield  # pragma: no cover

    mock_notify = MagicMock()
    with patch("ingest.aoi_watch.notify", mock_notify):
        result = check_new_ignition(aoi, engine=_BrokenEngine())

    assert result is None
    mock_notify.assert_not_called()
