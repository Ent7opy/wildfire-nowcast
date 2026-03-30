"""Tests for GET /fires/detections/{detection_id} with weather context."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.core.weather import classify_rh_fire_risk
from api.deps import get_fire_repo
from api.fires.repository import FireRepository
from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DETECTION = {
    "id": 42,
    "lat": 38.5,
    "lon": -122.8,
    "acq_time": datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc),
    "confidence": 85.0,
    "brightness": 340.2,
    "bright_t31": 310.5,
    "frp": 25.3,
    "sensor": "VIIRS",
    "source": "FIRMS",
    "confidence_score": 0.85,
    "persistence_score": 0.7,
    "landcover_score": 0.6,
    "weather_score": 0.5,
    "false_source_masked": False,
    "fire_likelihood": 0.78,
    "denoised_score": 0.9,
    "is_noise": False,
    "event_id": "evt_001",
    "event_score": 0.92,
    "denoiser_decision": "fire",
    "review_required": False,
}

_WEATHER_CONTEXT = {
    "wind_speed_ms": 12.4,
    "wind_direction_deg": 230.0,
    "relative_humidity_pct": 18.0,
    "rh_fire_risk": "elevated",
    "temperature_c": 36.2,
    "precip_mm_24h": 0.0,
    "source_run_time": "2026-03-28T06:00:00+00:00",
    "data_age_hours": 6.0,
    "resolution_note": "GFS 0.25\u00b0 \u2014 nearest grid point (~25 km)",
    "bias_correction": {
        "applied": True,
        "method": "affine (fitted against ERA5 reanalysis)",
        "variables": ["u10", "v10", "t2m", "rh2m"],
    },
}


def _make_repo(**method_overrides) -> FireRepository:
    repo = MagicMock(spec=FireRepository)
    repo.validate_bbox.return_value = None
    for name, value in method_overrides.items():
        getattr(repo, name).return_value = value
    return repo


# ---------------------------------------------------------------------------
# Happy path — weather data available
# ---------------------------------------------------------------------------

@patch("api.routes.fires.get_weather_context_for_point", return_value=_WEATHER_CONTEXT)
def test_detection_detail_with_weather(mock_wx, monkeypatch):
    """Full detection detail response includes weather block."""
    fake_repo = _make_repo(get_fire_detection_by_id=_DETECTION.copy())
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get("/fires/detections/42")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == 42
    assert payload["weather"] is not None
    assert payload["weather"]["wind_speed_ms"] == 12.4
    assert payload["weather"]["wind_direction_deg"] == 230.0
    assert payload["weather"]["relative_humidity_pct"] == 18.0
    assert payload["weather"]["rh_fire_risk"] == "elevated"
    assert payload["weather"]["temperature_c"] == 36.2
    assert payload["weather"]["precip_mm_24h"] == 0.0
    assert payload["weather"]["source_run_time"] == "2026-03-28T06:00:00+00:00"
    assert payload["weather"]["data_age_hours"] == 6.0
    assert "GFS" in payload["weather"]["resolution_note"]
    assert payload["weather"]["bias_correction"]["applied"] is True
    assert payload["weather_unavailable_reason"] is None

    mock_wx.assert_called_once_with(
        lat=38.5, lon=-122.8, ref_time=_DETECTION["acq_time"]
    )


# ---------------------------------------------------------------------------
# Null case — no weather data
# ---------------------------------------------------------------------------

@patch("api.routes.fires.get_weather_context_for_point", return_value=None)
def test_detection_detail_weather_null(mock_wx, monkeypatch):
    """When no weather data covers the point, weather is null with a reason."""
    fake_repo = _make_repo(get_fire_detection_by_id=_DETECTION.copy())
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get("/fires/detections/42")

    assert response.status_code == 200
    payload = response.json()
    assert payload["weather"] is None
    assert payload["weather_unavailable_reason"] is not None
    assert "GFS" in payload["weather_unavailable_reason"]


# ---------------------------------------------------------------------------
# 404 — detection not found
# ---------------------------------------------------------------------------

def test_detection_detail_not_found(monkeypatch):
    """Unknown detection ID returns 404."""
    fake_repo = _make_repo(get_fire_detection_by_id=None)
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get("/fires/detections/999999")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# RH fire-risk thresholds
# ---------------------------------------------------------------------------

def test_rh_critical_below_15():
    assert classify_rh_fire_risk(14.9) == "critical"
    assert classify_rh_fire_risk(0.0) == "critical"


def test_rh_elevated_below_25():
    assert classify_rh_fire_risk(15.0) == "elevated"
    assert classify_rh_fire_risk(24.9) == "elevated"


def test_rh_normal_at_25_and_above():
    assert classify_rh_fire_risk(25.0) == "normal"
    assert classify_rh_fire_risk(80.0) == "normal"


def test_rh_boundary_exactly_15():
    """15% is the boundary between critical and elevated — 15 is elevated."""
    assert classify_rh_fire_risk(15.0) == "elevated"


def test_rh_boundary_exactly_25():
    """25% is the boundary between elevated and normal — 25 is normal."""
    assert classify_rh_fire_risk(25.0) == "normal"


# ---------------------------------------------------------------------------
# Bias correction is present in weather response
# ---------------------------------------------------------------------------

@patch("api.routes.fires.get_weather_context_for_point", return_value=_WEATHER_CONTEXT)
def test_bias_correction_in_response(mock_wx, monkeypatch):
    """Response weather block includes bias correction metadata."""
    fake_repo = _make_repo(get_fire_detection_by_id=_DETECTION.copy())
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get("/fires/detections/42")
    bc = response.json()["weather"]["bias_correction"]

    assert bc["applied"] is True
    assert "ERA5" in bc["method"]
    assert "u10" in bc["variables"]
    assert "rh2m" in bc["variables"]


# ---------------------------------------------------------------------------
# Cache-control header
# ---------------------------------------------------------------------------

@patch("api.routes.fires.get_weather_context_for_point", return_value=None)
def test_detection_detail_cache_header(mock_wx, monkeypatch):
    fake_repo = _make_repo(get_fire_detection_by_id=_DETECTION.copy())
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get("/fires/detections/42")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "max-age=60"
