from datetime import datetime, timezone
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.deps import get_fire_repo
from api.main import app
from api.tests.conftest import make_fire_repo as _make_repo

client = TestClient(app)


# ---------------------------------------------------------------------------
# Tests — detections
# ---------------------------------------------------------------------------

def test_get_detections_endpoint_basic(monkeypatch):
    """Test that the /fires/detections endpoint works and calls the repo helper."""
    mock_detections = [
        {"id": 1, "lat": 42.0, "lon": 21.0, "acq_time": datetime(2025, 1, 1, tzinfo=timezone.utc)}
    ]
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={
            "data": mock_detections,
            "next_cursor": None,
            "has_more": False,
            "limit": 1000,
        }
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get(
        "/fires/detections",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert len(data["detections"]) == 1

    _, kwargs = fake_repo.async_list_fire_detections_bbox_time.call_args
    assert kwargs["include_noise"] is False
    assert kwargs["min_confidence"] is None
    assert "denoised_score" not in kwargs["columns"]


def test_get_fires_endpoint_alias(monkeypatch):
    """Test that the /fires endpoint aliases /fires/detections."""
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get(
        "/fires",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["detections"] == []


def test_get_detections_endpoint_with_min_confidence(monkeypatch):
    """Test that min_confidence is passed to repo."""
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    client.get(
        "/fires/detections",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
            "min_confidence": 80,
        },
    )

    _, kwargs = fake_repo.async_list_fire_detections_bbox_time.call_args
    assert kwargs["min_confidence"] == 80.0


def test_get_detections_endpoint_with_denoiser_fields(monkeypatch):
    """Test that include_denoiser_fields adds columns."""
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    client.get(
        "/fires/detections",
        params={
            "min_lon": 20.0, "min_lat": 40.0, "max_lon": 22.0, "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
            "include_denoiser_fields": "true",
        },
    )

    _, kwargs = fake_repo.async_list_fire_detections_bbox_time.call_args
    assert "denoised_score" in kwargs["columns"]
    assert "is_noise" in kwargs["columns"]
    assert "event_id" in kwargs["columns"]
    assert "event_score" in kwargs["columns"]
    assert "denoiser_decision" in kwargs["columns"]
    assert "review_required" in kwargs["columns"]


def test_get_detections_endpoint_with_include_noise(monkeypatch):
    """Test that include_noise is passed to repo."""
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    client.get(
        "/fires/detections",
        params={
            "min_lon": 20.0, "min_lat": 40.0, "max_lon": 22.0, "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
            "include_noise": "true",
        },
    )

    _, kwargs = fake_repo.async_list_fire_detections_bbox_time.call_args
    assert kwargs["include_noise"] is True


# ---------------------------------------------------------------------------
# Tests — events
# ---------------------------------------------------------------------------

def test_get_events_endpoint(monkeypatch):
    """Test that /fires/events delegates to repo.async_list_fire_events_bbox_time."""
    fake_repo = _make_repo(
        async_list_fire_events_bbox_time={"data": [{"event_id": "evt_1", "event_score": 0.95}], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get(
        "/fires/events",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
            "min_event_score": 0.5,
            "include_review_required": "false",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["events"][0]["event_id"] == "evt_1"
    assert payload["next_cursor"] is None
    assert payload["has_more"] is False

    _, kwargs = fake_repo.async_list_fire_events_bbox_time.call_args
    assert kwargs["min_event_score"] == 0.5
    assert kwargs["include_review_required"] is False


def test_get_events_endpoint_passthrough_geom_geojson(monkeypatch):
    """Test that /fires/events returns event geometry payload when provided by repo."""
    fake_repo = _make_repo(
        async_list_fire_events_bbox_time={"data": [
            {
                "event_id": "evt_geom_1",
                "event_score": 0.93,
                "geom_geojson": '{"type":"MultiPolygon","coordinates":[]}',
            }
        ], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get(
        "/fires/events",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["events"][0]["event_id"] == "evt_geom_1"
    assert payload["events"][0]["geom_geojson"] == '{"type":"MultiPolygon","coordinates":[]}'


def test_get_events_endpoint_passthrough_geom_provenance(monkeypatch):
    """Test that /fires/events preserves geometry provenance fields from repo."""
    fake_repo = _make_repo(
        async_list_fire_events_bbox_time={"data": [
            {
                "event_id": "evt_geom_2",
                "geom_source": "estimated",
                "geom_method": "estimated_concave",
                "geom_quality": 0.42,
                "authority_profile": None,
                "authoritative_perimeter_id": None,
            }
        ], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get(
        "/fires/events",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    event = payload["events"][0]
    assert event["geom_source"] == "estimated"
    assert event["geom_method"] == "estimated_concave"
    assert event["geom_quality"] == 0.42
    assert event["authority_profile"] is None
    assert event["authoritative_perimeter_id"] is None


# ---------------------------------------------------------------------------
# Tests — fronts
# ---------------------------------------------------------------------------

def test_get_fronts_endpoint(monkeypatch):
    """Test that /fires/fronts delegates to repo.async_list_fire_fronts_bbox_time."""
    fake_repo = _make_repo(
        async_list_fire_fronts_bbox_time={"data": [{"front_id": "front_1", "event_id": "evt_1"}], "next_cursor": None, "has_more": False, "limit": 800}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get(
        "/fires/fronts",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
            "min_event_score": 0.5,
            "include_review_required": "false",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["fronts"][0]["front_id"] == "front_1"
    assert payload["next_cursor"] is None
    assert payload["has_more"] is False

    _, kwargs = fake_repo.async_list_fire_fronts_bbox_time.call_args
    assert kwargs["min_event_score"] == 0.5
    assert kwargs["include_review_required"] is False


def test_get_fronts_endpoint_passthrough_geom_provenance(monkeypatch):
    """Test that /fires/fronts preserves geometry provenance fields from repo."""
    fake_repo = _make_repo(
        async_list_fire_fronts_bbox_time={"data": [
            {
                "front_id": "front_geom_1",
                "event_id": "evt_geom_2",
                "geom_source": "authoritative",
                "geom_method": "authoritative",
                "geom_quality": 0.97,
                "authority_profile": "wfigs_us",
                "authoritative_perimeter_id": 12345,
            }
        ], "next_cursor": None, "has_more": False, "limit": 800}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get(
        "/fires/fronts",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    front = payload["fronts"][0]
    assert front["geom_source"] == "authoritative"
    assert front["geom_method"] == "authoritative"
    assert front["geom_quality"] == 0.97
    assert front["authority_profile"] == "wfigs_us"
    assert front["authoritative_perimeter_id"] == 12345


# ---------------------------------------------------------------------------
# Tests — cache control headers
# ---------------------------------------------------------------------------

def test_detections_cache_control_header(monkeypatch):
    """Verify /fires/detections sets Cache-Control: max-age=60 (E4)."""
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get(
        "/fires/detections",
        params={
            "min_lon": 20.0, "min_lat": 40.0, "max_lon": 22.0, "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("cache-control") == "max-age=60"


def test_fires_alias_cache_control_header(monkeypatch):
    """Verify /fires alias also sets Cache-Control: max-age=60 (E4)."""
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get(
        "/fires",
        params={
            "min_lon": 20.0, "min_lat": 40.0, "max_lon": 22.0, "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("cache-control") == "max-age=60"


# ---------------------------------------------------------------------------
# Tests — reverse geocode
# ---------------------------------------------------------------------------

def test_get_reverse_geocode_endpoint(monkeypatch):
    """Test that /fires/reverse-geocode delegates to reverse_geocode_point."""
    import api.routes.fires as fires_module

    mock_reverse = MagicMock(
        return_value={
            "lat": 34.1,
            "lon": -118.2,
            "provider": "nominatim",
            "cache_hit": False,
            "status": "resolved",
            "location_name": "California, United States",
        }
    )
    monkeypatch.setattr(fires_module, "reverse_geocode_point", mock_reverse)

    response = client.get(
        "/fires/reverse-geocode",
        params={"lat": 34.1, "lon": -118.2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "resolved"
    assert payload["location_name"] == "California, United States"
    _, kwargs = mock_reverse.call_args
    assert kwargs["lat"] == 34.1
    assert kwargs["lon"] == -118.2


def test_get_reverse_geocode_endpoint_bad_request(monkeypatch):
    """Test ValueError from service is mapped to HTTP 400."""
    import api.routes.fires as fires_module

    mock_reverse = MagicMock(side_effect=ValueError("Unsupported geocoding provider: bad"))
    monkeypatch.setattr(fires_module, "reverse_geocode_point", mock_reverse)

    response = client.get(
        "/fires/reverse-geocode",
        params={"lat": 34.1, "lon": -118.2},
    )

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Tests — FakeFireRepo injection (acceptance criterion: no DB required)
# ---------------------------------------------------------------------------

class FakeFireRepo:
    """Minimal hand-written stub used to verify Depends injection works end-to-end."""

    def validate_bbox(self, bbox):
        pass

    async def async_list_fire_detections_bbox_time(self, bbox, start_time, end_time, **kwargs):
        return {
            "data": [{"id": 99, "lat": 50.0, "lon": 10.0}],
            "next_cursor": None,
            "has_more": False,
            "limit": 500,
        }

    async def async_list_fire_events_bbox_time(self, bbox, start_time, end_time, **kwargs):
        return {"data": [], "next_cursor": None, "has_more": False, "limit": 1000}

    async def async_list_fire_fronts_bbox_time(self, bbox, start_time, end_time, **kwargs):
        return {"data": [], "next_cursor": None, "has_more": False, "limit": 2000}


def test_fake_fire_repo_injection_no_db(monkeypatch):
    """Routes must work with a hand-written FakeFireRepo — no database needed."""
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: FakeFireRepo())

    response = client.get(
        "/fires/detections",
        params={
            "min_lon": 9.0,
            "min_lat": 49.0,
            "max_lon": 11.0,
            "max_lat": 51.0,
            "start_time": "2025-06-01T00:00:00Z",
            "end_time": "2025-06-02T00:00:00Z",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["detections"][0]["id"] == 99


# ---------------------------------------------------------------------------
# Tests — cursor pagination
# ---------------------------------------------------------------------------

_BASE_PARAMS = {
    "min_lon": 20.0,
    "min_lat": 40.0,
    "max_lon": 22.0,
    "max_lat": 43.0,
    "start_time": "2025-01-01T00:00:00Z",
    "end_time": "2025-01-02T00:00:00Z",
}


def test_detections_response_includes_pagination_fields(monkeypatch):
    """Response must include next_cursor and has_more even when empty."""
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get("/fires/detections", params=_BASE_PARAMS)

    assert response.status_code == 200
    payload = response.json()
    assert "next_cursor" in payload
    assert "has_more" in payload
    assert payload["next_cursor"] is None
    assert payload["has_more"] is False


def test_detections_cursor_param_forwarded_to_repo(monkeypatch):
    """cursor query param must be forwarded to repo.async_list_fire_detections_bbox_time."""
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    client.get("/fires/detections", params={**_BASE_PARAMS, "cursor": "abc123"})

    _, kwargs = fake_repo.async_list_fire_detections_bbox_time.call_args
    assert kwargs["cursor"] == "abc123"


def test_detections_offset_param_forwarded_to_repo(monkeypatch):
    """Deprecated offset query param must be forwarded to repo."""
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    client.get("/fires/detections", params={**_BASE_PARAMS, "offset": 500})

    _, kwargs = fake_repo.async_list_fire_detections_bbox_time.call_args
    assert kwargs["offset"] == 500


def test_detections_response_has_more_true(monkeypatch):
    """When repo signals has_more=True, next_cursor must be non-None in the response."""
    fake_repo = _make_repo(
        async_list_fire_detections_bbox_time={
            "data": [{"id": 42, "lat": 41.0, "lon": 21.0}],
            "next_cursor": "dGVzdGN1cnNvcg==",
            "has_more": True,
            "limit": 1000,
        }
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get("/fires/detections", params=_BASE_PARAMS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_more"] is True
    assert payload["next_cursor"] == "dGVzdGN1cnNvcg=="


def test_detections_invalid_cursor_returns_400(monkeypatch):
    """A malformed cursor must produce HTTP 400."""
    fake_repo = _make_repo()
    fake_repo.async_list_fire_detections_bbox_time.side_effect = ValueError("Invalid cursor: ...")
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get("/fires/detections", params={**_BASE_PARAMS, "cursor": "!!!notvalid!!!"})

    assert response.status_code == 400


def test_events_cursor_param_forwarded_to_repo(monkeypatch):
    """cursor query param on /fires/events must be forwarded to repo."""
    fake_repo = _make_repo(
        async_list_fire_events_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    client.get("/fires/events", params={**_BASE_PARAMS, "cursor": "evtcursor"})

    _, kwargs = fake_repo.async_list_fire_events_bbox_time.call_args
    assert kwargs["cursor"] == "evtcursor"


def test_events_offset_param_forwarded_to_repo(monkeypatch):
    """Deprecated offset query param on /fires/events must be forwarded to repo."""
    fake_repo = _make_repo(
        async_list_fire_events_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 1000}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    client.get("/fires/events", params={**_BASE_PARAMS, "offset": 200})

    _, kwargs = fake_repo.async_list_fire_events_bbox_time.call_args
    assert kwargs["offset"] == 200


def test_events_response_has_more_true(monkeypatch):
    """When repo signals has_more=True, next_cursor must appear in events response."""
    fake_repo = _make_repo(
        async_list_fire_events_bbox_time={
            "data": [{"event_id": "evt_x"}],
            "next_cursor": "ZXZ0Y3Vyc29y",
            "has_more": True,
            "limit": 1000,
        }
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get("/fires/events", params=_BASE_PARAMS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_more"] is True
    assert payload["next_cursor"] == "ZXZ0Y3Vyc29y"


def test_fronts_cursor_param_forwarded_to_repo(monkeypatch):
    """cursor query param on /fires/fronts must be forwarded to repo."""
    fake_repo = _make_repo(
        async_list_fire_fronts_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 800}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    client.get("/fires/fronts", params={**_BASE_PARAMS, "cursor": "frontcursor"})

    _, kwargs = fake_repo.async_list_fire_fronts_bbox_time.call_args
    assert kwargs["cursor"] == "frontcursor"


def test_fronts_offset_param_forwarded_to_repo(monkeypatch):
    """Deprecated offset query param on /fires/fronts must be forwarded to repo."""
    fake_repo = _make_repo(
        async_list_fire_fronts_bbox_time={"data": [], "next_cursor": None, "has_more": False, "limit": 800}
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    client.get("/fires/fronts", params={**_BASE_PARAMS, "offset": 100})

    _, kwargs = fake_repo.async_list_fire_fronts_bbox_time.call_args
    assert kwargs["offset"] == 100


def test_fronts_response_has_more_true(monkeypatch):
    """When repo signals has_more=True, next_cursor must appear in fronts response."""
    fake_repo = _make_repo(
        async_list_fire_fronts_bbox_time={
            "data": [{"front_id": "front_z"}],
            "next_cursor": "ZnJvbnRjdXJzb3I=",
            "has_more": True,
            "limit": 800,
        }
    )
    monkeypatch.setitem(app.dependency_overrides, get_fire_repo, lambda: fake_repo)

    response = client.get("/fires/fronts", params=_BASE_PARAMS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_more"] is True
    assert payload["next_cursor"] == "ZnJvbnRjdXJzb3I="
