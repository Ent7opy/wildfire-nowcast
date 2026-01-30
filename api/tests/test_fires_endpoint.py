from datetime import datetime, timezone
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import api.routes.fires as fires
from api.main import app

client = TestClient(app)


def test_get_detections_endpoint_basic(monkeypatch):
    """Test that the /fires/detections endpoint works and calls the repo helper."""
    mock_detections = [
        {"id": 1, "lat": 42.0, "lon": 21.0, "acq_time": datetime(2025, 1, 1, tzinfo=timezone.utc)}
    ]
    mock_list = MagicMock(return_value={"data": mock_detections, "next_cursor": None, "has_more": False, "limit": 1000})
    # Monkeypatch where it's used
    monkeypatch.setattr(fires, "list_fire_detections_bbox_time", mock_list)

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
    
    # Verify defaults passed to repo
    _, kwargs = mock_list.call_args
    assert kwargs["include_noise"] is False
    assert kwargs["min_confidence"] is None
    assert "denoised_score" not in kwargs["columns"]


def test_get_fires_endpoint_alias(monkeypatch):
    """Test that the /fires endpoint aliases /fires/detections."""
    mock_list = MagicMock(return_value={"data": [], "next_cursor": None, "has_more": False, "limit": 1000})
    monkeypatch.setattr(fires, "list_fire_detections_bbox_time", mock_list)

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
    mock_list = MagicMock(return_value={"data": [], "next_cursor": None, "has_more": False, "limit": 1000})
    monkeypatch.setattr(fires, "list_fire_detections_bbox_time", mock_list)

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

    _, kwargs = mock_list.call_args
    assert kwargs["min_confidence"] == 80.0


def test_get_detections_endpoint_with_denoiser_fields(monkeypatch):
    """Test that include_denoiser_fields adds columns."""
    mock_list = MagicMock(return_value={"data": [], "next_cursor": None, "has_more": False, "limit": 1000})
    monkeypatch.setattr(fires, "list_fire_detections_bbox_time", mock_list)

    client.get(
        "/fires/detections",
        params={
            "min_lon": 20.0, "min_lat": 40.0, "max_lon": 22.0, "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
            "include_denoiser_fields": "true",
        },
    )

    _, kwargs = mock_list.call_args
    assert "denoised_score" in kwargs["columns"]
    assert "is_noise" in kwargs["columns"]


def test_get_detections_endpoint_with_include_noise(monkeypatch):
    """Test that include_noise is passed to repo."""
    mock_list = MagicMock(return_value={"data": [], "next_cursor": None, "has_more": False, "limit": 1000})
    monkeypatch.setattr(fires, "list_fire_detections_bbox_time", mock_list)

    client.get(
        "/fires/detections",
        params={
            "min_lon": 20.0, "min_lat": 40.0, "max_lon": 22.0, "max_lat": 43.0,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-02T00:00:00Z",
            "include_noise": "true",
        },
    )

    _, kwargs = mock_list.call_args
    assert kwargs["include_noise"] is True

