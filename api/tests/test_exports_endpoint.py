from unittest.mock import MagicMock
from uuid import uuid4
from datetime import datetime, timezone

from fastapi.testclient import TestClient

import api.routes.exports as exports_routes
from api.main import app

client = TestClient(app)

def test_export_aoi_geojson(monkeypatch):
    """Test exporting AOI as GeoJSON."""
    aoi_id = uuid4()
    mock_aoi = {
        "id": aoi_id,
        "name": "Export AOI",
        "description": "Desc",
        "geometry": {"type": "Polygon", "coordinates": [[[0,0], [1,1], [1,0], [0,0]]]},
        "area_km2": 10.0,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    
    mock_get = MagicMock(return_value=mock_aoi)
    monkeypatch.setattr(exports_routes.aois_repo, "get_aoi", mock_get)

    response = client.get(f"/aois/{aoi_id}/export?format=geojson")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    
    data = response.json()
    assert data["type"] == "Feature"
    assert data["geometry"]["type"] == "Polygon"
    assert data["properties"]["name"] == "Export AOI"


def test_export_fires_csv(monkeypatch):
    """Test exporting fires as CSV."""
    mock_detections = [
        {"id": 1, "lat": 10.0, "lon": 20.0, "acq_time": datetime(2026, 1, 1, tzinfo=timezone.utc), "confidence": 100, "frp": 10, "sensor": "V", "source": "f"},
        {"id": 2, "lat": 11.0, "lon": 21.0, "acq_time": datetime(2026, 1, 1, tzinfo=timezone.utc), "confidence": 90, "frp": 20, "sensor": "V", "source": "f"}
    ]
    
    mock_list = MagicMock(return_value=mock_detections)
    monkeypatch.setattr(exports_routes.fires_repo, "list_fire_detections_bbox_time", mock_list)

    response = client.get(
        "/fires/export",
        params={
            "min_lon": 0, "min_lat": 0, "max_lon": 30, "max_lat": 30,
            "start_time": "2026-01-01T00:00:00Z", "end_time": "2026-01-02T00:00:00Z",
            "format": "csv"
        }
    )
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    content = response.text
    assert "id,lat,lon" in content
    assert "1,10.0,20.0" in content


def test_export_forecast_contours_geojson(monkeypatch):
    """Test exporting forecast contours as GeoJSON."""
    mock_contours = [
        {"horizon_hours": 24, "threshold": 0.5, "geom_geojson": '{"type": "MultiPolygon", "coordinates": []}'}
    ]
    
    mock_list = MagicMock(return_value=mock_contours)
    monkeypatch.setattr(exports_routes.forecast_repo, "list_contours_for_run", mock_list)

    response = client.get("/forecast/123/contours/export?format=geojson")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1
    assert data["features"][0]["properties"]["horizon_hours"] == 24
