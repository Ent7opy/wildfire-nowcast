from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4
from datetime import datetime, timezone
import tempfile
import os
from pathlib import Path

from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter

import api.routes.exports as exports_routes
from api.main import app

# Mock FastAPILimiter to avoid Redis initialization requirement
mock_redis = AsyncMock()
mock_redis.evalsha = AsyncMock(return_value=0)  # Return 0 to indicate rate limit not exceeded
FastAPILimiter.redis = mock_redis
FastAPILimiter.identifier = AsyncMock(return_value="test_identifier")
FastAPILimiter.http_callback = AsyncMock()
FastAPILimiter.lua_sha = "mock_sha"
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


def test_create_export_job(monkeypatch):
    """Test creating an async export job."""
    job_id = uuid4()
    mock_job = {
        "id": job_id,
        "kind": "fires_csv",
        "status": "queued",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)
    }
    
    mock_create = MagicMock(return_value=mock_job)
    mock_enqueue = MagicMock()
    
    monkeypatch.setattr(exports_routes.jobs_repo, "create_job", mock_create)
    monkeypatch.setattr(exports_routes.queue, "enqueue", mock_enqueue)
    
    response = client.post(
        "/exports",
        json={
            "kind": "fires_csv",
            "request": {
                "min_lon": 0, "min_lat": 0, "max_lon": 30, "max_lat": 30,
                "start_time": "2026-01-01T00:00:00Z", "end_time": "2026-01-02T00:00:00Z"
            }
        }
    )
    
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    mock_create.assert_called_once()
    mock_enqueue.assert_called_once()


def test_export_task_fires_csv(monkeypatch):
    """Test the export_task worker function for fires_csv."""
    from api.exports import worker
    
    job_id = uuid4()
    request = {
        "min_lon": 0, "min_lat": 0, "max_lon": 30, "max_lat": 30,
        "start_time": "2026-01-01T00:00:00Z", "end_time": "2026-01-02T00:00:00Z"
    }
    
    mock_detections = [
        {"id": 1, "lat": 10.0, "lon": 20.0, "acq_time": datetime(2026, 1, 1, tzinfo=timezone.utc), "confidence": 100, "frp": 10, "sensor": "V", "source": "f"},
        {"id": 2, "lat": 11.0, "lon": 21.0, "acq_time": datetime(2026, 1, 1, tzinfo=timezone.utc), "confidence": 90, "frp": 20, "sensor": "V", "source": "f"}
    ]
    
    with patch("api.exports.repo.update_job_status") as mock_update, \
         patch("api.fires.repo.list_fire_detections_bbox_time", return_value=mock_detections):
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_base_dir = Path(tmpdir) / "exports"
            
            with patch("api.exports.worker.settings.exports_dir", export_base_dir):
                worker.export_task(job_id, "fires_csv", request)
        
        assert mock_update.call_count == 2
        mock_update.assert_any_call(job_id, "running")
        
        final_call = mock_update.call_args_list[1]
        assert final_call[0][0] == job_id
        assert final_call[0][1] == "succeeded"
        result = final_call[1]["result"]
        assert "file_path" in result
        assert "download_url" in result
        assert result["row_count"] == 2


def test_download_export_job(monkeypatch):
    """Test downloading a completed export job."""
    job_id = uuid4()
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("id,lat,lon\n1,10.0,20.0\n")
        temp_file = f.name
    
    try:
        mock_job = {
            "id": job_id,
            "kind": "fires_csv",
            "status": "succeeded",
            "result": {
                "file_path": temp_file,
                "download_url": f"/exports/{job_id}/download"
            }
        }
        
        mock_get = MagicMock(return_value=mock_job)
        monkeypatch.setattr(exports_routes.jobs_repo, "get_job", mock_get)
        
        response = client.get(f"/exports/{job_id}/download")
        assert response.status_code == 200
        assert "id,lat,lon" in response.text
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
