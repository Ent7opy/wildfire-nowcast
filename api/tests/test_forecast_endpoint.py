import json
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.forecast import forecast_router

# Create a test app
app = FastAPI()
app.include_router(forecast_router)
client = TestClient(app)


def test_get_forecast_not_found():
    with patch("api.forecast.repo.get_latest_forecast_run", return_value=None):
        response = client.get(
            "/forecast",
            params={
                "region_name": "balkans",
                "min_lon": 0,
                "min_lat": 0,
                "max_lon": 1,
                "max_lat": 1,
            },
        )
        assert response.status_code == 200
        assert response.json() == {"run": None}


def test_get_forecast_success():
    mock_run = {
        "id": 101,
        "region_name": "balkans",
        "status": "completed",
        "model_name": "TestModel",
        "model_version": "v1",
        "forecast_reference_time": "2025-01-01T00:00:00+00:00",
        "metadata": {},
        "bbox_geojson": json.dumps({
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        }),
    }

    mock_rasters = [
        {
            "horizon_hours": 24,
            "file_format": "COG",
            "storage_path": "data/forecasts/balkans/run_101/spread_h024_cog.tif",
        }
    ]

    mock_contours = [
        {
            "horizon_hours": 24,
            "threshold": 0.5,
            "geom_geojson": json.dumps({
                "type": "MultiPolygon",
                "coordinates": []
            }),
        }
    ]

    with patch("api.forecast.repo.get_latest_forecast_run", return_value=mock_run), \
         patch("api.forecast.repo.list_rasters_for_run", return_value=mock_rasters), \
         patch("api.forecast.repo.list_contours_for_run", return_value=mock_contours):
        
        response = client.get(
            "/forecast",
            params={
                "region_name": "balkans",
                "min_lon": 0.2,
                "min_lat": 0.2,
                "max_lon": 0.8,
                "max_lat": 0.8,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check run details
        assert data["run"]["id"] == 101
        assert "bbox" in data["run"]
        assert data["run"]["bbox"]["type"] == "Polygon"
        
        # Check rasters
        assert len(data["rasters"]) == 1
        raster = data["rasters"][0]
        # Check TiTiler URL enrichment
        assert "tilejson_url" in raster
        # URL encoding might affect slashes, so let's just check the presence of the path
        assert "tilejson.json" in raster["tilejson_url"]
        
        # Check contours
        contours = data["contours"]
        assert contours["type"] == "FeatureCollection"
        assert len(contours["features"]) == 1
        feat = contours["features"][0]
        assert feat["properties"]["horizon_hours"] == 24
        assert feat["properties"]["threshold"] == 0.5


def test_generate_forecast_persists_run():
    """Test that POST /forecast/generate creates a run record and persists contours."""
    from unittest.mock import MagicMock
    import numpy as np
    import xarray as xr
    
    mock_forecast = MagicMock()
    mock_forecast.probabilities = xr.DataArray(
        np.random.rand(3, 10, 10),
        dims=["time", "lat", "lon"],
        coords={
            "time": ["2025-01-01T12:00:00", "2025-01-01T18:00:00", "2025-01-02T00:00:00"],
            "lat": np.linspace(40.0, 41.0, 10),
            "lon": np.linspace(20.0, 21.0, 10),
            "lead_time_hours": ("time", [24, 30, 36]),
        },
    )
    mock_forecast.horizons_hours = [24, 30, 36]
    mock_forecast.forecast_reference_time = "2025-01-01T00:00:00+00:00"
    
    with patch("api.routes.forecast.create_spread_forecast_run", return_value=42), \
         patch("api.routes.forecast.run_spread_forecast", return_value=mock_forecast), \
         patch("api.routes.forecast.get_region_grid_spec"), \
         patch("api.routes.forecast.get_grid_window_for_bbox"), \
         patch("api.routes.forecast.save_forecast_rasters", return_value=[]), \
         patch("api.routes.forecast.insert_spread_forecast_rasters"), \
         patch("api.routes.forecast.build_contour_records", return_value=[]), \
         patch("api.routes.forecast.insert_spread_forecast_contours"), \
         patch("api.routes.forecast.finalize_spread_forecast_run") as mock_finalize:
        
        response = client.post(
            "/forecast/generate",
            json={
                "min_lon": 20.0,
                "min_lat": 40.0,
                "max_lon": 21.0,
                "max_lat": 41.0,
                "region_name": "balkans",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that run.id is non-null
        assert data["run"]["id"] == 42
        assert data["run"]["status"] == "completed"
        
        # Verify finalize was called with completed status
        mock_finalize.assert_called_once()
        args = mock_finalize.call_args
        assert args[0][0] == 42
        assert args[1]["status"] == "completed"


def test_create_jit_forecast_valid_bbox():
    """Test POST /forecast/jit with valid bbox creates job and returns job_id with status='queued'."""
    from uuid import uuid4
    from unittest.mock import MagicMock
    
    mock_job_id = uuid4()
    mock_job = {"id": mock_job_id, "status": "queued", "created_at": "2025-01-19T00:00:00"}
    
    with patch("api.forecast.repo.create_jit_job", return_value=mock_job), \
         patch("api.forecast.worker.queue.enqueue") as mock_enqueue:
        
        response = client.post(
            "/forecast/jit",
            json={"bbox": [20.0, 40.0, 21.0, 41.0]},
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == str(mock_job_id)
        assert data["status"] == "queued"
        mock_enqueue.assert_called_once()


def test_create_jit_forecast_invalid_bbox_length():
    """Test POST /forecast/jit with invalid bbox (wrong length) returns 400 error."""
    response = client.post(
        "/forecast/jit",
        json={"bbox": [20.0, 40.0, 21.0]},
    )
    
    assert response.status_code == 400
    assert "bbox must have exactly 4 elements" in response.json()["detail"]


def test_create_jit_forecast_enqueue_failure():
    """Test POST /forecast/jit updates job status to 'failed' and returns 500 error on enqueue failure."""
    from uuid import uuid4
    
    mock_job_id = uuid4()
    mock_job = {"id": mock_job_id, "status": "queued", "created_at": "2025-01-19T00:00:00"}
    
    with patch("api.forecast.repo.create_jit_job", return_value=mock_job), \
         patch("api.forecast.worker.queue.enqueue", side_effect=Exception("Queue unavailable")), \
         patch("api.forecast.repo.update_jit_job_status") as mock_update_status:
        
        response = client.post(
            "/forecast/jit",
            json={"bbox": [20.0, 40.0, 21.0, 41.0]},
        )
        
        assert response.status_code == 500
        assert "Failed to enqueue JIT forecast" in response.json()["detail"]
        mock_update_status.assert_called_once_with(mock_job_id, "failed", error="Queue unavailable")


def test_get_jit_status_not_found():
    """Test GET /forecast/jit/{job_id} returns 404 when job does not exist."""
    from uuid import uuid4
    
    non_existent_job_id = uuid4()
    
    with patch("api.forecast.repo.get_jit_job", return_value=None):
        response = client.get(f"/forecast/jit/{non_existent_job_id}")
        
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]


def test_get_jit_status_pending():
    """Test GET /forecast/jit/{job_id} returns pending status with progress message."""
    from uuid import uuid4
    from datetime import datetime
    
    mock_job_id = uuid4()
    mock_job = {
        "id": mock_job_id,
        "status": "pending",
        "created_at": datetime(2025, 1, 19, 0, 0, 0),
        "updated_at": datetime(2025, 1, 19, 0, 0, 1),
    }
    
    with patch("api.forecast.repo.get_jit_job", return_value=mock_job):
        response = client.get(f"/forecast/jit/{mock_job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == str(mock_job_id)
        assert data["status"] == "pending"
        assert data["progress_message"] == "Job is queued and waiting to start..."


def test_get_jit_status_completed_with_result():
    """Test GET /forecast/jit/{job_id} returns completed status with result data."""
    from uuid import uuid4
    from datetime import datetime
    
    mock_job_id = uuid4()
    mock_result = {
        "run_id": 42,
        "forecast_url": "http://example.com/forecast/42"
    }
    mock_job = {
        "id": mock_job_id,
        "status": "completed",
        "result": mock_result,
        "created_at": datetime(2025, 1, 19, 0, 0, 0),
        "updated_at": datetime(2025, 1, 19, 0, 5, 0),
    }
    
    with patch("api.forecast.repo.get_jit_job", return_value=mock_job):
        response = client.get(f"/forecast/jit/{mock_job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == str(mock_job_id)
        assert data["status"] == "completed"
        assert data["progress_message"] == "Forecast complete!"
        assert data["result"] == mock_result


def test_get_jit_status_failed_with_error():
    """Test GET /forecast/jit/{job_id} returns failed status with error details."""
    from uuid import uuid4
    from datetime import datetime
    
    mock_job_id = uuid4()
    mock_error = "Weather data unavailable for requested region"
    mock_job = {
        "id": mock_job_id,
        "status": "failed",
        "error": mock_error,
        "created_at": datetime(2025, 1, 19, 0, 0, 0),
        "updated_at": datetime(2025, 1, 19, 0, 2, 30),
    }
    
    with patch("api.forecast.repo.get_jit_job", return_value=mock_job):
        response = client.get(f"/forecast/jit/{mock_job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == str(mock_job_id)
        assert data["status"] == "failed"
        assert data["progress_message"] == "Job failed"
        assert data["error"] == mock_error


def test_get_jit_status_all_intermediate_statuses():
    """Test GET /forecast/jit/{job_id} returns correct progress messages for all intermediate statuses."""
    from uuid import uuid4
    from datetime import datetime
    
    mock_job_id = uuid4()
    
    statuses_and_messages = [
        ("ingesting_terrain", "Downloading terrain data..."),
        ("ingesting_weather", "Fetching weather data..."),
        ("running_forecast", "Generating spread forecast..."),
    ]
    
    for status, expected_message in statuses_and_messages:
        mock_job = {
            "id": mock_job_id,
            "status": status,
            "created_at": datetime(2025, 1, 19, 0, 0, 0),
            "updated_at": datetime(2025, 1, 19, 0, 1, 0),
        }
        
        with patch("api.forecast.repo.get_jit_job", return_value=mock_job):
            response = client.get(f"/forecast/jit/{mock_job_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == status
            assert data["progress_message"] == expected_message

