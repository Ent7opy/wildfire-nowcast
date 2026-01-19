"""Integration test for JIT forecast pipeline end-to-end.

This test validates the full JIT pipeline flow from API request to completion,
including status polling and result validation. Uses a small bbox and mocks
expensive operations (real terrain/weather downloads) to keep runtime manageable.
"""
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.mark.integration
@pytest.mark.skip(reason="Integration test requires ingest/ml dependencies not installed in api environment")
def test_jit_pipeline_end_to_end():
    """Test full JIT pipeline: POST /forecast/jit -> poll status -> validate results.
    
    This test exercises the complete JIT forecast workflow:
    1. POST to /forecast/jit with a small bbox
    2. Poll /forecast/jit/{job_id} until completed
    3. Assert forecast results exist and contain valid rasters + contours
    
    The test uses a small bbox (10km x 10km) and mocks expensive operations
    (terrain/weather downloads, forecast computation) to keep runtime under 2 minutes.
    """
    # Small test bbox (10km x 10km region in Balkans)
    test_bbox = [20.0, 40.0, 20.1, 40.1]
    
    # Create mock terrain ingestion result
    mock_terrain_id = 1
    
    def mock_ingest_terrain(bbox, output_dir):
        """Mock terrain ingestion - returns quickly without downloading."""
        return mock_terrain_id
    
    # Create mock weather ingestion result
    mock_weather_run_id = 42
    
    def mock_ingest_weather(bbox, forecast_time, output_dir, horizon_hours):
        """Mock weather ingestion - returns quickly without downloading."""
        return mock_weather_run_id
    
    # Create mock forecast result
    mock_forecast = MagicMock()
    mock_forecast.probabilities = xr.DataArray(
        np.random.rand(3, 10, 10),
        dims=["time", "lat", "lon"],
        coords={
            "time": [
                datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 21, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 22, 0, 0, 0, tzinfo=timezone.utc),
            ],
            "lat": np.linspace(40.0, 40.1, 10),
            "lon": np.linspace(20.0, 20.1, 10),
            "lead_time_hours": ("time", [24, 48, 72]),
        },
    )
    mock_forecast.probabilities.attrs = {
        "weather_bias_corrected": False,
        "calibration_applied": False,
    }
    mock_forecast.horizons_hours = [24, 48, 72]
    mock_forecast.forecast_reference_time = datetime(2026, 1, 19, 0, 0, 0, tzinfo=timezone.utc)
    
    # Mock forecast run creation
    mock_run_id = 123
    
    # Mock raster records
    mock_raster_records = [
        {
            "horizon_hours": h,
            "file_format": "COG",
            "storage_path": f"data/forecasts/location-based/run_{mock_run_id}/spread_h{h:03d}_cog.tif",
        }
        for h in [24, 48, 72]
    ]
    
    # Mock contour records
    mock_contour_records = [
        {
            "horizon_hours": h,
            "threshold": t,
            "geom_geojson": '{"type": "MultiPolygon", "coordinates": []}',
        }
        for h in [24, 48, 72]
        for t in [0.3, 0.5, 0.7]
    ]
    
    with patch("ingest.dem_preprocess.ingest_terrain_for_bbox", side_effect=mock_ingest_terrain), \
         patch("ingest.weather_ingest.ingest_weather_for_bbox", side_effect=mock_ingest_weather), \
         patch("ml.spread.service.run_spread_forecast", return_value=mock_forecast), \
         patch("ingest.spread_repository.create_spread_forecast_run", return_value=mock_run_id), \
         patch("ingest.spread_forecast.save_forecast_rasters", return_value=mock_raster_records), \
         patch("ingest.spread_forecast.build_contour_records", return_value=mock_contour_records), \
         patch("ingest.spread_repository.insert_spread_forecast_rasters"), \
         patch("ingest.spread_repository.insert_spread_forecast_contours"), \
         patch("ingest.spread_repository.finalize_spread_forecast_run"), \
         patch("api.forecast.worker.queue") as mock_queue:
        
        # Configure mock queue to run task synchronously instead of enqueuing
        def run_task_sync(task_func, *args, **kwargs):
            """Execute task synchronously instead of enqueuing."""
            task_func(*args)
            return MagicMock()
        
        mock_queue.enqueue.side_effect = run_task_sync
        
        # Step 1: POST to /forecast/jit to create job
        response = client.post(
            "/forecast/jit",
            json={
                "bbox": test_bbox,
                "horizons_hours": [24, 48, 72],
            },
        )
        
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        
        job_id = data["job_id"]
        
        # Step 2: Poll status endpoint until completed
        max_polls = 30  # Maximum 30 polls (30 seconds at 1 poll/second)
        poll_interval = 1.0  # 1 second between polls
        status = None
        
        for i in range(max_polls):
            response = client.get(f"/forecast/jit/{job_id}")
            assert response.status_code == 200
            
            status_data = response.json()
            status = status_data["status"]
            
            # Validate response structure
            assert "job_id" in status_data
            assert "status" in status_data
            assert "progress_message" in status_data
            assert "created_at" in status_data
            assert "updated_at" in status_data
            
            # Check if job is in terminal state
            if status in ("completed", "failed"):
                break
            
            # Wait before next poll
            if i < max_polls - 1:
                time.sleep(poll_interval)
        
        # Step 3: Assert job completed successfully
        assert status == "completed", f"Job did not complete. Final status: {status}"
        
        # Step 4: Validate result structure
        assert "result" in status_data, "Completed job should have result field"
        result = status_data["result"]
        
        # Validate result contains expected fields
        assert "terrain_id" in result
        assert result["terrain_id"] == mock_terrain_id
        
        assert "weather_run_id" in result
        assert result["weather_run_id"] == mock_weather_run_id
        
        assert "forecast_run_id" in result
        assert result["forecast_run_id"] == mock_run_id
        
        assert "tilejson_urls" in result
        assert isinstance(result["tilejson_urls"], list)
        assert len(result["tilejson_urls"]) == 3  # One per horizon (24, 48, 72)
        
        # Validate TileJSON URLs are properly formatted
        for url in result["tilejson_urls"]:
            assert "tilejson.json" in url
            assert "/cog/" in url
        
        # Step 5: Verify no error field exists
        assert "error" not in status_data or status_data["error"] is None


@pytest.mark.integration
@pytest.mark.skip(reason="Integration test requires ingest/ml dependencies not installed in api environment")
def test_jit_pipeline_handles_errors():
    """Test JIT pipeline properly handles and reports errors.
    
    Validates that when an error occurs during processing,
    the job status transitions to 'failed' and error details are captured.
    """
    test_bbox = [20.0, 40.0, 20.1, 40.1]
    
    # Mock terrain ingestion to raise an error
    def mock_ingest_terrain_error(bbox, output_dir):
        raise RuntimeError("Failed to download DEM tiles")
    
    with patch("ingest.dem_preprocess.ingest_terrain_for_bbox", side_effect=mock_ingest_terrain_error), \
         patch("api.forecast.worker.queue") as mock_queue:
        
        # Configure mock queue to run task synchronously
        def run_task_sync(task_func, *args, **kwargs):
            task_func(*args)
            return MagicMock()
        
        mock_queue.enqueue.side_effect = run_task_sync
        
        # Create job
        response = client.post(
            "/forecast/jit",
            json={"bbox": test_bbox},
        )
        
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        
        # Poll for completion
        max_polls = 10
        status = None
        
        for _ in range(max_polls):
            response = client.get(f"/forecast/jit/{job_id}")
            assert response.status_code == 200
            
            status_data = response.json()
            status = status_data["status"]
            
            if status in ("completed", "failed"):
                break
            
            time.sleep(0.5)
        
        # Verify job failed
        assert status == "failed", "Job should have failed due to terrain error"
        
        # Verify error details are captured
        assert "error" in status_data
        assert status_data["error"] is not None
        assert "RuntimeError" in status_data["error"]
        assert "Failed to download DEM tiles" in status_data["error"]


@pytest.mark.integration
@pytest.mark.skip(reason="Integration test requires ingest/ml dependencies not installed in api environment")
def test_jit_pipeline_invalid_bbox():
    """Test JIT pipeline rejects invalid bbox format."""
    # Invalid bbox (only 3 elements instead of 4)
    response = client.post(
        "/forecast/jit",
        json={"bbox": [20.0, 40.0, 20.1]},
    )
    
    assert response.status_code == 400
    assert "bbox must have exactly 4 elements" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.skip(reason="Integration test requires ingest/ml dependencies not installed in api environment")
def test_jit_pipeline_status_not_found():
    """Test status endpoint returns 404 for non-existent job."""
    non_existent_job_id = uuid4()
    
    with patch("api.forecast.repo.get_jit_job", return_value=None):
        response = client.get(f"/forecast/jit/{non_existent_job_id}")
        
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.skip(reason="Integration test requires ingest/ml dependencies not installed in api environment")
def test_jit_pipeline_caching_behavior():
    """Test caching behavior for terrain and weather ingestion.
    
    This test validates that:
    1. First request triggers terrain and weather ingestion
    2. Second request with same bbox reuses cached terrain and weather (cache hit)
    3. Weather cache expires after 6 hours
    4. Terrain cache persists indefinitely
    
    Uses real DB queries (find_cached_terrain/find_cached_weather) to validate
    caching logic, but mocks expensive operations (actual downloads).
    """
    test_bbox = [20.0, 40.0, 20.1, 40.1]
    mock_terrain_id = 1
    mock_forecast = MagicMock()
    mock_forecast.probabilities = xr.DataArray(
        np.random.rand(3, 10, 10),
        dims=["time", "lat", "lon"],
        coords={
            "time": [
                datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 21, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 22, 0, 0, 0, tzinfo=timezone.utc),
            ],
            "lat": np.linspace(40.0, 40.1, 10),
            "lon": np.linspace(20.0, 20.1, 10),
            "lead_time_hours": ("time", [24, 48, 72]),
        },
    )
    mock_forecast.probabilities.attrs = {
        "weather_bias_corrected": False,
        "calibration_applied": False,
    }
    mock_forecast.horizons_hours = [24, 48, 72]
    mock_forecast.forecast_reference_time = datetime(2026, 1, 19, 0, 0, 0, tzinfo=timezone.utc)
    mock_run_id = 123
    mock_raster_records = [
        {
            "horizon_hours": h,
            "file_format": "COG",
            "storage_path": f"data/forecasts/location-based/run_{mock_run_id}/spread_h{h:03d}_cog.tif",
        }
        for h in [24, 48, 72]
    ]
    mock_contour_records = [
        {
            "horizon_hours": h,
            "threshold": t,
            "geom_geojson": '{"type": "MultiPolygon", "coordinates": []}',
        }
        for h in [24, 48, 72]
        for t in [0.3, 0.5, 0.7]
    ]
    
    # Track ingestion call counts
    terrain_calls = []
    weather_calls = []
    
    def mock_ingest_terrain(bbox, output_dir):
        terrain_calls.append(bbox)
        # Simulate DB insert for terrain_features_metadata
        from sqlalchemy import text
        from api.db import get_engine
        with get_engine().begin() as conn:
            conn.execute(text("""
                INSERT INTO terrain_features_metadata
                (id, region_name, source_dem_metadata_id, slope_path, aspect_path,
                 crs_epsg, cell_size_deg, origin_lat, origin_lon, grid_n_lat, grid_n_lon,
                 bbox, slope_units, aspect_units)
                VALUES (
                    :id, 'test-region', 1, 'slope.tif', 'aspect.tif',
                    4326, 0.1, :min_lat, :min_lon, 10, 10,
                    ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326),
                    'degrees', 'degrees'
                )
                ON CONFLICT DO NOTHING
            """), {
                "id": mock_terrain_id,
                "min_lon": bbox[0],
                "min_lat": bbox[1],
                "max_lon": bbox[2],
                "max_lat": bbox[3],
            })
        return mock_terrain_id
    
    def mock_ingest_weather(bbox, forecast_time, output_dir, horizon_hours):
        weather_calls.append((bbox, forecast_time, horizon_hours))
        # Simulate DB insert for weather_runs
        from sqlalchemy import text
        from api.db import get_engine
        with get_engine().begin() as conn:
            result = conn.execute(text("""
                INSERT INTO weather_runs
                (model, run_time, horizon_hours, step_hours,
                 bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat,
                 file_format, storage_path, status)
                VALUES (
                    'GFS', :run_time, :horizon_hours, 3,
                    :min_lon, :min_lat, :max_lon, :max_lat,
                    'netcdf', 'weather.nc', 'completed'
                )
                RETURNING id
            """), {
                "run_time": forecast_time,
                "horizon_hours": horizon_hours,
                "min_lon": bbox[0],
                "min_lat": bbox[1],
                "max_lon": bbox[2],
                "max_lat": bbox[3],
            })
            return result.fetchone()[0]
    
    with patch("ingest.dem_preprocess.ingest_terrain_for_bbox", side_effect=mock_ingest_terrain), \
         patch("ingest.weather_ingest.ingest_weather_for_bbox", side_effect=mock_ingest_weather), \
         patch("ml.spread.service.run_spread_forecast", return_value=mock_forecast), \
         patch("ingest.spread_repository.create_spread_forecast_run", return_value=mock_run_id), \
         patch("ingest.spread_forecast.save_forecast_rasters", return_value=mock_raster_records), \
         patch("ingest.spread_forecast.build_contour_records", return_value=mock_contour_records), \
         patch("ingest.spread_repository.insert_spread_forecast_rasters"), \
         patch("ingest.spread_repository.insert_spread_forecast_contours"), \
         patch("ingest.spread_repository.finalize_spread_forecast_run"), \
         patch("api.forecast.worker.queue") as mock_queue:
        
        def run_task_sync(task_func, *args, **kwargs):
            task_func(*args)
            return MagicMock()
        
        mock_queue.enqueue.side_effect = run_task_sync
        
        # Test 1: First request should trigger both terrain and weather ingestion
        response1 = client.post(
            "/forecast/jit",
            json={"bbox": test_bbox, "horizons_hours": [24, 48, 72]},
        )
        assert response1.status_code == 202
        job_id_1 = response1.json()["job_id"]
        
        # Poll first job to completion
        for _ in range(10):
            response = client.get(f"/forecast/jit/{job_id_1}")
            if response.json()["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        
        assert response.json()["status"] == "completed"
        assert len(terrain_calls) == 1, "First request should ingest terrain"
        assert len(weather_calls) == 1, "First request should ingest weather"
        
        # Test 2: Second request with same bbox should reuse cached data
        response2 = client.post(
            "/forecast/jit",
            json={"bbox": test_bbox, "horizons_hours": [24, 48, 72]},
        )
        assert response2.status_code == 202
        job_id_2 = response2.json()["job_id"]
        
        # Poll second job to completion
        for _ in range(10):
            response = client.get(f"/forecast/jit/{job_id_2}")
            if response.json()["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        
        assert response.json()["status"] == "completed"
        assert len(terrain_calls) == 1, "Second request should reuse cached terrain (no new ingestion)"
        assert len(weather_calls) == 1, "Second request should reuse cached weather (no new ingestion)"
        
        # Test 3: Weather cache expires after 6 hours
        # Simulate 7 hours passing by updating created_at timestamp
        from sqlalchemy import text
        from api.db import get_engine
        with get_engine().begin() as conn:
            conn.execute(text("""
                UPDATE weather_runs
                SET created_at = created_at - INTERVAL '7 hours'
                WHERE status = 'completed'
            """))
        
        response3 = client.post(
            "/forecast/jit",
            json={"bbox": test_bbox, "horizons_hours": [24, 48, 72]},
        )
        assert response3.status_code == 202
        job_id_3 = response3.json()["job_id"]
        
        # Poll third job to completion
        for _ in range(10):
            response = client.get(f"/forecast/jit/{job_id_3}")
            if response.json()["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        
        assert response.json()["status"] == "completed"
        assert len(terrain_calls) == 1, "Third request should still reuse cached terrain (persists indefinitely)"
        assert len(weather_calls) == 2, "Third request should re-ingest weather (cache expired)"
        
        # Test 4: Verify terrain cache persists (already validated above)
        # The fact that terrain_calls stayed at 1 after all three requests confirms
        # terrain caching works indefinitely
