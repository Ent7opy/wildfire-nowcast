import json
import os
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import WildfireError, wildfire_error_handler
from api.routes.forecast import forecast_router
from ml.spread.region_key import bbox_region_name

# Create a test app
app = FastAPI()
app.include_router(forecast_router)
app.add_exception_handler(WildfireError, wildfire_error_handler)
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
    import pytest
    from unittest.mock import MagicMock
    import numpy as np
    import xarray as xr
    
    pytest.skip("This test requires ingest/ml dependencies which are not installed in api test environment")
    
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
    
    with patch("ingest.spread_repository.create_spread_forecast_run", return_value=42), \
         patch("ml.spread.service.run_spread_forecast", return_value=mock_forecast), \
         patch("api.fires.service.get_region_grid_spec"), \
         patch("api.core.grid.get_grid_window_for_bbox"), \
         patch("ingest.spread_forecast.save_forecast_rasters", return_value=[]), \
         patch("ingest.spread_repository.insert_spread_forecast_rasters"), \
         patch("ingest.spread_forecast.build_contour_records", return_value=[]), \
         patch("ingest.spread_repository.insert_spread_forecast_contours"), \
         patch("ingest.spread_repository.finalize_spread_forecast_run") as mock_finalize:
        
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


def test_create_jit_forecast_accepts_new_optional_fields():
    """Test POST /forecast/jit accepts model_id + strict/cache flags."""
    from uuid import uuid4
    from api.forecast.model_catalog import get_spread_model_catalog

    mock_job_id = uuid4()
    mock_job = {"id": mock_job_id, "status": "queued", "created_at": "2025-01-19T00:00:00"}
    catalog_json = json.dumps(
        {
            "spread_v1_prod": {
                "model_name": "LearnedSpreadModelV1",
                "model_params": {"model_run_dir": "models/spread_v1/run_123"},
            }
        }
    )

    get_spread_model_catalog.cache_clear()
    with patch.dict(os.environ, {"SPREAD_MODEL_CATALOG_JSON": catalog_json}), \
         patch("api.forecast.repo.create_jit_job", return_value=mock_job) as mock_create, \
         patch("api.forecast.worker.queue.enqueue"):

        response = client.post(
            "/forecast/jit",
            json={
                "bbox": [20.0, 40.0, 21.0, 41.0],
                "model_id": "spread_v1_prod",
                "strict_inputs": True,
                "use_result_cache": False,
            },
        )

        assert response.status_code == 202
        create_args = mock_create.call_args.args
        persisted_request = create_args[1]
        assert persisted_request["model_id"] == "spread_v1_prod"
        assert persisted_request["model_name"] == "LearnedSpreadModelV1"
        assert persisted_request["model_params"] == {"model_run_dir": "models/spread_v1/run_123"}
        assert persisted_request["strict_inputs"] is True
        assert persisted_request["use_result_cache"] is False
    get_spread_model_catalog.cache_clear()


def test_create_jit_forecast_defaults_new_flags():
    """Test POST /forecast/jit defaults strict=false and result-cache=true."""
    from uuid import uuid4

    mock_job_id = uuid4()
    mock_job = {"id": mock_job_id, "status": "queued", "created_at": "2025-01-19T00:00:00"}

    with patch("api.forecast.repo.create_jit_job", return_value=mock_job) as mock_create, \
         patch("api.forecast.worker.queue.enqueue"):
        response = client.post(
            "/forecast/jit",
            json={"bbox": [20.0, 40.0, 21.0, 41.0]},
        )

        assert response.status_code == 202
        persisted_request = mock_create.call_args.args[1]
        assert persisted_request["strict_inputs"] is False
        assert persisted_request["use_result_cache"] is True


def test_create_jit_forecast_invalid_horizons_rejected():
    """Test POST /forecast/jit rejects duplicate/invalid horizons."""
    response = client.post(
        "/forecast/jit",
        json={
            "bbox": [20.0, 40.0, 21.0, 41.0],
            "horizons_hours": [24, 24],
        },
    )
    assert response.status_code == 422
    assert "horizons_hours must not contain duplicates" in response.json()["detail"]


def test_create_jit_forecast_from_front_resolves_bbox_and_enforces_strict_inputs():
    from uuid import uuid4

    front_id = "front_abc_123"
    bbox = (20.01, 40.01, 20.99, 40.99)
    mock_job_id = uuid4()
    mock_job = {"id": mock_job_id, "status": "queued", "created_at": "2025-01-19T00:00:00"}

    with (
        patch(
            "api.routes.forecast.get_fire_front_by_id",
            return_value={
                "front_id": front_id,
                "bbox_min_lon": bbox[0],
                "bbox_min_lat": bbox[1],
                "bbox_max_lon": bbox[2],
                "bbox_max_lat": bbox[3],
            },
        ),
        patch("api.forecast.repo.create_jit_job", return_value=mock_job) as mock_create,
        patch("api.forecast.worker.queue.enqueue"),
    ):
        response = client.post(
            "/forecast/jit/from-front",
            json={"front_id": front_id, "horizons_hours": [24, 48]},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == str(mock_job_id)
    assert body["status"] == "queued"
    assert body["front_id"] == front_id
    assert body["bbox"] == [bbox[0], bbox[1], bbox[2], bbox[3]]

    persisted_request = mock_create.call_args.args[1]
    assert persisted_request["front_id"] == front_id
    assert persisted_request["strict_inputs"] is True
    assert persisted_request["region_name"] == bbox_region_name(bbox)


def test_create_jit_forecast_from_front_not_found():
    with patch("api.routes.forecast.get_fire_front_by_id", return_value=None):
        response = client.post(
            "/forecast/jit/from-front",
            json={"front_id": "missing_front"},
        )

    assert response.status_code == 404
    assert "front_id not found" in response.json()["detail"]


def test_create_jit_forecast_invalid_reference_time_rejected():
    """Test POST /forecast/jit rejects invalid ISO reference time."""
    response = client.post(
        "/forecast/jit",
        json={
            "bbox": [20.0, 40.0, 21.0, 41.0],
            "forecast_reference_time": "2026-13-99T25:00:00Z",
        },
    )
    assert response.status_code == 422
    assert "Invalid ISO 8601 datetime format" in response.json()["detail"]


def test_create_jit_forecast_default_reference_time_is_canonical():
    """Test POST /forecast/jit stores canonical default forecast_reference_time."""
    from uuid import uuid4

    mock_job_id = uuid4()
    mock_job = {"id": mock_job_id, "status": "queued", "created_at": "2025-01-19T00:00:00"}
    canonical_time = datetime(2026, 2, 11, 14, 0, 0, tzinfo=timezone.utc)

    with patch("api.routes.forecast._default_forecast_reference_time", return_value=canonical_time), \
         patch("api.forecast.repo.create_jit_job", return_value=mock_job) as mock_create, \
         patch("api.forecast.worker.queue.enqueue"):
        response = client.post(
            "/forecast/jit",
            json={"bbox": [20.0, 40.0, 21.0, 41.0]},
        )

    assert response.status_code == 202
    persisted_request = mock_create.call_args.args[1]
    assert persisted_request["forecast_reference_time"] == canonical_time.isoformat()


def test_create_jit_forecast_invalid_model_selection():
    """Test POST /forecast/jit fails fast for unknown model."""
    response = client.post(
        "/forecast/jit",
        json={
            "bbox": [20.0, 40.0, 21.0, 41.0],
            "model_name": "NoSuchModel",
        },
    )

    assert response.status_code == 422
    assert "Unsupported model" in response.json()["detail"]


def test_create_jit_forecast_rejects_direct_artifact_paths():
    """Test POST /forecast/jit rejects raw model artifact paths in request."""
    response = client.post(
        "/forecast/jit",
        json={
            "bbox": [20.0, 40.0, 21.0, 41.0],
            "model_name": "HeuristicSpreadModelV0",
            "model_params": {"model_run_dir": "/tmp/unsafe"},
        },
    )

    assert response.status_code == 422
    assert "Direct model artifact paths are not allowed" in response.json()["detail"]


def test_create_jit_forecast_unknown_model_id():
    """Test POST /forecast/jit rejects unknown model_id."""
    response = client.post(
        "/forecast/jit",
        json={
            "bbox": [20.0, 40.0, 21.0, 41.0],
            "model_id": "unknown",
        },
    )

    assert response.status_code == 422
    assert "Unsupported model_id" in response.json()["detail"]


def test_create_jit_forecast_rejects_tampered_catalog_signature():
    """Test POST /forecast/jit fails when catalog signature verification fails."""
    catalog_json = json.dumps(
        {
            "spread_v1_prod": {
                "model_name": "LearnedSpreadModelV1",
                "model_params": {"model_run_dir": "models/spread_v1/run_123"},
            }
        }
    )

    with patch.dict(
        os.environ,
        {
            "SPREAD_MODEL_CATALOG_JSON": catalog_json,
            "SPREAD_MODEL_CATALOG_REQUIRE_SIGNATURE": "true",
            "SPREAD_MODEL_CATALOG_SIGNING_KEY": "test-key",
            "SPREAD_MODEL_CATALOG_SIGNATURE": "bad-signature",
        },
    ):
        response = client.post(
            "/forecast/jit",
            json={
                "bbox": [20.0, 40.0, 21.0, 41.0],
                "model_id": "spread_v1_prod",
            },
        )

    assert response.status_code == 422
    assert "signature mismatch" in response.json()["detail"]


def test_create_jit_forecast_invalid_bbox_length():
    """Test POST /forecast/jit with invalid bbox (wrong length) raises InvalidBoundingBoxError."""
    response = client.post(
        "/forecast/jit",
        json={"bbox": [20.0, 40.0, 21.0]},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "InvalidBoundingBoxError"
    assert "bbox must have exactly 4 elements" in body["detail"]


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
        "forecast_run_id": 42,
        "run_id": 42,
        "cache_hit": False,
        "cache_source": None,
        "forecast_url": "http://example.com/forecast/42",
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


def test_generate_forecast_invalid_model_selection_returns_422():
    """Test POST /forecast/generate fails fast for unknown model selection."""
    response = client.post(
        "/forecast/generate",
        json={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 21.0,
            "max_lat": 41.0,
            "region_name": "balkans",
            "model_name": "NoSuchModel",
        },
    )

    assert response.status_code == 422
    assert "Unsupported model" in response.json()["detail"]


def test_generate_forecast_invalid_reference_time_returns_422():
    """Test POST /forecast/generate rejects invalid ISO reference time."""
    response = client.post(
        "/forecast/generate",
        json={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 21.0,
            "max_lat": 41.0,
            "region_name": "balkans",
            "forecast_reference_time": "invalid-datetime",
        },
    )
    assert response.status_code == 422
    assert "Invalid ISO 8601 datetime format" in response.json()["detail"]


def test_generate_forecast_invalid_horizons_returns_422():
    """Test POST /forecast/generate rejects invalid horizons."""
    response = client.post(
        "/forecast/generate",
        json={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 21.0,
            "max_lat": 41.0,
            "region_name": "balkans",
            "horizons_hours": [0, 24],
        },
    )
    assert response.status_code == 422
    assert "horizons_hours must contain only positive integers" in response.json()["detail"]


def test_generate_forecast_returns_cached_result_when_available():
    """Test POST /forecast/generate serves cached run when request key matches."""
    cached_run = {
        "id": 123,
        "model_name": "HeuristicSpreadModelV0",
        "model_version": "v0",
        "forecast_reference_time": datetime(2026, 1, 19, 0, 0, tzinfo=timezone.utc),
        "region_name": "balkans",
        "status": "completed",
        "metadata": {"cache_key": "abc"},
        "bbox_geojson": json.dumps({"type": "Polygon", "coordinates": [[[20, 40], [21, 40], [21, 41], [20, 41], [20, 40]]]}),
    }
    cached_rasters = [
        {
            "horizon_hours": 24,
            "file_format": "COG",
            "storage_path": "data/forecasts/balkans/run_123/spread_h024_cog.tif",
        }
    ]
    cached_contours = [
        {
            "horizon_hours": 24,
            "threshold": 0.5,
            "geom_geojson": json.dumps({"type": "MultiPolygon", "coordinates": []}),
        }
    ]

    with patch("api.forecast.repo.build_forecast_result_cache_key", return_value="abc"), \
         patch("api.routes.forecast.acquire_forecast_result_lock", return_value=object()) as mock_lock, \
         patch("api.routes.forecast.release_forecast_result_lock") as mock_release, \
         patch("api.forecast.repo.find_cached_forecast_run", return_value=cached_run), \
         patch("api.forecast.repo.list_rasters_for_run", return_value=cached_rasters), \
         patch("api.forecast.repo.list_contours_for_run", return_value=cached_contours):
        response = client.post(
            "/forecast/generate",
            json={
                "min_lon": 20.0,
                "min_lat": 40.0,
                "max_lon": 21.0,
                "max_lat": 41.0,
                "region_name": "balkans",
                "use_result_cache": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cache_hit"] is True
    assert payload["cache_source"] == "forecast_result"
    assert payload["run"]["id"] == 123
    mock_lock.assert_called_once_with("abc")
    mock_release.assert_called_once()


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


def test_cache_key_differs_by_model_id():
    """Promoting a new model changes the cache key, causing natural cache misses."""
    from datetime import datetime, timezone
    from api.forecast.repo import build_forecast_result_cache_key

    common = dict(
        bbox=(-120.0, 37.0, -119.0, 38.0),
        forecast_reference_time=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        horizons_hours=[6, 12, 24],
        region_name="test-region",
        model_name="HeuristicSpreadModelV0",
        model_params={},
        strict_inputs=False,
        thresholds=[0.3, 0.5, 0.7],
    )

    key_v1 = build_forecast_result_cache_key(**common, model_id="model-id-v1")
    key_v2 = build_forecast_result_cache_key(**common, model_id="model-id-v2")
    key_none = build_forecast_result_cache_key(**common, model_id=None)

    assert key_v1 != key_v2, "Different model_id values must produce different cache keys"
    assert key_v1 != key_none, "model_id=None must produce a different key than a named model_id"
    assert key_v2 != key_none
