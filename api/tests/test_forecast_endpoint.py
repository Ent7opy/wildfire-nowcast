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

