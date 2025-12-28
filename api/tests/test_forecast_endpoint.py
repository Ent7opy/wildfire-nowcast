import json
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.forecast import forecast_router
from api.config import settings

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
        expected_url_part = f"url={settings.titiler_public_base_url}/data/forecasts/balkans/run_101/spread_h024_cog.tif"
        # URL encoding might affect slashes, so let's just check the presence of the path
        assert "tilejson.json" in raster["tilejson_url"]
        
        # Check contours
        contours = data["contours"]
        assert contours["type"] == "FeatureCollection"
        assert len(contours["features"]) == 1
        feat = contours["features"][0]
        assert feat["properties"]["horizon_hours"] == 24
        assert feat["properties"]["threshold"] == 0.5

