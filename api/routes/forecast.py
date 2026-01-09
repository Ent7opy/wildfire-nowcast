"""FastAPI routes for spread forecasts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import quote_plus

from fastapi import APIRouter
from pydantic import BaseModel

from api.config import settings
from api.forecast import repo

forecast_router = APIRouter(prefix="/forecast", tags=["forecast"])


class GenerateForecastRequest(BaseModel):
    """Request body for generating a forecast on-the-fly."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    region_name: str | None = None
    forecast_reference_time: str | None = None  # ISO format string
    horizons_hours: list[int] | None = None


@forecast_router.get("")
async def get_forecast(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    region_name: str | None = None,
):
    """Get the latest spread forecast for an AOI.

    Returns run metadata, raster asset pointers (with TiTiler URLs),
    and vector contours as GeoJSON.
    
    If region_name is not provided, this will trigger a location-based forecast
    using the bbox (terrain will be empty/optional).
    """
    bbox = (min_lon, min_lat, max_lon, max_lat)
    
    # If region_name is provided, try to get existing forecast run
    # Otherwise, we'll generate on-the-fly (not yet implemented in repo)
    if region_name:
        run = repo.get_latest_forecast_run(region_name, bbox)
    else:
        # Location-based forecasting - no pre-computed runs
        run = None

    if not run:
        return {"run": None}

    run_id = run["id"]
    rasters = repo.list_rasters_for_run(run_id)
    contours = repo.list_contours_for_run(run_id)

    # Enrich rasters with TileJSON URLs for TiTiler
    for r in rasters:
        # Map local storage path to TiTiler-internal path
        # e.g. "data/forecasts/run_1/spread_h024_cog.tif" -> "/data/forecasts/run_1/spread_h024_cog.tif"
        storage_path = str(r["storage_path"])
        titiler_path = storage_path.replace(
            settings.data_dir_local_prefix, settings.data_dir_titiler_mount
        )

        # Build TileJSON URL. TiTiler COG endpoint takes a 'url' query parameter.
        # When running in Docker, this 'url' can be a path to a file mounted inside the TiTiler container.
        encoded_path = quote_plus(titiler_path)
        r["tilejson_url"] = (
            f"{settings.titiler_public_base_url}/cog/WebMercatorQuad/tilejson.json?url={encoded_path}"
        )

    # Build GeoJSON FeatureCollection for contours
    features = []
    for c in contours:
        features.append(
            {
                "type": "Feature",
                "geometry": json.loads(c["geom_geojson"]),
                "properties": {
                    "horizon_hours": c["horizon_hours"],
                    "threshold": c["threshold"],
                },
            }
        )

    # Convert run bbox to dict if it exists
    if run.get("bbox_geojson"):
        run["bbox"] = json.loads(run.pop("bbox_geojson"))

    return {
        "run": run,
        "rasters": rasters,
        "contours": {"type": "FeatureCollection", "features": features},
    }


@forecast_router.post("/generate")
def generate_forecast_endpoint(request: GenerateForecastRequest):
    """Generate a spread forecast on-the-fly for a given bbox.
    
    This endpoint generates forecasts dynamically without requiring pre-computed runs.
    For location-based forecasting (no region_name), terrain will be empty/optional.
    """
    from ml.spread.service import SpreadForecastRequest, run_spread_forecast
    
    bbox = (request.min_lon, request.min_lat, request.max_lon, request.max_lat)
    
    # Parse horizons
    if request.horizons_hours:
        horizons = request.horizons_hours
    else:
        horizons = [24, 48, 72]
    
    # Parse forecast_reference_time
    if request.forecast_reference_time:
        forecast_reference_time = datetime.fromisoformat(request.forecast_reference_time.replace("Z", "+00:00"))
        if forecast_reference_time.tzinfo is None:
            forecast_reference_time = forecast_reference_time.replace(tzinfo=timezone.utc)
    else:
        forecast_reference_time = datetime.now(timezone.utc)
    
    # Create request
    spread_request = SpreadForecastRequest(
        region_name=request.region_name,
        bbox=bbox,
        forecast_reference_time=forecast_reference_time,
        horizons_hours=horizons,
    )
    
    # Generate forecast
    forecast = run_spread_forecast(spread_request)
    
    # Convert to response format (similar to get_forecast but without persistence)
    # For now, return a simplified response with the forecast data
    # In the future, we could persist this and return run metadata
    
    # Convert probabilities to a format suitable for the UI
    # For now, return metadata - full implementation would create rasters/contours
    # The probabilities array is too large to return directly; would need rasterization
    return {
        "run": {
            "id": None,  # No persisted run
            "model_name": "heuristic_v0",
            "forecast_reference_time": forecast_reference_time.isoformat(),
            "region_name": request.region_name,
            "status": "completed",
        },
        "forecast": {
            "bbox": bbox,
            "horizons_hours": list(forecast.horizons_hours),
            "probabilities_shape": list(forecast.probabilities.shape),
            "probabilities_min": float(forecast.probabilities.min().values),
            "probabilities_max": float(forecast.probabilities.max().values),
            "probabilities_mean": float(forecast.probabilities.mean().values),
        },
        "rasters": [],  # Would need to generate rasters for full implementation
        "contours": {"type": "FeatureCollection", "features": []},  # Would need to generate contours
        "note": "Forecast generated successfully. Raster/contour generation not yet implemented for on-the-fly forecasts.",
    }

