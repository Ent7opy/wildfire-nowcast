"""FastAPI routes for spread forecasts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import quote_plus
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from api.config import settings
from api.forecast import repo
from api.forecast.worker import queue, run_jit_forecast_pipeline

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


class JitForecastRequest(BaseModel):
    """Request body for JIT forecast pipeline (bbox-only, no region_name required)."""
    bbox: list[float]
    forecast_reference_time: str | None = None
    horizons_hours: list[int] | None = None


class JitForecastResponse(BaseModel):
    """Response body for JIT forecast pipeline initiation."""
    job_id: UUID
    status: str


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


@forecast_router.post("/jit", response_model=JitForecastResponse, status_code=status.HTTP_202_ACCEPTED)
def create_jit_forecast(request: JitForecastRequest):
    """Enqueue a JIT forecast pipeline for arbitrary bbox.

    Accepts a bbox without requiring region_name. The pipeline will:
    1. Ingest terrain data for the bbox
    2. Ingest weather data for the bbox
    3. Generate spread forecast
    4. Persist results

    Returns job_id for tracking the job.
    """
    if len(request.bbox) != 4:
        raise HTTPException(
            status_code=400,
            detail="bbox must have exactly 4 elements: [min_lon, min_lat, max_lon, max_lat]"
        )

    bbox = tuple(request.bbox)

    forecast_params = {
        "forecast_reference_time": request.forecast_reference_time,
        "horizons_hours": request.horizons_hours or [24, 48, 72],
    }

    job = repo.create_jit_job(bbox, {"bbox": request.bbox, **forecast_params})
    job_id = job["id"]

    try:
        queue.enqueue(run_jit_forecast_pipeline, job_id, bbox, forecast_params)
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        repo.update_jit_job_status(job_id, "failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue JIT forecast: {str(e)}"
        )


@forecast_router.get("/jit/{job_id}")
def get_jit_forecast_status(job_id: UUID):
    """Get JIT forecast job status.

    Returns current job status with user-friendly progress messages.
    Includes result data on completion and error details on failure.

    Example:
        GET /forecast/jit/550e8400-e29b-41d4-a716-446655440000
    """
    job = repo.get_jit_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    status_messages = {
        "pending": "Job is queued and waiting to start...",
        "ingesting_terrain": "Downloading terrain data...",
        "ingesting_weather": "Fetching weather data...",
        "running_forecast": "Generating spread forecast...",
        "completed": "Forecast complete!",
        "failed": "Job failed"
    }

    response = {
        "job_id": job["id"],
        "status": job["status"],
        "progress_message": status_messages.get(job["status"], "Processing..."),
        "created_at": job["created_at"].isoformat() if job.get("created_at") else None,
        "updated_at": job["updated_at"].isoformat() if job.get("updated_at") else None,
    }

    if job.get("result"):
        response["result"] = job["result"]

    if job.get("error"):
        response["error"] = job["error"]

    return response


@forecast_router.post("/generate")
def generate_forecast_endpoint(request: GenerateForecastRequest):
    """Generate a spread forecast on-the-fly for a given bbox and persist it.

    This endpoint generates forecasts dynamically and persists contours and rasters.
    Requires region_name to determine the grid for persistence.
    """
    from ml.spread.service import SpreadForecastRequest, run_spread_forecast
    from api.fires.service import get_region_grid_spec
    from api.core.grid import get_grid_window_for_bbox
    from ingest.spread_forecast import save_forecast_rasters, build_contour_records
    from ingest.spread_repository import (
        create_spread_forecast_run,
        insert_spread_forecast_rasters,
        insert_spread_forecast_contours,
        finalize_spread_forecast_run,
    )
    from ingest.config import REPO_ROOT
    from pathlib import Path

    if not request.region_name:
        raise HTTPException(
            status_code=400,
            detail="region_name is required for persisting forecasts"
        )

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

    # Create run record
    run_id = create_spread_forecast_run(
        region_name=request.region_name,
        model_name="HeuristicSpreadModelV0",
        model_version="v0",
        forecast_reference_time=forecast_reference_time,
        bbox=bbox,
    )

    try:
        # Create request
        spread_request = SpreadForecastRequest(
            region_name=request.region_name,
            bbox=bbox,
            forecast_reference_time=forecast_reference_time,
            horizons_hours=horizons,
        )

        # Generate forecast
        forecast = run_spread_forecast(spread_request)

        # Get grid and window for persistence
        grid = get_region_grid_spec(request.region_name)
        window = get_grid_window_for_bbox(grid, bbox, clip=True)

        # Persist rasters
        run_dir = REPO_ROOT / "data" / "forecasts" / request.region_name / f"run_{run_id}"
        raster_records = save_forecast_rasters(forecast, grid, window, run_dir, emit_cog=True)
        insert_spread_forecast_rasters(run_id, raster_records)

        # Generate and persist contours
        contour_records = build_contour_records(
            forecast=forecast,
            grid=grid,
            window=window,
            thresholds=[0.3, 0.5, 0.7],
        )
        insert_spread_forecast_contours(run_id, contour_records)

        # Capture operational metadata from forecast output
        extra_meta = {}
        try:
            attrs = dict(getattr(forecast.probabilities, "attrs", {}) or {})
            for k in (
                "weather_bias_corrected",
                "weather_bias_corrector_path",
                "calibration_applied",
                "calibration_source",
            ):
                if k in attrs:
                    extra_meta[k] = attrs.get(k)
        except Exception:
            pass

        # Finalize
        finalize_spread_forecast_run(run_id, status="completed", extra_metadata=extra_meta)

        # Enrich rasters with TileJSON URLs for TiTiler
        for r in raster_records:
            storage_path = str(r["storage_path"])
            titiler_path = storage_path.replace(
                settings.data_dir_local_prefix, settings.data_dir_titiler_mount
            )
            encoded_path = quote_plus(titiler_path)
            r["tilejson_url"] = (
                f"{settings.titiler_public_base_url}/cog/WebMercatorQuad/tilejson.json?url={encoded_path}"
            )

        # Return response similar to GET /forecast
        return {
            "run": {
                "id": run_id,
                "model_name": "HeuristicSpreadModelV0",
                "model_version": "v0",
                "forecast_reference_time": forecast_reference_time.isoformat(),
                "region_name": request.region_name,
                "status": "completed",
            },
            "rasters": raster_records,
            "contours": {"type": "FeatureCollection", "features": [
                {
                    "type": "Feature",
                    "geometry": json.loads(c["geom_geojson"]),
                    "properties": {
                        "horizon_hours": c["horizon_hours"],
                        "threshold": c["threshold"],
                    },
                }
                for c in contour_records
            ]},
        }

    except Exception as e:
        finalize_spread_forecast_run(run_id, status="failed", extra_metadata={"error": str(e)})
        raise

