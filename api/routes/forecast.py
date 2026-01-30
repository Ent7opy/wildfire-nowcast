"""FastAPI routes for spread forecasts."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
import re
from typing import AsyncGenerator
from urllib.parse import quote_plus
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel

from api.config import settings
from api.fires.repo import validate_bbox
from api.forecast import repo
from api.forecast.worker import queue, run_jit_forecast_pipeline, handle_jit_pipeline_failure

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


@forecast_router.get("", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
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


@forecast_router.post(
    "/jit",
    response_model=JitForecastResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(RateLimiter(times=10, seconds=60))],
)
def create_jit_forecast(request: JitForecastRequest):
    """Enqueue a JIT forecast pipeline for arbitrary bbox.

    Accepts a bbox without requiring region_name. The pipeline will:
    1. Ingest terrain data for the bbox
    2. Ingest weather data for the bbox
    3. Generate spread forecast
    4. Persist results

    Args:
        request: JIT forecast request containing:
            - bbox: [min_lon, min_lat, max_lon, max_lat] in decimal degrees (WGS84)
            - forecast_reference_time (optional): ISO timestamp (defaults to current time)
            - horizons_hours (optional): list of forecast horizons in hours (defaults to [24, 48, 72])

    Returns:
        JitForecastResponse with job_id and status='queued'

    Example:
        ```bash
        curl -X POST http://localhost:8000/forecast/jit \\
          -H "Content-Type: application/json" \\
          -d '{
            "bbox": [20.0, 40.0, 21.0, 41.0],
            "horizons_hours": [24, 48, 72]
          }'
        ```

        Response:
        ```json
        {
          "job_id": "550e8400-e29b-41d4-a716-446655440000",
          "status": "queued"
        }
        ```
    """
    if len(request.bbox) != 4:
        raise HTTPException(
            status_code=400,
            detail="bbox must have exactly 4 elements: [min_lon, min_lat, max_lon, max_lat]"
        )

    bbox = tuple(request.bbox)

    # Validate bbox coordinates
    try:
        validate_bbox(bbox)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    forecast_params = {
        "forecast_reference_time": request.forecast_reference_time,
        "horizons_hours": request.horizons_hours or [24, 48, 72],
    }

    job = repo.create_jit_job(bbox, {"bbox": request.bbox, **forecast_params})
    job_id = job["id"]

    try:
        queue.enqueue(
            run_jit_forecast_pipeline,
            job_id,
            bbox,
            forecast_params,
            on_failure=handle_jit_pipeline_failure
        )
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        repo.update_jit_job_status(job_id, "failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue JIT forecast: {str(e)}"
        )


@forecast_router.get("/jit/{job_id}", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
def get_jit_forecast_status(job_id: UUID):
    """Get JIT forecast job status.

    Returns current job status with user-friendly progress messages.
    Includes result data on completion and error details on failure.

    Args:
        job_id: UUID of the JIT forecast job

    Returns:
        Job status response containing:
        - job_id: UUID of the job
        - status: pending|ingesting_terrain|ingesting_weather|running_forecast|completed|failed
        - progress_message: Human-readable status message
        - created_at: ISO timestamp when job was created
        - updated_at: ISO timestamp of last status update
        - result (if completed): forecast run_id and asset URLs
        - error (if failed): error message

    Example:
        ```bash
        curl http://localhost:8000/forecast/jit/550e8400-e29b-41d4-a716-446655440000
        ```

        Response (in progress):
        ```json
        {
          "job_id": "550e8400-e29b-41d4-a716-446655440000",
          "status": "ingesting_weather",
          "progress_message": "Fetching weather data...",
          "created_at": "2026-01-19T12:00:00Z",
          "updated_at": "2026-01-19T12:00:15Z"
        }
        ```

        Response (completed):
        ```json
        {
          "job_id": "550e8400-e29b-41d4-a716-446655440000",
          "status": "completed",
          "progress_message": "Forecast complete!",
          "created_at": "2026-01-19T12:00:00Z",
          "updated_at": "2026-01-19T12:02:30Z",
          "result": {
            "run_id": 123,
            "raster_urls": ["..."],
            "contour_geojson": {"type": "FeatureCollection", "features": [...]}
          }
        }
        ```
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


async def _job_status_event_stream(
    job_id: UUID,
    poll_interval: float = 2.0,
    max_duration: float = 300.0,  # 5 minutes max
) -> AsyncGenerator[str, None]:
    """Generate SSE events for JIT forecast job status updates.
    
    Streams job status updates as Server-Sent Events for real-time
    progress monitoring without polling.
    
    Args:
        job_id: UUID of the JIT forecast job to monitor
        poll_interval: Seconds between status checks (default: 2.0)
        max_duration: Maximum seconds to keep stream open (default: 300)
        
    Yields:
        SSE formatted event strings with job status updates
    """
    start_time = asyncio.get_event_loop().time()
    last_status = None
    
    status_messages = {
        "pending": "Job is queued and waiting to start...",
        "ingesting_terrain": "Downloading terrain data...",
        "ingesting_weather": "Fetching weather data...",
        "running_forecast": "Generating spread forecast...",
        "completed": "Forecast complete!",
        "failed": "Job failed"
    }
    
    try:
        while True:
            # Check for timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_duration:
                yield f"event: timeout\ndata: {json.dumps({'message': 'Stream timeout - check status endpoint'})}\n\n"
                break
            
            # Get current job status
            job = repo.get_jit_job(job_id)
            
            if not job:
                yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
                break
            
            current_status = job["status"]
            
            # Only send update if status changed
            if current_status != last_status:
                last_status = current_status
                
                event_data = {
                    "job_id": str(job_id),
                    "status": current_status,
                    "progress_message": status_messages.get(current_status, "Processing..."),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                
                if job.get("result"):
                    event_data["result"] = job["result"]
                
                if job.get("error"):
                    event_data["error"] = job["error"]
                
                yield f"event: status\ndata: {json.dumps(event_data)}\n\n"
                
                # End stream if job is complete or failed
                if current_status in ("completed", "failed"):
                    break
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
            
    except asyncio.CancelledError:
        # Client disconnected
        yield f"event: close\ndata: {json.dumps({'message': 'Client disconnected'})}\n\n"
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"


@forecast_router.get(
    "/jit/{job_id}/stream",
    response_class=StreamingResponse,
    dependencies=[Depends(RateLimiter(times=10, seconds=60))],
)
async def stream_jit_forecast_status(
    job_id: UUID,
    poll_interval: float = 2.0,
):
    """Stream JIT forecast job status updates via Server-Sent Events (SSE).
    
    Provides real-time status updates for long-running JIT forecast jobs
    without requiring repeated polling. The connection remains open until
    the job completes, fails, or times out (5 minutes max).
    
    Args:
        job_id: UUID of the JIT forecast job to monitor
        poll_interval: Seconds between status checks (default: 2.0, min: 1.0, max: 10.0)
        
    Returns:
        StreamingResponse with SSE content type for real-time updates
        
    Example:
        ```javascript
        // Browser client example
        const eventSource = new EventSource('/forecast/jit/550e8400-e29b-41d4-a716-446655440000/stream');
        
        eventSource.addEventListener('status', (e) => {
            const data = JSON.parse(e.data);
            console.log('Status:', data.status);
            console.log('Message:', data.progress_message);
            
            if (data.status === 'completed') {
                console.log('Result:', data.result);
                eventSource.close();
            }
        });
        
        eventSource.addEventListener('error', (e) => {
            console.error('Error:', JSON.parse(e.data));
            eventSource.close();
        });
        ```
        
    Events:
        - status: Job status update (queued, ingesting_terrain, ingesting_weather,
                 running_forecast, completed, failed)
        - completed: Job finished successfully (includes result)
        - failed: Job failed (includes error details)
        - error: Stream error (includes error message)
        - timeout: Stream timed out after 5 minutes
        - close: Stream closed (client disconnected)
    """
    # Validate poll_interval
    poll_interval = max(1.0, min(poll_interval, 10.0))
    
    # Check job exists before starting stream
    job = repo.get_jit_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return StreamingResponse(
        _job_status_event_stream(job_id, poll_interval=poll_interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )


def _parse_iso8601_datetime(value: str) -> datetime:
    """Parse ISO 8601 datetime string with robust handling of various formats.
    
    Handles:
    - 'Z' suffix (UTC) - converted to '+00:00' for fromisoformat compatibility
    - No timezone (assumes UTC)
    - Various timezone offset formats (+00:00, +0000, etc.)
    
    Args:
        value: ISO 8601 datetime string
        
    Returns:
        Timezone-aware datetime in UTC
        
    Raises:
        ValueError: If the string cannot be parsed as a valid datetime
    """
    if not value or not value.strip():
        raise ValueError("Empty datetime string")
    
    value = value.strip()
    
    # Handle 'Z' suffix - replace with +00:00 for fromisoformat compatibility
    if value.endswith('Z'):
        value = value[:-1] + '+00:00'
    
    # Handle +0000 format (without colon) - convert to +00:00
    # Match patterns like +0000, -0500, +0530 at the end of the string
    tz_pattern = r'([+-])(\d{2})(\d{2})$'
    match = re.search(tz_pattern, value)
    if match and ':00' not in value[-6:]:  # Only if not already in +00:00 format
        sign, hours, minutes = match.groups()
        value = value[:match.start()] + f"{sign}{hours}:{minutes}"
    
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f"Invalid ISO 8601 datetime format: {value}") from e
    
    # If no timezone was specified, assume UTC
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        # Convert to UTC
        parsed = parsed.astimezone(timezone.utc)
    
    return parsed


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

    # Parse forecast_reference_time with robust ISO 8601 handling
    if request.forecast_reference_time:
        forecast_reference_time = _parse_iso8601_datetime(request.forecast_reference_time)
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
                "weather_fallback_used",
                "weather_fallback_reason",
                "terrain_fallback_used",
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
                "metadata": extra_meta if extra_meta else None,
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

