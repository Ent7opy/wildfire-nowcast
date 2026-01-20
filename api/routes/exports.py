"""FastAPI routes for synchronous data exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import os
from typing import Any, Generator
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status, Depends
from fastapi.responses import StreamingResponse, FileResponse
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel

from api.aois import repo as aois_repo
from api.exports import repo as jobs_repo
from api.exports.worker import queue, export_task
from api.fires import repo as fires_repo
from api.forecast import repo as forecast_repo
from api.exports.map_renderer import render_map_png

exports_router = APIRouter(tags=["exports"])

MAX_SYNC_FEATURES = 10000


def _stream_csv(data: list[dict[str, Any]], filename: str) -> StreamingResponse:
    if not data:
        resp = Response(content="", media_type="text/csv")
        resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
        return resp

    fieldnames = list(data[0].keys())
    
    def iter_csv() -> Generator[str, None, None]:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)
        
        for row in data:
            writer.writerow(row)
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    response = StreamingResponse(iter_csv(), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


def _json_response(data: Any, filename: str) -> Response:
    # MVP: dump to JSON string. For large data, this should be streaming too.
    content = json.dumps(data)
    response = Response(content=content, media_type="application/json")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


@exports_router.get("/aois/{aoi_id}/export")
def export_aoi(aoi_id: UUID, format: str = Query("geojson", pattern="^(geojson)$")):
    """Export an AOI geometry."""
    aoi = aois_repo.get_aoi(aoi_id)
    if not aoi:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found")
    
    # Construct Feature
    feature = {
        "type": "Feature",
        "geometry": aoi["geometry"],
        "properties": {
            "id": str(aoi["id"]),
            "name": aoi["name"],
            "description": aoi["description"],
            "area_km2": aoi["area_km2"],
            "created_at": str(aoi["created_at"]),
        }
    }
    
    return _json_response(feature, f"aoi_{aoi_id}.geojson")


@exports_router.get("/fires/export")
def export_fires(
    min_lon: float = Query(..., description="Minimum longitude"),
    min_lat: float = Query(..., description="Minimum latitude"),
    max_lon: float = Query(..., description="Maximum longitude"),
    max_lat: float = Query(..., description="Maximum latitude"),
    start_time: str = Query(..., description="Start time (ISO 8601)"),
    end_time: str = Query(..., description="End time (ISO 8601)"),
    format: str = Query("csv", pattern="^(csv|geojson)$"),
    limit: int = Query(1000, ge=1, le=MAX_SYNC_FEATURES),
):
    """Export fire detections."""
    
    # Parse times (simplified for MVP, ideally share parsing logic)
    try:
        dt_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        dt_end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        if dt_start.tzinfo is None:
            dt_start = dt_start.replace(tzinfo=timezone.utc)
        if dt_end.tzinfo is None:
            dt_end = dt_end.replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamps")

    # Reuse repo logic
    detections = fires_repo.list_fire_detections_bbox_time(
        bbox=(min_lon, min_lat, max_lon, max_lat),
        start_time=dt_start,
        end_time=dt_end,
        limit=limit,
        columns=["id", "lat", "lon", "acq_time", "confidence", "frp", "sensor", "source"] 
    )
    
    if format == "csv":
        return _stream_csv(detections, f"fires_{start_time}_{end_time}.csv")
    
    if format == "geojson":
        features = []
        for d in detections:
            props = dict(d)
            lat = props.pop("lat")
            lon = props.pop("lon")
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {k: str(v) if k == "acq_time" else v for k, v in props.items()}
            })
        fc = {"type": "FeatureCollection", "features": features}
        return _json_response(fc, f"fires_{start_time}_{end_time}.geojson")


@exports_router.get("/forecast/{run_id}/contours/export")
def export_forecast_contours(run_id: int, format: str = Query("geojson", pattern="^(geojson)$")):
    """Export forecast contours."""
    contours = forecast_repo.list_contours_for_run(run_id)
    if not contours:
        raise HTTPException(status_code=404, detail="No contours found for run")
        
    features = []
    for c in contours:
        features.append({
            "type": "Feature",
            "geometry": json.loads(c["geom_geojson"]),
            "properties": {
                "horizon_hours": c["horizon_hours"],
                "threshold": c["threshold"]
            }
        })
    
    fc = {"type": "FeatureCollection", "features": features}
    return _json_response(fc, f"forecast_run_{run_id}_contours.geojson")


@exports_router.get("/risk/export")
def export_risk(
    min_lon: float = Query(..., description="Minimum longitude"),
    min_lat: float = Query(..., description="Minimum latitude"),
    max_lon: float = Query(..., description="Maximum longitude"),
    max_lat: float = Query(..., description="Maximum latitude"),
    format: str = Query("geojson", pattern="^(csv|geojson)$"),
    cell_size_km: float = Query(10.0, ge=1.0, le=100.0, description="Grid cell size in kilometers"),
    time: str = Query(None, description="Reference time for weather (ISO 8601)"),
    include_weather: bool = Query(True, description="Include weather factors"),
):
    """Export fire risk grid for the requested bbox."""
    from api.risk.grid import compute_risk_grid
    
    # Parse reference time if provided
    ref_time = None
    if time is not None:
        try:
            ref_time = datetime.fromisoformat(time.replace("Z", "+00:00"))
        except ValueError:
            pass
    
    # Compute risk grid
    risk_data = compute_risk_grid(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        cell_size_km=cell_size_km,
        ref_time=ref_time,
        include_weather=include_weather,
    )
    
    if format == "geojson":
        filename = f"risk_{min_lon}_{min_lat}_{max_lon}_{max_lat}.geojson"
        return _json_response(risk_data, filename)
    
    if format == "csv":
        # Convert grid features to CSV rows
        rows = []
        for feature in risk_data["features"]:
            props = feature["properties"]
            # Extract cell center from geometry
            coords = feature["geometry"]["coordinates"][0]
            center_lon = sum(c[0] for c in coords[:-1]) / 4
            center_lat = sum(c[1] for c in coords[:-1]) / 4
            
            row = {
                "center_lon": center_lon,
                "center_lat": center_lat,
                "risk_score": props["risk_score"],
                "risk_level": props["risk_level"],
            }
            # Add component scores if available
            if "components" in props:
                for key, val in props["components"].items():
                    row[f"component_{key}"] = val
            
            rows.append(row)
        
        filename = f"risk_{min_lon}_{min_lat}_{max_lon}_{max_lat}.csv"
        return _stream_csv(rows, filename)


@exports_router.get("/map.png")
def export_map_png(
    min_lon: float = Query(..., description="Minimum longitude"),
    min_lat: float = Query(..., description="Minimum latitude"),
    max_lon: float = Query(..., description="Maximum longitude"),
    max_lat: float = Query(..., description="Maximum latitude"),
    start_time: str = Query(None, description="Start time for fires (ISO 8601)"),
    end_time: str = Query(None, description="End time for fires (ISO 8601)"),
    run_id: int = Query(None, description="Forecast run ID"),
    include_fires: bool = Query(True, description="Include fire detections"),
    include_risk: bool = Query(True, description="Include risk heatmap"),
    include_forecast: bool = Query(False, description="Include forecast contours"),
    width: int = Query(1600, ge=400, le=4000, description="Image width in pixels"),
    height: int = Query(900, ge=300, le=3000, description="Image height in pixels"),
):
    """Export current map view as PNG image.
    
    Renders a static map image with selected layers (fires, risk, forecast).
    The image includes a legend and timestamp.
    """
    from api.risk.grid import compute_risk_grid
    
    bbox = (min_lon, min_lat, max_lon, max_lat)
    
    # Gather layer data
    fires = None
    if include_fires and start_time and end_time:
        try:
            dt_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            dt_end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            if dt_start.tzinfo is None:
                dt_start = dt_start.replace(tzinfo=timezone.utc)
            if dt_end.tzinfo is None:
                dt_end = dt_end.replace(tzinfo=timezone.utc)
            
            fires = fires_repo.list_fire_detections_bbox_time(
                bbox=bbox,
                start_time=dt_start,
                end_time=dt_end,
                limit=5000,
                columns=["lat", "lon", "frp"]
            )
        except Exception:
            # If fires fail, just skip them
            fires = None
    
    risk_grid = None
    if include_risk:
        try:
            risk_grid = compute_risk_grid(
                min_lon=min_lon,
                min_lat=min_lat,
                max_lon=max_lon,
                max_lat=max_lat,
                cell_size_km=10.0,
                include_weather=True,
            )
        except Exception:
            # If risk fails, skip it
            risk_grid = None
    
    forecast_contours = None
    if include_forecast and run_id:
        try:
            forecast_contours = forecast_repo.list_contours_for_run(run_id)
        except Exception:
            forecast_contours = None
    
    # Render PNG
    try:
        png_bytes = render_map_png(
            bbox=bbox,
            fires=fires,
            risk_grid=risk_grid,
            forecast_contours=forecast_contours,
            width=width,
            height=height,
            title="Wildfire Nowcast & Forecast",
        )
        
        # Return PNG response
        filename = f"map_{min_lon}_{min_lat}_{max_lon}_{max_lat}.png"
        response = Response(content=png_bytes, media_type="image/png")
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        return response
        
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PNG export requires Pillow library. Install with: pip install pillow"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PNG: {str(e)}"
        )


# Async Exports

class ExportJobRequest(BaseModel):
    kind: str
    request: dict[str, Any]

class ExportJobResponse(BaseModel):
    job_id: UUID
    status: str

@exports_router.post(
    "/exports",
    response_model=ExportJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(RateLimiter(times=10, seconds=60))]
)
def create_export_job(job_request: ExportJobRequest):
    """Enqueue an async export job."""
    job = jobs_repo.create_job(job_request.kind, job_request.request)
    job_id = job["id"]
    
    try:
        # Enqueue in Redis
        queue.enqueue(export_task, job_id, job_request.kind, job_request.request)
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        # Update job status to failed so it's not stuck
        jobs_repo.update_job_status(job_id, "failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue export job: {str(e)}"
        )

@exports_router.get("/exports/{job_id}")
def get_export_job(job_id: UUID):
    """Get export job status."""
    job = jobs_repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job

@exports_router.get("/exports/{job_id}/download")
def download_export_job(job_id: UUID):
    """Download export result."""
    job = jobs_repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    
    if job["status"] != "succeeded":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job not ready or failed")
        
    result = job.get("result") or {}
    file_path = result.get("file_path")
    
    if file_path:
        if not os.path.exists(file_path):
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found on server")
        
        # Derive filename from kind and original file
        kind = job.get("kind", "export")
        original_name = os.path.basename(file_path)
        filename = f"{kind}_{job_id}_{original_name}"
        
        return FileResponse(file_path, filename=filename)
    
    return result
