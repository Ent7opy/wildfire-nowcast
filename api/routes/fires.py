from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query

from api.fires.repo import list_fire_detections_bbox_time

fires_router = APIRouter(prefix="/fires", tags=["fires"])


def _list_detections(
    *,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_time: datetime,
    end_time: datetime,
    min_confidence: Optional[float],
    include_noise: bool,
    include_denoiser_fields: bool,
    limit: Optional[int],
):
    columns = [
        "id",
        "lat",
        "lon",
        "acq_time",
        "confidence",
        "brightness",
        "bright_t31",
        "frp",
        "sensor",
        "source",
    ]
    if include_denoiser_fields:
        columns.extend(["denoised_score", "is_noise"])

    detections = list_fire_detections_bbox_time(
        bbox=(min_lon, min_lat, max_lon, max_lat),
        start_time=start_time,
        end_time=end_time,
        columns=columns,
        include_noise=include_noise,
        limit=limit,
        min_confidence=min_confidence,
    )

    return {"count": len(detections), "detections": detections}


@fires_router.get("")
async def get_fires(
    min_lon: float = Query(..., description="Minimum longitude (west boundary)"),
    min_lat: float = Query(..., description="Minimum latitude (south boundary)"),
    max_lon: float = Query(..., description="Maximum longitude (east boundary)"),
    max_lat: float = Query(..., description="Maximum latitude (north boundary)"),
    start_time: datetime = Query(..., description="Start time for the query window (ISO 8601 format)"),
    end_time: datetime = Query(..., description="End time for the query window (ISO 8601 format)"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=100.0),
    include_noise: bool = Query(False, description="Include detections explicitly marked as noise."),
    include_denoiser_fields: bool = Query(
        False, description="Include denoised_score and is_noise in response."
    ),
    limit: Optional[int] = Query(None, gt=0, le=10000),
):
    """Alias for `/fires/detections` (kept for UI/backward compatibility)."""
    return _list_detections(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start_time=start_time,
        end_time=end_time,
        include_noise=include_noise,
        include_denoiser_fields=include_denoiser_fields,
        limit=limit,
        min_confidence=min_confidence,
    )


@fires_router.get("/detections")
async def get_detections(
    min_lon: float = Query(..., description="Minimum longitude (west boundary)"),
    min_lat: float = Query(..., description="Minimum latitude (south boundary)"),
    max_lon: float = Query(..., description="Maximum longitude (east boundary)"),
    max_lat: float = Query(..., description="Maximum latitude (north boundary)"),
    start_time: datetime = Query(..., description="Start time for the query window (ISO 8601 format)"),
    end_time: datetime = Query(..., description="End time for the query window (ISO 8601 format)"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=100.0),
    include_noise: bool = Query(False, description="Include detections explicitly marked as noise."),
    include_denoiser_fields: bool = Query(
        False, description="Include denoised_score and is_noise in response."
    ),
    limit: Optional[int] = Query(None, gt=0, le=10000),
):
    """
    Get raw fire detections within a spatio-temporal window.
    
    By default, only non-noise detections (or those not yet scored) are returned.
    """
    return _list_detections(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start_time=start_time,
        end_time=end_time,
        include_noise=include_noise,
        include_denoiser_fields=include_denoiser_fields,
        limit=limit,
        min_confidence=min_confidence,
    )

