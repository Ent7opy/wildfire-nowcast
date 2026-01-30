from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_limiter.depends import RateLimiter

from api.fires.repo import validate_bbox, list_fire_detections_bbox_time


# Standard fire detection columns - defined centrally to stay in sync with schema
FIRE_DETECTION_BASE_COLUMNS = [
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
    "confidence_score",
    "persistence_score",
    "landcover_score",
    "weather_score",
    "false_source_masked",
    "fire_likelihood",
]

FIRE_DETECTION_DENOISER_COLUMNS = ["denoised_score", "is_noise"]

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
    min_fire_likelihood: Optional[float],
    include_noise: bool,
    include_masked: bool,
    include_denoiser_fields: bool,
    limit: Optional[int],
):
    # Validate bbox coordinates
    try:
        validate_bbox((min_lon, min_lat, max_lon, max_lat))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    columns = FIRE_DETECTION_BASE_COLUMNS.copy()
    if include_denoiser_fields:
        columns.extend(FIRE_DETECTION_DENOISER_COLUMNS)

    detections = list_fire_detections_bbox_time(
        bbox=(min_lon, min_lat, max_lon, max_lat),
        start_time=start_time,
        end_time=end_time,
        columns=columns,
        include_noise=include_noise,
        include_masked=include_masked,
        limit=limit,
        min_confidence=min_confidence,
        min_fire_likelihood=min_fire_likelihood,
    )

    return {"count": len(detections), "detections": detections}


@fires_router.get("", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
async def get_fires(
    min_lon: float = Query(..., description="Minimum longitude (west boundary)"),
    min_lat: float = Query(..., description="Minimum latitude (south boundary)"),
    max_lon: float = Query(..., description="Maximum longitude (east boundary)"),
    max_lat: float = Query(..., description="Maximum latitude (north boundary)"),
    start_time: datetime = Query(..., description="Start time for the query window (ISO 8601 format)"),
    end_time: datetime = Query(..., description="End time for the query window (ISO 8601 format)"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=100.0, description="Minimum FIRMS confidence (deprecated, use min_fire_likelihood)"),
    min_fire_likelihood: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum fire likelihood score"),
    include_noise: bool = Query(False, description="Include detections explicitly marked as noise."),
    include_masked: bool = Query(False, description="Include detections near known industrial false-positive sources."),
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
        include_masked=include_masked,
        include_denoiser_fields=include_denoiser_fields,
        limit=limit,
        min_confidence=min_confidence,
        min_fire_likelihood=min_fire_likelihood,
    )


@fires_router.get("/detections", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
async def get_detections(
    min_lon: float = Query(..., description="Minimum longitude (west boundary)"),
    min_lat: float = Query(..., description="Minimum latitude (south boundary)"),
    max_lon: float = Query(..., description="Maximum longitude (east boundary)"),
    max_lat: float = Query(..., description="Maximum latitude (north boundary)"),
    start_time: datetime = Query(..., description="Start time for the query window (ISO 8601 format)"),
    end_time: datetime = Query(..., description="End time for the query window (ISO 8601 format)"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=100.0, description="Minimum FIRMS confidence (deprecated, use min_fire_likelihood)"),
    min_fire_likelihood: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum fire likelihood score"),
    include_noise: bool = Query(False, description="Include detections explicitly marked as noise."),
    include_masked: bool = Query(False, description="Include detections near known industrial false-positive sources."),
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
        include_masked=include_masked,
        include_denoiser_fields=include_denoiser_fields,
        limit=limit,
        min_confidence=min_confidence,
        min_fire_likelihood=min_fire_likelihood,
    )

