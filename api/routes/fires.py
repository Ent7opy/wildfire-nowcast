from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi_limiter.depends import RateLimiter

from api.core.weather import get_weather_context_for_point
from api.deps import cache_60, get_fire_repo
from api.errors import InvalidBoundingBoxError
from api.fires.repository import FireRepository
from api.fires.geocoding import reverse_geocode_point


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

FIRE_DETECTION_DENOISER_COLUMNS = [
    "denoised_score",
    "is_noise",
    "event_id",
    "event_score",
    "denoiser_decision",
    "review_required",
]

fires_router = APIRouter(prefix="/fires", tags=["fires"])


def _list_detections(
    repo: FireRepository,
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
    cursor: Optional[str],
    offset: Optional[int],
):
    # Validate bbox coordinates
    try:
        repo.validate_bbox((min_lon, min_lat, max_lon, max_lat))
    except ValueError as e:
        raise InvalidBoundingBoxError(str(e)) from e

    columns = FIRE_DETECTION_BASE_COLUMNS.copy()
    if include_denoiser_fields:
        columns.extend(FIRE_DETECTION_DENOISER_COLUMNS)

    try:
        result = repo.list_fire_detections_bbox_time(
            bbox=(min_lon, min_lat, max_lon, max_lat),
            start_time=start_time,
            end_time=end_time,
            columns=columns,
            include_noise=include_noise,
            include_masked=include_masked,
            limit=limit,
            min_confidence=min_confidence,
            min_fire_likelihood=min_fire_likelihood,
            cursor=cursor,
            offset=offset,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "count": len(result["data"]),
        "detections": result["data"],
        "next_cursor": result["next_cursor"],
        "has_more": result["has_more"],
    }


@fires_router.get("", dependencies=[Depends(RateLimiter(times=30, seconds=60)), Depends(cache_60)])
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
    cursor: Optional[str] = Query(None, description="Opaque pagination cursor from a previous response's next_cursor field."),
    offset: Optional[int] = Query(None, ge=0, description="(Deprecated) Row offset for pagination. Use cursor instead."),
    repo: FireRepository = Depends(get_fire_repo),
):
    """Alias for `/fires/detections` (kept for UI/backward compatibility)."""
    return _list_detections(
        repo,
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
        cursor=cursor,
        offset=offset,
    )


@fires_router.get("/detections", dependencies=[Depends(RateLimiter(times=30, seconds=60)), Depends(cache_60)])
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
    cursor: Optional[str] = Query(None, description="Opaque pagination cursor from a previous response's next_cursor field."),
    offset: Optional[int] = Query(None, ge=0, description="(Deprecated) Row offset for pagination. Use cursor instead."),
    repo: FireRepository = Depends(get_fire_repo),
):
    """
    Get raw fire detections within a spatio-temporal window.

    Supports cursor-based pagination via the ``cursor`` param (pass the
    ``next_cursor`` value from the previous response). The legacy ``offset``
    param is still accepted but performs a sequential scan — prefer ``cursor``.

    By default, only non-noise detections (or those not yet scored) are returned.
    """
    return _list_detections(
        repo,
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
        cursor=cursor,
        offset=offset,
    )


@fires_router.get(
    "/detections/{detection_id}",
    dependencies=[Depends(RateLimiter(times=60, seconds=60)), Depends(cache_60)],
)
async def get_detection_detail(
    detection_id: int = Path(..., description="Fire detection primary key"),
    repo: FireRepository = Depends(get_fire_repo),
):
    """Return a single fire detection with weather context.

    The response includes a ``weather`` block when GFS data covers the
    detection's location.  When no weather data is available, ``weather``
    is ``null`` with a ``weather_unavailable_reason`` string.
    """
    detection = repo.get_fire_detection_by_id(detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="Detection not found")

    acq_time = detection.get("acq_time")
    lat = detection.get("lat")
    lon = detection.get("lon")

    weather = None
    weather_unavailable_reason: str | None = None

    if lat is not None and lon is not None and acq_time is not None:
        weather = get_weather_context_for_point(
            lat=lat,
            lon=lon,
            ref_time=acq_time,
        )
        if weather is None:
            weather_unavailable_reason = (
                "No GFS weather run covers this location within the tolerance window"
            )
    else:
        weather_unavailable_reason = (
            "Detection is missing coordinates or acquisition time"
        )

    return {
        **detection,
        "weather": weather,
        "weather_unavailable_reason": weather_unavailable_reason,
    }


@fires_router.get("/events", dependencies=[Depends(RateLimiter(times=30, seconds=60)), Depends(cache_60)])
async def get_events(
    min_lon: float = Query(..., description="Minimum longitude (west boundary)"),
    min_lat: float = Query(..., description="Minimum latitude (south boundary)"),
    max_lon: float = Query(..., description="Maximum longitude (east boundary)"),
    max_lat: float = Query(..., description="Maximum latitude (north boundary)"),
    start_time: datetime = Query(..., description="Start time for the query window (ISO 8601 format)"),
    end_time: datetime = Query(..., description="End time for the query window (ISO 8601 format)"),
    min_event_score: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Minimum event-level denoiser score."
    ),
    include_review_required: bool = Query(
        True,
        description="Include events currently marked as requiring review.",
    ),
    limit: Optional[int] = Query(1000, gt=0, le=10000),
    cursor: Optional[str] = Query(None, description="Opaque pagination cursor from a previous response's next_cursor field."),
    offset: Optional[int] = Query(None, ge=0, description="(Deprecated) Row offset for pagination. Use cursor instead."),
    repo: FireRepository = Depends(get_fire_repo),
):
    """Get fire events within a spatio-temporal window."""
    try:
        repo.validate_bbox((min_lon, min_lat, max_lon, max_lat))
    except ValueError as e:
        raise InvalidBoundingBoxError(str(e)) from e

    try:
        result = repo.list_fire_events_bbox_time(
            bbox=(min_lon, min_lat, max_lon, max_lat),
            start_time=start_time,
            end_time=end_time,
            min_event_score=min_event_score,
            include_review_required=include_review_required,
            limit=int(limit or 1000),
            cursor=cursor,
            offset=offset,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "count": len(result["data"]),
        "events": result["data"],
        "next_cursor": result["next_cursor"],
        "has_more": result["has_more"],
    }


@fires_router.get("/fronts", dependencies=[Depends(RateLimiter(times=30, seconds=60)), Depends(cache_60)])
async def get_fronts(
    min_lon: float = Query(..., description="Minimum longitude (west boundary)"),
    min_lat: float = Query(..., description="Minimum latitude (south boundary)"),
    max_lon: float = Query(..., description="Maximum longitude (east boundary)"),
    max_lat: float = Query(..., description="Maximum latitude (north boundary)"),
    start_time: datetime = Query(..., description="Start time for the query window (ISO 8601 format)"),
    end_time: datetime = Query(..., description="End time for the query window (ISO 8601 format)"),
    min_event_score: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Minimum linked event-level denoiser score."
    ),
    include_review_required: bool = Query(
        True,
        description="Include fronts linked to events currently marked as requiring review.",
    ),
    limit: Optional[int] = Query(2000, gt=0, le=10000),
    cursor: Optional[str] = Query(None, description="Opaque pagination cursor from a previous response's next_cursor field."),
    offset: Optional[int] = Query(None, ge=0, description="(Deprecated) Row offset for pagination. Use cursor instead."),
    repo: FireRepository = Depends(get_fire_repo),
):
    """Get fire fronts within a spatio-temporal window."""
    try:
        repo.validate_bbox((min_lon, min_lat, max_lon, max_lat))
    except ValueError as e:
        raise InvalidBoundingBoxError(str(e)) from e

    try:
        result = repo.list_fire_fronts_bbox_time(
            bbox=(min_lon, min_lat, max_lon, max_lat),
            start_time=start_time,
            end_time=end_time,
            min_event_score=min_event_score,
            include_review_required=include_review_required,
            limit=int(limit or 2000),
            cursor=cursor,
            offset=offset,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "count": len(result["data"]),
        "fronts": result["data"],
        "next_cursor": result["next_cursor"],
        "has_more": result["has_more"],
    }


@fires_router.get("/reverse-geocode", dependencies=[Depends(RateLimiter(times=120, seconds=60))])
async def get_reverse_geocode(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude for reverse geocoding."),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude for reverse geocoding."),
):
    """Resolve a human-readable place label for a coordinate."""
    try:
        return reverse_geocode_point(lat=lat, lon=lon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
