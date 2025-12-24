from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query

from api.fires.repo import list_fire_detections_bbox_time

fires_router = APIRouter(prefix="/fires", tags=["fires"])


@fires_router.get("/detections")
async def get_detections(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_time: datetime,
    end_time: datetime,
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
    columns = ["id", "lat", "lon", "acq_time", "confidence", "frp", "sensor", "source"]
    if include_denoiser_fields:
        columns.extend(["denoised_score", "is_noise"])

    detections = list_fire_detections_bbox_time(
        bbox=(min_lon, min_lat, max_lon, max_lat),
        start_time=start_time,
        end_time=end_time,
        columns=columns,
        include_noise=include_noise,
        limit=limit,
    )

    return {"count": len(detections), "detections": detections}

