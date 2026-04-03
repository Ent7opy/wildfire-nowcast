"""Ignition probability endpoint."""

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import JSONResponse

from api.deps import cache_control
from api.ignition.grid import (
    IgnitionInferenceFailed,
    IgnitionModelUnavailable,
    compute_ignition_grid,
)

ignition_router = APIRouter()

cache_21600 = cache_control(21600)

_VALID_HORIZONS = {"now", "+24h", "+48h"}


@ignition_router.get("", dependencies=[Depends(cache_21600)])
def get_ignition(
    response: Response,
    min_lon: float = Query(...),
    min_lat: float = Query(...),
    max_lon: float = Query(...),
    max_lat: float = Query(...),
    cell_size_km: float = Query(10.0, ge=1.0, le=50.0),
    horizon: str = Query("now"),
):
    # Ensure HTTP intermediaries cache each horizon as a distinct resource
    response.headers["Vary"] = "horizon"

    if horizon not in _VALID_HORIZONS:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_horizon", "detail": f"horizon must be one of {sorted(_VALID_HORIZONS)}"},
        )

    try:
        result = compute_ignition_grid(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            cell_size_km=cell_size_km,
            horizon=horizon,
        )
    except (IgnitionModelUnavailable, IgnitionInferenceFailed):
        return JSONResponse(
            status_code=503,
            content={
                "error": "ignition_model_unavailable",
                "detail": "No promoted ignition model found.",
            },
        )

    return result
