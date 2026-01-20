"""Risk index endpoint returning fire risk surface."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query

from api.risk.grid import compute_risk_grid

risk_router = APIRouter(prefix="/risk", tags=["risk"])


@risk_router.get("")
async def get_risk(
    min_lon: float = Query(..., description="Minimum longitude (west boundary)"),
    min_lat: float = Query(..., description="Minimum latitude (south boundary)"),
    max_lon: float = Query(..., description="Maximum longitude (east boundary)"),
    max_lat: float = Query(..., description="Maximum latitude (north boundary)"),
    cell_size_km: float = Query(10.0, ge=1.0, le=100.0, description="Grid cell size in kilometers"),
    time: Optional[str] = Query(None, description="Reference time for weather (ISO 8601, default: now)"),
    include_weather: bool = Query(True, description="Include dynamic weather factors in risk score"),
):
    """
    Return a fire risk heatmap for the requested bbox as a grid of cells.
    
    Each grid cell contains:
    - risk_score (0-1): composite fire risk index
    - risk_level: categorical classification (low/medium/high)
    - components: breakdown of static (landcover) and dynamic (weather) factors
    
    Risk scoring:
    - Static factor (60% weight): landcover-based flammability
      - Forest/shrub/grassland: high risk
      - Cropland: moderate risk
      - Urban/water/desert: low risk
    - Dynamic factor (40% weight, if include_weather=True): weather plausibility
      - Low humidity, wind: increases risk
      - High humidity, recent precipitation: decreases risk
    
    The grid resolution is controlled by cell_size_km (default 10km).
    Grid is capped at 500 cells total to prevent large responses.
    """
    # Parse reference time if provided
    ref_time = None
    if time is not None:
        try:
            ref_time = datetime.fromisoformat(time.replace("Z", "+00:00"))
        except ValueError:
            # Fall back to current time if parsing fails
            pass
    
    return compute_risk_grid(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        cell_size_km=cell_size_km,
        ref_time=ref_time,
        include_weather=include_weather,
    )
