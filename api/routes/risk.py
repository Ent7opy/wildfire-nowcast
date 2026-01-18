"""Risk index endpoint returning baseline risk surface."""

from fastapi import APIRouter, Query

risk_router = APIRouter(prefix="/risk", tags=["risk"])


@risk_router.get("")
async def get_risk(
    min_lon: float = Query(..., description="Minimum longitude (west boundary)"),
    min_lat: float = Query(..., description="Minimum latitude (south boundary)"),
    max_lon: float = Query(..., description="Maximum longitude (east boundary)"),
    max_lat: float = Query(..., description="Maximum latitude (north boundary)"),
):
    """
    Return a simple baseline risk surface for the requested bbox.
    
    This is a placeholder baseline that returns a uniform low risk value
    across the requested area. Future versions will incorporate terrain,
    weather, vegetation, and historical fire data.
    """
    # Return a simple GeoJSON FeatureCollection with a single polygon
    # covering the bbox with a baseline risk score
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]]
                },
                "properties": {
                    "risk_score": 0.3,
                    "risk_level": "low",
                    "description": "Baseline risk (placeholder)"
                }
            }
        ]
    }
