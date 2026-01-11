"""FastAPI proxy for vector tiles."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, Depends
import httpx
from fastapi_limiter.depends import RateLimiter
from api.config import settings

tiles_router = APIRouter(prefix="/tiles", tags=["tiles"])

# Map internal layer names to pg_tileserv function/table names
LAYER_MAPPING = {
    "fires": "public.mvt_fires",
    "forecast_contours": "public.mvt_forecast_contours",
    "aois": "public.mvt_aois",
}

@tiles_router.get(
    "/{layer}/{z}/{x}/{y}.pbf",
    dependencies=[Depends(RateLimiter(times=500, seconds=60))]
)
async def proxy_tile(layer: str, z: int, x: int, y: int, request: Request):
    """Proxy vector tile requests to pg_tileserv."""
    if layer not in LAYER_MAPPING:
        raise HTTPException(status_code=404, detail=f"Layer {layer} not found")
    
    internal_layer = LAYER_MAPPING[layer]
    url = f"{settings.vector_tiles_internal_base_url}/{internal_layer}/{z}/{x}/{y}.pbf"
    
    # Forward query parameters
    params = dict(request.query_params)
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, params=params)
            # Proxy the response
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
                headers={
                    # Default conservative cache control if missing
                    "Cache-Control": resp.headers.get("Cache-Control", "public, max-age=60") 
                }
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Tile server unavailable: {e}")
