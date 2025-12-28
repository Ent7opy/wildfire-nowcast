"""FastAPI routes for spread forecasts."""

from __future__ import annotations

import json
from urllib.parse import quote_plus

from fastapi import APIRouter

from api.config import settings
from api.forecast import repo

forecast_router = APIRouter(prefix="/forecast", tags=["forecast"])


@forecast_router.get("")
async def get_forecast(
    region_name: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
):
    """Get the latest spread forecast for an AOI.

    Returns run metadata, raster asset pointers (with TiTiler URLs),
    and vector contours as GeoJSON.
    """
    bbox = (min_lon, min_lat, max_lon, max_lat)
    run = repo.get_latest_forecast_run(region_name, bbox)

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

