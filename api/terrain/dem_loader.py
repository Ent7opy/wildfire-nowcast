"""Helpers to read DEM rasters for downstream consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import rioxarray  # type: ignore
from xarray import DataArray

from api.terrain.repo import get_latest_dem_metadata_for_region


def _ensure_xy(da: DataArray) -> DataArray:
    if "x" not in da.dims and "lon" in da.dims:
        da = da.rename({"lon": "x"})
    if "y" not in da.dims and "lat" in da.dims:
        da = da.rename({"lat": "y"})
    return da


def load_dem_for_bbox(
    region_name: str, bbox: Tuple[float, float, float, float]
) -> DataArray:
    """Load the latest DEM for a region and clip to bbox (lon/lat)."""
    metadata = get_latest_dem_metadata_for_region(region_name)
    if metadata is None:
        raise ValueError(f"No DEM metadata found for region '{region_name}'.")

    raster_path = Path(metadata.raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"DEM raster not found at {raster_path}")

    da = rioxarray.open_rasterio(raster_path, masked=True)
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    da = _ensure_xy(da)

    min_lon, min_lat, max_lon, max_lat = bbox
    clipped = da.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat, crs="EPSG:4326")
    return _ensure_xy(clipped)

