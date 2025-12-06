"""Helpers to read DEM rasters for downstream consumers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import rioxarray  # type: ignore
from xarray import DataArray

from api.core.grid import DEFAULT_CELL_SIZE_DEG, GridSpec
from api.terrain.repo import TerrainMetadata, get_latest_dem_metadata_for_region


def _ensure_xy(da: DataArray) -> DataArray:
    if "x" not in da.dims and "lon" in da.dims:
        da = da.rename({"lon": "x"})
    if "y" not in da.dims and "lat" in da.dims:
        da = da.rename({"lat": "y"})
    return da


def grid_spec_from_metadata(metadata: TerrainMetadata) -> GridSpec:
    """Reconstruct a GridSpec from stored terrain metadata."""
    cell_size = metadata.cell_size_deg or DEFAULT_CELL_SIZE_DEG
    origin_lat = metadata.origin_lat
    origin_lon = metadata.origin_lon
    if origin_lat is None or origin_lon is None:
        min_lon, min_lat, _, _ = metadata.bbox
        origin_lat = math.floor(min_lat / cell_size) * cell_size
        origin_lon = math.floor(min_lon / cell_size) * cell_size

    n_lat = metadata.grid_n_lat
    n_lon = metadata.grid_n_lon
    if n_lat is None or n_lon is None:
        _, _, max_lon, max_lat = metadata.bbox
        n_lat = int(math.ceil((max_lat - origin_lat) / cell_size))
        n_lon = int(math.ceil((max_lon - origin_lon) / cell_size))

    crs = f"EPSG:{metadata.crs_epsg}"
    return GridSpec(
        crs=crs,
        cell_size_deg=cell_size,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        n_lat=n_lat,
        n_lon=n_lon,
    )


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

