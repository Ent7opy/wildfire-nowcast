"""Helpers to read DEM rasters for downstream consumers.

Conventions and alignment: see `docs/terrain_grid.md`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import rioxarray  # type: ignore
from xarray import DataArray

from api.core.grid import DEFAULT_CELL_SIZE_DEG, GridSpec
from api.terrain.repo import TerrainMetadata, find_fallback_dem, get_latest_dem_metadata_for_region
from api.terrain.validate import validate_raster_matches_grid

_LOGGER = logging.getLogger(__name__)


def _ensure_xy(da: DataArray) -> DataArray:
    if "x" not in da.dims and "lon" in da.dims:
        da = da.rename({"lon": "x"})
    if "y" not in da.dims and "lat" in da.dims:
        da = da.rename({"lat": "y"})
    return da


def _to_analysis_convention(da: DataArray) -> DataArray:
    """Normalize DEM output to analysis convention: dims (lat, lon), ascending coords."""
    da = _ensure_xy(da)
    # Rename to canonical dim names
    da = da.rename({"y": "lat", "x": "lon"})
    # Ensure monotonic increasing lat/lon (south→north, west→east)
    if "lat" in da.coords:
        da = da.sortby("lat")
    if "lon" in da.coords:
        da = da.sortby("lon")
    # Ensure canonical dim order for consumers (lat, lon)
    return da.transpose("lat", "lon")


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


def _flat_dem_stub(
    bbox: Tuple[float, float, float, float],
    cell_size_deg: float = DEFAULT_CELL_SIZE_DEG,
) -> DataArray:
    """Return a zero-elevation DataArray for bbox with terrain_fallback_used=True in attrs."""
    import xarray as xr

    min_lon, min_lat, max_lon, max_lat = bbox
    lats = np.arange(min_lat + cell_size_deg / 2, max_lat, cell_size_deg)
    lons = np.arange(min_lon + cell_size_deg / 2, max_lon, cell_size_deg)
    data = np.zeros((len(lats), len(lons)), dtype=np.float32)
    da = xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lats, "lon": lons})
    da.attrs["terrain_fallback_used"] = True
    da.attrs["fallback_reason"] = "dem_file_missing"
    return da


def load_dem_for_bbox(
    region_name: str, bbox: Tuple[float, float, float, float]
) -> DataArray:
    """Load the latest DEM for a region and clip to bbox (lon/lat).

    Fallback chain when the primary DEM file is missing on disk:
    1. Try any other registered DEM for the region (resolution ladder, finest first).
    2. Try 'global_base' region.
    3. Return a flat-terrain stub (elevation=0) with ``terrain_fallback_used=True`` in
       ``.attrs``.  A WARNING is logged so operators know to ingest the missing tile.
    """
    metadata = get_latest_dem_metadata_for_region(region_name)
    if metadata is None:
        metadata = get_latest_dem_metadata_for_region("global_base")

    if metadata is None:
        raise ValueError(f"No DEM metadata found for region '{region_name}' or 'global_base'.")

    raster_path = Path(metadata.raster_path)
    if not raster_path.exists():
        fallback_md = find_fallback_dem([region_name, "global_base"], skip_path=raster_path)
        if fallback_md is not None:
            _LOGGER.warning(
                "DEM raster missing at %s for region '%s' (bbox=%s). "
                "Falling back to lower-resolution DEM: %s (%.0f m). "
                "Mitigation: re-ingest the missing tile.",
                raster_path,
                region_name,
                bbox,
                fallback_md.raster_path,
                fallback_md.resolution_m,
            )
            raster_path = Path(fallback_md.raster_path)
            metadata = fallback_md
        else:
            _LOGGER.warning(
                "DEM raster missing at %s for region '%s' (bbox=%s) and no lower-resolution "
                "fallback found. Returning flat-terrain stub (elevation=0, terrain_fallback_used=True). "
                "Mitigation: run DEM ingest for this region.",
                raster_path,
                region_name,
                bbox,
            )
            return _flat_dem_stub(bbox)

    # Fail fast if the raster is misaligned with the stored grid.
    grid = grid_spec_from_metadata(metadata)
    validate_raster_matches_grid(raster_path, grid, strict=True)

    da = rioxarray.open_rasterio(raster_path, masked=True)
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    da = _ensure_xy(da)

    min_lon, min_lat, max_lon, max_lat = bbox
    clipped = da.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat, crs="EPSG:4326")
    return _to_analysis_convention(clipped)

