"""Helpers to read slope/aspect rasters for downstream consumers.

Conventions and alignment: see `docs/terrain_grid.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import rioxarray  # type: ignore
from xarray import DataArray

from api.core.grid import GridSpec
from api.terrain.features_repo import (
    TerrainFeaturesMetadata,
    get_latest_terrain_features_metadata_for_region,
)
from api.terrain.validate import validate_terrain_stack


def _ensure_xy(da: DataArray) -> DataArray:
    if "x" not in da.dims and "lon" in da.dims:
        da = da.rename({"lon": "x"})
    if "y" not in da.dims and "lat" in da.dims:
        da = da.rename({"lat": "y"})
    return da


def _to_analysis_convention(da: DataArray) -> DataArray:
    """Normalize output to analysis convention: dims (lat, lon), ascending coords."""
    da = _ensure_xy(da)
    da = da.rename({"y": "lat", "x": "lon"})
    if "lat" in da.coords:
        da = da.sortby("lat")
    if "lon" in da.coords:
        da = da.sortby("lon")
    return da.transpose("lat", "lon")


def _load_feature_raster(path: Path) -> DataArray:
    da = rioxarray.open_rasterio(path, masked=True)
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    # Keep spatial dimension names as x/y here so rioxarray can reliably
    # identify spatial axes for operations like rio.clip_box. We convert to
    # analysis convention (lat/lon) after clipping, mirroring dem_loader.py.
    return _ensure_xy(da)


def _require_latest_metadata(region_name: str) -> TerrainFeaturesMetadata:
    metadata = get_latest_terrain_features_metadata_for_region(region_name)
    if metadata is None:
        raise ValueError(f"No terrain features metadata found for region '{region_name}'.")
    return metadata


def load_slope_aspect_for_bbox(
    region_name: str,
    bbox: Tuple[float, float, float, float],
) -> tuple[DataArray, DataArray]:
    """Load the latest slope/aspect for a region and clip to bbox (lon/lat)."""
    metadata = _require_latest_metadata(region_name)
    slope_path = Path(metadata.slope_path)
    aspect_path = Path(metadata.aspect_path)
    if not slope_path.exists():
        raise FileNotFoundError(f"Slope raster not found at {slope_path}")
    if not aspect_path.exists():
        raise FileNotFoundError(f"Aspect raster not found at {aspect_path}")

    # Fail fast if the stored rasters do not match the stored grid contract.
    grid = GridSpec(
        crs=f"EPSG:{metadata.crs_epsg}",
        cell_size_deg=float(metadata.cell_size_deg),
        origin_lat=float(metadata.origin_lat),
        origin_lon=float(metadata.origin_lon),
        n_lat=int(metadata.grid_n_lat),
        n_lon=int(metadata.grid_n_lon),
    )
    validate_terrain_stack(None, slope_path, aspect_path, grid, strict=True)

    slope = _load_feature_raster(slope_path)
    aspect = _load_feature_raster(aspect_path)

    min_lon, min_lat, max_lon, max_lat = bbox
    slope_clip = slope.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat, crs="EPSG:4326")
    aspect_clip = aspect.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat, crs="EPSG:4326")
    return _to_analysis_convention(slope_clip), _to_analysis_convention(aspect_clip)

