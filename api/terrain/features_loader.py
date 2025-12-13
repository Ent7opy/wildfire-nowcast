"""Helpers to read slope/aspect rasters for downstream consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import rioxarray  # type: ignore
from xarray import DataArray

from api.terrain.features_repo import (
    TerrainFeaturesMetadata,
    get_latest_terrain_features_metadata_for_region,
)


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
    return _to_analysis_convention(da)


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

    slope = _load_feature_raster(slope_path)
    aspect = _load_feature_raster(aspect_path)

    min_lon, min_lat, max_lon, max_lat = bbox
    slope_clip = slope.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat, crs="EPSG:4326")
    aspect_clip = aspect.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat, crs="EPSG:4326")
    return slope_clip, aspect_clip

