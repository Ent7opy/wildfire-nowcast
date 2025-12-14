"""Windowed terrain feature loading (slope/aspect and optional DEM).

Key conventions (must match `api.core.grid`):
- CRS: EPSG:4326 (lon/lat).
- Indices: half-open windows `(i0:i1, j0:j1)`.
- Returned arrays: dims `(lat, lon)` with **lat increasing** and **lon increasing**.

Contributor docs: see `docs/terrain_grid.md`.

Important: GeoTIFFs are typically stored north-up (row 0 is north). We read GeoTIFF
windows using rasterio and then flip rows once to match the analysis convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from api.core.grid import GridSpec, GridWindow, get_grid_window_for_bbox
from api.terrain import features_repo, repo
from api.terrain.validate import validate_terrain_stack

if TYPE_CHECKING:  # pragma: no cover
    from shapely.geometry.base import BaseGeometry

BBox: TypeAlias = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


@dataclass(frozen=True, slots=True)
class TerrainWindow:
    window: GridWindow
    slope: np.ndarray  # (lat, lon) with lat/lon ascending
    aspect: np.ndarray  # (lat, lon) with lat/lon ascending
    elevation: np.ndarray | None = None  # (lat, lon), optional
    valid_data_mask: np.ndarray | None = None  # (lat, lon) True where data is valid
    aoi_mask: np.ndarray | None = None  # (lat, lon) True where inside AOI polygon
    generated_at: datetime = field(default_factory=datetime.utcnow)


def _grid_spec_from_features_metadata(
    metadata: features_repo.TerrainFeaturesMetadata,
) -> GridSpec:
    return GridSpec(
        crs=f"EPSG:{metadata.crs_epsg}",
        cell_size_deg=float(metadata.cell_size_deg),
        origin_lat=float(metadata.origin_lat),
        origin_lon=float(metadata.origin_lon),
        n_lat=int(metadata.grid_n_lat),
        n_lon=int(metadata.grid_n_lon),
    )


def _read_window_as_analysis_array(
    raster_path: Path,
    grid: GridSpec,
    window: GridWindow,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a raster subset by grid window and normalize to analysis convention.

    Returns `(data, valid_mask)` with shape `(n_lat, n_lon)` of the window.
    - `data`: float array, nodata filled with NaN if nodata exists.
    - `valid_mask`: boolean array True where data is not masked.
    """
    height = window.i1 - window.i0
    width = window.j1 - window.j0
    if height <= 0 or width <= 0:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, empty.astype(bool)

    # Convert analysis indices (south→north) into raster rows (north→south).
    # Raster row 0 is north; analysis i=0 is south.
    row_off = grid.n_lat - window.i1
    col_off = window.j0
    win = Window(col_off=col_off, row_off=row_off, width=width, height=height)

    with rasterio.open(raster_path) as src:
        band = src.read(1, window=win, masked=True)

    # Fill nodata with NaN (mirrors rioxarray.open_rasterio(masked=True)).
    if isinstance(band, np.ma.MaskedArray):
        # NOTE: `band.mask` can be `np.ma.nomask` (a scalar False) when there is no
        # nodata/mask information. `np.ma.getmaskarray(...)` always returns an array
        # shaped like the data, which keeps downstream operations (e.g. flipud) safe.
        valid = ~np.ma.getmaskarray(band)
        data = np.asarray(band.filled(np.nan), dtype=np.float32)
    else:  # pragma: no cover (rasterio returns masked arrays for masked=True)
        data = np.asarray(band, dtype=np.float32)
        valid = np.ones_like(data, dtype=bool)

    # Flip to analysis order: lat increasing (south→north).
    data = np.flipud(data)
    valid = np.flipud(valid)
    return data, valid


def _load_terrain_window_from_features_md(
    region_name: str,
    features_md: features_repo.TerrainFeaturesMetadata,
    bbox: BBox,
    *,
    include_dem: bool,
    clip: bool,
) -> tuple[TerrainWindow, GridSpec, Path]:
    """Internal helper to ensure a single metadata/grid is used end-to-end.

    Returns `(terrain_window, grid, slope_path)` where `grid` is the grid used to compute
    `terrain_window.window` and read the raster data.
    """
    grid = _grid_spec_from_features_metadata(features_md)
    win = get_grid_window_for_bbox(grid, bbox, clip=clip)

    slope_path = Path(features_md.slope_path)
    aspect_path = Path(features_md.aspect_path)
    if not slope_path.exists():
        raise FileNotFoundError(f"Slope raster not found at {slope_path}")
    if not aspect_path.exists():
        raise FileNotFoundError(f"Aspect raster not found at {aspect_path}")

    # Fail fast if slope/aspect (and optional DEM) are not exactly aligned to the grid.
    dem_path: Path | None = None
    if include_dem:
        dem_md = repo.get_latest_dem_metadata_for_region(region_name)
        if dem_md is None:
            raise ValueError(f"No DEM metadata found for region '{region_name}'.")
        dem_path = Path(dem_md.raster_path)
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM raster not found at {dem_path}")
    validate_terrain_stack(dem_path, slope_path, aspect_path, grid, strict=True)

    slope, slope_valid = _read_window_as_analysis_array(slope_path, grid, win)
    aspect, aspect_valid = _read_window_as_analysis_array(aspect_path, grid, win)
    valid = slope_valid & aspect_valid

    elevation = None
    if include_dem:
        assert dem_path is not None  # for type checkers
        elevation, dem_valid = _read_window_as_analysis_array(dem_path, grid, win)
        valid = valid & dem_valid

    # For empty windows we keep masks as empty arrays (not None) to preserve shapes.
    valid_mask = valid.astype(bool) if valid.size else valid
    return (
        TerrainWindow(
            window=win,
            slope=slope,
            aspect=aspect,
            elevation=elevation,
            valid_data_mask=valid_mask,
        ),
        grid,
        slope_path,
    )


def load_terrain_window(
    region_name: str,
    bbox: BBox,
    *,
    include_dem: bool = False,
    clip: bool = True,
) -> TerrainWindow:
    """Load slope/aspect (and optionally DEM) cut to a grid-aligned window for bbox."""
    features_md = features_repo.get_latest_terrain_features_metadata_for_region(region_name)
    if features_md is None:
        raise ValueError(f"No terrain features metadata found for region '{region_name}'.")
    tw, _grid, _slope_path = _load_terrain_window_from_features_md(
        region_name,
        features_md,
        bbox,
        include_dem=include_dem,
        clip=clip,
    )
    return tw


def load_terrain_for_aoi(
    region_name: str,
    aoi: "BaseGeometry | BBox",
    *,
    include_dem: bool = False,
    return_mask: bool = True,
    clip: bool = True,
) -> TerrainWindow:
    """Load terrain for a bbox or polygon AOI.

    - If AOI is a bbox: identical to `load_terrain_window(...)`.
    - If AOI is a polygon: loads the bbox window and optionally returns an AOI mask
      (True where pixel center is inside polygon).
    """
    if isinstance(aoi, tuple) and len(aoi) == 4:
        return load_terrain_window(region_name, aoi, include_dem=include_dem, clip=clip)

    # Shapely geometry path (polygon or any geometry with bounds).
    features_md = features_repo.get_latest_terrain_features_metadata_for_region(region_name)
    if features_md is None:
        raise ValueError(f"No terrain features metadata found for region '{region_name}'.")

    bounds = aoi.bounds  # (minx, miny, maxx, maxy)
    bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    tw, grid, slope_path = _load_terrain_window_from_features_md(
        region_name,
        features_md,
        bbox,
        include_dem=include_dem,
        clip=clip,
    )
    if not return_mask:
        return tw

    height = tw.window.i1 - tw.window.i0
    width = tw.window.j1 - tw.window.j0
    if height <= 0 or width <= 0:
        return TerrainWindow(
            window=tw.window,
            slope=tw.slope,
            aspect=tw.aspect,
            elevation=tw.elevation,
            valid_data_mask=tw.valid_data_mask,
            aoi_mask=np.empty((0, 0), dtype=bool),
        )

    # Rasterize polygon onto the *raster* window grid (north-up), then flip to analysis.
    row_off = grid.n_lat - tw.window.i1
    col_off = tw.window.j0
    rio_win = Window(col_off=col_off, row_off=row_off, width=width, height=height)

    with rasterio.open(slope_path) as src:
        transform = src.window_transform(rio_win)

    # Burn 1 where pixel center is inside the polygon.
    mask_raster = rasterize(
        [(aoi.__geo_interface__, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    aoi_mask = np.flipud(mask_raster.astype(bool))

    return TerrainWindow(
        window=tw.window,
        slope=tw.slope,
        aspect=tw.aspect,
        elevation=tw.elevation,
        valid_data_mask=tw.valid_data_mask,
        aoi_mask=aoi_mask,
    )

