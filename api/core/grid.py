"""Canonical analysis grid specification helpers.

Conventions (analysis order)
- **CRS**: EPSG:4326 (WGS84).
- **Resolution**: fixed `0.01°` in both latitude and longitude (by default).
- **Indexing**: array indices are `(i, j) = (lat_index, lon_index)`.
- **Coordinate order**: `lat` and `lon` are 1D arrays that are *monotonic increasing*
  (south → north, west → east).
- **Origin fields**: `origin_lat` and `origin_lon` represent the *southern*/*western*
  cell **edges**. Coordinate outputs (`grid_coords`, `index_to_latlon`,
  `window_coords`) return **cell centers**.

Contributor docs
- See `docs/grid_choice.md` (why EPSG:4326 @ 0.01°) and `docs/terrain_grid.md`
  (full grid + terrain alignment contract, pitfalls, usage recipes).

Note on GeoTIFFs
GeoTIFF rasters are commonly stored “north-up” where row 0 corresponds to the
northernmost pixels (i.e., latitude decreases with increasing row index).
Downstream loaders (e.g. DEM) must normalize to the analysis convention before
returning arrays to ML/analysis consumers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

DEFAULT_CRS = "EPSG:4326"
DEFAULT_CELL_SIZE_DEG = 0.01


# Epsilon for floating-point precision handling in grid calculations
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Grid definition for rasters on the analysis grid."""

    crs: str = DEFAULT_CRS
    cell_size_deg: float = DEFAULT_CELL_SIZE_DEG
    origin_lat: float = 0.0  # lower (southern) edge
    origin_lon: float = 0.0  # left (western) edge
    n_lat: int = 0
    n_lon: int = 0

    @classmethod
    def from_bbox_tuple(
        cls,
        bbox: tuple[float, float, float, float],
        *,
        cell_size_deg: float = DEFAULT_CELL_SIZE_DEG,
        crs: str = DEFAULT_CRS,
    ) -> "GridSpec":
        """Construct a grid from a bbox tuple (min_lon, min_lat, max_lon, max_lat).

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            cell_size_deg: Grid cell size in degrees
            crs: Coordinate reference system

        Returns:
            GridSpec instance
        """
        if len(bbox) != 4:
            raise ValueError(f"bbox tuple must have 4 elements, got {len(bbox)}")
        min_lon, min_lat, max_lon, max_lat = bbox
        return cls.from_bounds(
            lat_min=min_lat,
            lat_max=max_lat,
            lon_min=min_lon,
            lon_max=max_lon,
            cell_size_deg=cell_size_deg,
            crs=crs,
        )

    @classmethod
    def from_bounds(
        cls,
        *,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        cell_size_deg: float = DEFAULT_CELL_SIZE_DEG,
        crs: str = DEFAULT_CRS,
    ) -> "GridSpec":
        """Construct a grid from individual bounds parameters.

        Args:
            lat_min: Minimum latitude (southern bound)
            lat_max: Maximum latitude (northern bound)
            lon_min: Minimum longitude (western bound)
            lon_max: Maximum longitude (eastern bound)
            cell_size_deg: Grid cell size in degrees
            crs: Coordinate reference system

        Returns:
            GridSpec instance
        """
        cell = float(cell_size_deg)
        # Add epsilon before floor to handle floating-point precision edge cases
        origin_lat = math.floor((lat_min / cell) + _EPSILON) * cell
        origin_lon = math.floor((lon_min / cell) + _EPSILON) * cell
        n_lat = int(math.ceil((lat_max - origin_lat) / cell))
        n_lon = int(math.ceil((lon_max - origin_lon) / cell))
        return cls(
            crs=crs,
            cell_size_deg=cell,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            n_lat=n_lat,
            n_lon=n_lon,
        )

    @classmethod
    def from_bbox(
        cls,
        lat_min: float | tuple[float, float, float, float],
        lat_max: float | None = None,
        lon_min: float | None = None,
        lon_max: float | None = None,
        *,
        cell_size_deg: float = DEFAULT_CELL_SIZE_DEG,
        crs: str = DEFAULT_CRS,
    ) -> "GridSpec":
        """Construct a grid snapped down to the cell size from a bbox.

        .. deprecated::
            Use :meth:`from_bbox_tuple` or :meth:`from_bounds` for clearer semantics.

        Accepts either:
        - Four separate arguments: from_bbox(lat_min, lat_max, lon_min, lon_max)
        - A single bbox tuple: from_bbox((min_lon, min_lat, max_lon, max_lat))
        """
        if isinstance(lat_min, (tuple, list)):
            if lat_max is not None or lon_min is not None or lon_max is not None:
                raise ValueError("Cannot mix bbox tuple and individual arguments")
            return cls.from_bbox_tuple(
                lat_min,
                cell_size_deg=cell_size_deg,
                crs=crs,
            )
        else:
            if lat_max is None or lon_min is None or lon_max is None:
                raise ValueError("Must provide all four bbox coordinates")
            return cls.from_bounds(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                cell_size_deg=cell_size_deg,
                crs=crs,
            )


@dataclass(frozen=True, slots=True)
class GridWindow:
    """A grid-aligned half-open window with 1D cell-center coordinates.

    Conventions:
    - Indices are half-open (Python slicing): `(i0:i1, j0:j1)`.
    - `i` increases south → north (latitude increases).
    - `j` increases west → east (longitude increases).
    - `lat`/`lon` are **cell centers** and are monotonic increasing.
    """

    i0: int
    i1: int
    j0: int
    j1: int
    lat: np.ndarray
    lon: np.ndarray


def grid_bounds(grid: GridSpec) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) bounds for a grid."""
    max_lat = grid.origin_lat + grid.n_lat * grid.cell_size_deg
    max_lon = grid.origin_lon + grid.n_lon * grid.cell_size_deg
    return (grid.origin_lon, grid.origin_lat, max_lon, max_lat)


def grid_coords(grid: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Return 1D arrays of lat/lon cell centers for the grid."""
    cell = grid.cell_size_deg
    lat = grid.origin_lat + (np.arange(grid.n_lat) + 0.5) * cell
    lon = grid.origin_lon + (np.arange(grid.n_lon) + 0.5) * cell
    return lat, lon


def latlon_to_index(grid: GridSpec, lat: np.ndarray | float, lon: np.ndarray | float) -> Tuple[np.ndarray, np.ndarray]:
    """Map lat/lon coordinates to integer grid indices (i=lat, j=lon)."""
    cell = grid.cell_size_deg
    i = np.floor((np.asarray(lat) - grid.origin_lat) / cell).astype(int)
    j = np.floor((np.asarray(lon) - grid.origin_lon) / cell).astype(int)
    return i, j


def index_to_latlon(
    grid: GridSpec, i: np.ndarray | int, j: np.ndarray | int
) -> Tuple[np.ndarray, np.ndarray]:
    """Map integer grid indices to lat/lon cell-center coordinates.

    Parameters
    - **i**: latitude index/indices (0-based, south → north)
    - **j**: longitude index/indices (0-based, west → east)

    Returns
    - **lat**: latitude center(s)
    - **lon**: longitude center(s)
    """
    cell = grid.cell_size_deg
    lat = grid.origin_lat + (np.asarray(i) + 0.5) * cell
    lon = grid.origin_lon + (np.asarray(j) + 0.5) * cell
    return lat, lon


def bbox_to_window(
    grid: GridSpec,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    *,
    clip: bool = True,
) -> Tuple[int, int, int, int]:
    """Convert a lon/lat bbox to a half-open index window on the grid.

    The returned window is **half-open** (Python slicing style):
    - `i0:i1` spans latitude indices (south → north)
    - `j0:j1` spans longitude indices (west → east)

    Edge behavior
    - `i0`/`j0` use `floor` (snap down) so a bbox that starts inside a cell includes it.
    - `i1`/`j1` use `ceil` (snap up) so a bbox that ends inside a cell includes it.
    - If `clip=True`, indices are clamped to the grid extent `[0, n_lat]` / `[0, n_lon]`.
      If the bbox lies completely outside, the result may be empty (e.g. `i0 == i1`).

    Note:
        Uses epsilon correction to handle floating-point precision edge cases where
        coordinates at exact cell boundaries could be misassigned due to rounding errors.
    """
    cell = grid.cell_size_deg

    # Add epsilon before floor/ceil to handle floating-point precision edge cases
    # This prevents coordinates at exact cell boundaries from being misassigned
    i0 = int(math.floor((min_lat - grid.origin_lat) / cell + _EPSILON))
    i1 = int(math.ceil((max_lat - grid.origin_lat) / cell - _EPSILON))
    j0 = int(math.floor((min_lon - grid.origin_lon) / cell + _EPSILON))
    j1 = int(math.ceil((max_lon - grid.origin_lon) / cell - _EPSILON))

    if clip:
        i0 = max(0, min(grid.n_lat, i0))
        i1 = max(0, min(grid.n_lat, i1))
        j0 = max(0, min(grid.n_lon, j0))
        j1 = max(0, min(grid.n_lon, j1))

    return i0, i1, j0, j1


def window_coords(grid: GridSpec, i0: int, i1: int, j0: int, j1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return 1D arrays of lat/lon cell centers for a `(i0:i1, j0:j1)` window."""
    cell = grid.cell_size_deg
    lat = grid.origin_lat + (np.arange(i0, i1) + 0.5) * cell
    lon = grid.origin_lon + (np.arange(j0, j1) + 0.5) * cell
    return lat, lon


def get_grid_window_for_bbox(
    grid: GridSpec,
    bbox: tuple[float, float, float, float],
    *,
    clip: bool = True,
) -> GridWindow:
    """Compute a grid-aligned window for a lon/lat bbox.

    Parameters
    - **bbox**: `(min_lon, min_lat, max_lon, max_lat)` in EPSG:4326.
    - **clip**: if True, clamp indices to `[0..n_lat]` / `[0..n_lon]`.

    Returns
    - A `GridWindow` with half-open indices and cell-center `lat`/`lon` arrays.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    i0, i1, j0, j1 = bbox_to_window(
        grid,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        clip=clip,
    )
    lat, lon = window_coords(grid, i0, i1, j0, j1)
    return GridWindow(i0=i0, i1=i1, j0=j0, j1=j1, lat=lat, lon=lon)

