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
    def from_bbox(
        cls,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        *,
        cell_size_deg: float = DEFAULT_CELL_SIZE_DEG,
        crs: str = DEFAULT_CRS,
    ) -> "GridSpec":
        """Construct a grid snapped down to the cell size from a bbox."""
        cell = float(cell_size_deg)
        origin_lat = math.floor(lat_min / cell) * cell
        origin_lon = math.floor(lon_min / cell) * cell
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
    """
    cell = grid.cell_size_deg

    i0 = int(math.floor((min_lat - grid.origin_lat) / cell))
    i1 = int(math.ceil((max_lat - grid.origin_lat) / cell))
    j0 = int(math.floor((min_lon - grid.origin_lon) / cell))
    j1 = int(math.ceil((max_lon - grid.origin_lon) / cell))

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

