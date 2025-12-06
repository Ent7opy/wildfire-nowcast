"""Canonical analysis grid specification helpers."""

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

