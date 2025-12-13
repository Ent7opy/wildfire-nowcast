"""Deterministic slope/aspect derivation on the canonical grid.

Conventions
- **Slope**: degrees, range [0, 90]
- **Aspect**: degrees, range [0, 360), clockwise from North (0/360=N, 90=E)
- **Aspect direction**: **downslope** (direction of steepest descent)

Important
Input rasters are stored as north-up GeoTIFFs (row index increases southward).
This module expects:
- `z` shaped (height, width) matching raster row/col order
- `lat_centers_deg` shaped (height,) giving the latitude center for each raster row
"""

from __future__ import annotations

import numpy as np

METERS_PER_DEG_AT_EQUATOR = 111_320.0


def compute_slope_aspect(
    z: np.ndarray,
    *,
    cell_size_deg: float,
    lat_centers_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope and aspect from an elevation raster.

    Parameters
    - **z**: 2D array of elevations in meters. Use NaN for nodata.
    - **cell_size_deg**: pixel size in degrees (e.g., 0.01).
    - **lat_centers_deg**: 1D array (len=height) of row-center latitudes in degrees.

    Returns
    - **slope_deg**: float64 array, NaN where invalid.
    - **aspect_deg**: float64 array, NaN where invalid or slope ~ 0.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("z must be a 2D array (height, width)")
    height, _width = z.shape

    lat_centers_deg = np.asarray(lat_centers_deg, dtype=float)
    if lat_centers_deg.shape != (height,):
        raise ValueError("lat_centers_deg must have shape (height,)")

    cell = float(cell_size_deg)
    if not np.isfinite(cell) or cell <= 0:
        raise ValueError("cell_size_deg must be a positive finite number")

    # Convert cell sizes to approximate meters.
    dy_m = METERS_PER_DEG_AT_EQUATOR * cell
    dx_m = METERS_PER_DEG_AT_EQUATOR * np.cos(np.deg2rad(lat_centers_deg)) * cell
    dx_m = np.where(dx_m == 0, np.nan, dx_m)  # guard poles

    # Pixel gradients in index space (row, col). Row index increases southward.
    dz_di, dz_dj = np.gradient(z)

    # Convert to derivatives w.r.t. east (x) and north (y).
    dz_dx = dz_dj / dx_m[:, None]
    dz_dy_south = dz_di / dy_m
    dz_dy_north = -dz_dy_south

    # Slope: arctan of gradient magnitude.
    slope_rad = np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy_north * dz_dy_north))
    slope_deg = np.rad2deg(slope_rad)

    # Aspect: azimuth of steepest descent vector (-∇z), clockwise from north.
    vx = -dz_dx
    vy = -dz_dy_north
    aspect_rad = np.arctan2(vx, vy)
    aspect_deg = (np.rad2deg(aspect_rad) + 360.0) % 360.0

    # Mask invalid / flat cells.
    invalid = ~np.isfinite(slope_deg)
    flat = np.isfinite(slope_deg) & (slope_deg <= 1e-9)
    aspect_deg = np.where(invalid | flat, np.nan, aspect_deg)

    # Clip tiny numerical spillovers.
    slope_deg = np.clip(slope_deg, 0.0, 90.0)
    return slope_deg, aspect_deg

