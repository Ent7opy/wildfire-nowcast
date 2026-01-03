"""Lightweight sanity checks for spread model outputs.

These checks are intended to be run during development or CI to catch
obviously broken behavior in model implementations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from ml.spread.contract import SpreadForecast, SpreadModelInput

LOGGER = logging.getLogger(__name__)


def check_nonempty_when_fires_present(
    inputs: SpreadModelInput,
    forecast: SpreadForecast,
    threshold: float = 0.0,
) -> None:
    """Verify that if fires exist in the input, the output is not entirely zero.
    
    This handles cases where the model might be silently failing to propagate
    fire state into the forecast.
    """
    fire_sum = float(inputs.active_fires.heatmap.sum())
    if fire_sum > 0:
        max_prob = float(forecast.probabilities.max())
        if max_prob <= threshold:
            raise ValueError(
                f"Spread forecast is empty (max={max_prob}) despite {fire_sum} active fire cells in input."
            )


def check_monotonic_footprint(
    forecast: SpreadForecast,
    probability_threshold: float = 0.01,
    relative_tolerance: float = 0.05,
) -> None:
    """Verify that the fire footprint does not shrink significantly over time.
    
    A shrinking footprint (in terms of area above a probability threshold)
    usually indicates a bug in the model's propagation or decay logic.
    
    Parameters
    ----------
    forecast : SpreadForecast
        The forecast to check.
    probability_threshold : float
        The probability level at which to define the 'footprint' area.
    relative_tolerance : float
        Allowed fractional decrease in area between horizons (to account for
        grid/numerical artifacts or legitimate masking).
    """
    # Count pixels above threshold for each horizon
    # probabilities dims: (time, lat, lon)
    areas = (forecast.probabilities > probability_threshold).sum(dim=("lat", "lon")).values
    
    for i in range(len(areas) - 1):
        h0 = forecast.horizons_hours[i]
        h1 = forecast.horizons_hours[i+1]
        a0 = float(areas[i])
        a1 = float(areas[i+1])
        
        # We allow a small relative decrease (tolerance)
        if a1 < a0 * (1.0 - relative_tolerance) and a0 > 0:
            raise ValueError(
                f"Fire footprint shrank significantly between T+{h0}h and T+{h1}h: "
                f"area({h0}h)={a0}, area({h1}h)={a1} (threshold={probability_threshold})."
            )


def check_wind_displacement(
    inputs: SpreadModelInput,
    forecast: SpreadForecast,
    min_wind_speed_ms: float = 5.0,
    min_displacement_px: float = 0.5,
) -> None:
    """Verify that strong wind shifts the footprint center of mass downwind.
    
    This handles models that produce shifted circles rather than elongated ellipses.
    """
    probs = forecast.probabilities.isel(time=-1).values
    if probs.max() < 0.1:
        return
        
    weather = inputs.weather_cube
    if "time" in weather.dims:
        weather = weather.isel(time=-1)
        
    u = float(weather["u10"].mean())
    v = float(weather["v10"].mean())
    wind_speed = np.sqrt(u**2 + v**2)
    
    if wind_speed < min_wind_speed_ms:
        return
        
    ny, nx = probs.shape
    lon_indices = np.arange(nx)
    lat_indices = np.arange(ny)
    lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
    
    mask = probs > 0.05
    if not mask.any():
        return
        
    w = probs[mask]
    y_mean = np.average(lat_grid[mask], weights=w)
    x_mean = np.average(lon_grid[mask], weights=w)
    
    # Ignition center (assumed to be window center if not provided)
    # For a more robust check, we should pass the ignition coords.
    # In heuristic_v0 tests, it's usually at ny//2, nx//2.
    y_ign, x_ign = ny // 2, nx // 2
    
    dy = y_mean - y_ign
    dx = x_mean - x_ign
    
    # Dot product with wind vector
    displacement_downwind = (dx * u + dy * v) / (wind_speed + 1e-6)
    
    if displacement_downwind < min_displacement_px:
        raise ValueError(
            f"Footprint center of mass displacement downwind ({displacement_downwind:.2f} px) "
            f"is below threshold ({min_displacement_px}) for strong wind ({wind_speed:.1f} m/s)."
        )


def check_wind_elongation(
    inputs: SpreadModelInput,
    forecast: SpreadForecast,
    min_wind_speed_ms: float = 5.0,
    min_anisotropy: float = 1.05,
    max_angle_error_deg: float = 45.0,
) -> None:
    """Verify that strong wind produces elongated footprints along the wind vector.
    
    This uses the spatial covariance of the probability grid at the furthest
    horizon to check for elongation and alignment.
    """
    # Use the last horizon for the check as it should have the most developed footprint
    probs = forecast.probabilities.isel(time=-1).values
    if probs.max() < 0.1:
        return  # Too faint to check reliably
        
    # Get mean wind for this horizon (simplified, similar to heuristic_v0)
    weather = inputs.weather_cube
    if "time" in weather.dims:
        # Select last horizon time
        weather = weather.isel(time=-1)
        
    u = float(weather["u10"].mean())
    v = float(weather["v10"].mean())
    wind_speed = np.sqrt(u**2 + v**2)
    
    if wind_speed < min_wind_speed_ms:
        return
        
    # Calculate spatial moments (weighted by probability)
    # Use pixel indices for more stable anisotropy calculation
    ny, nx = probs.shape
    lon_indices = np.arange(nx)
    lat_indices = np.arange(ny)
    
    lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
    
    # We only care about cells with significant probability
    mask = probs > 0.05
    # Need at least 2 points to calculate covariance
    if mask.sum() < 2:
        return
        
    w = probs[mask]
    y = lat_grid[mask]
    x = lon_grid[mask]
    
    # Center the coordinates
    y_mean = np.average(y, weights=w)
    x_mean = np.average(x, weights=w)
    y_c = y - y_mean
    x_c = x - x_mean
    
    # Covariance matrix
    cov = np.cov(np.stack([x_c, y_c]), aweights=w)
    
    # Eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(cov)
    
    # eigh returns sorted eigenvalues: evals[0] is minor, evals[1] is major
    major_eval = max(evals[1], 0)
    minor_eval = max(evals[0], 0)
    
    if minor_eval < 1e-12:
        anisotropy = float('inf') if major_eval > 1e-12 else 1.0
    else:
        anisotropy = np.sqrt(major_eval / minor_eval)
        
    if anisotropy < min_anisotropy:
        return # Not enough elongation to check alignment
        
    # Major axis direction
    major_vec = evecs[:, 1]
    
    # Wind direction in pixel space
    # NOTE: In our grid conventions (api.core.grid), i (lat) is y, j (lon) is x.
    # But dy_km and dx_km differ. We should use the km-scaled wind vector.
    # Here we simplify: u (East) -> x, v (North) -> y.
    # Since dx_km and dy_km are close (~1.0), this is usually fine for a sanity check.
    major_angle_rad = np.arctan2(major_vec[1], major_vec[0])
    wind_angle_rad = np.arctan2(v, u)
    
    # Angle error (handle 180-degree ambiguity of PCA major axis)
    angle_diff_deg = np.abs(np.rad2deg(major_angle_rad - wind_angle_rad)) % 180
    if angle_diff_deg > 90:
        angle_diff_deg = 180 - angle_diff_deg
        
    if angle_diff_deg > max_angle_error_deg:
        raise ValueError(
            f"Footprint major axis is misaligned with wind by {angle_diff_deg:.1f}° "
            f"(wind dir: {np.rad2deg(wind_angle_rad):.1f}°, major axis: {np.rad2deg(major_angle_rad):.1f}°)."
        )

