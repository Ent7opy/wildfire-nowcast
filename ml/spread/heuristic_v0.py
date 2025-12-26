"""Baseline heuristic spread model (v0).

This model implements a simple rule-based spread using wind direction and speed
to produce a downwind-biased probability footprint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from datetime import timezone
from typing import Sequence

import numpy as np
import xarray as xr
from scipy.signal import fftconvolve

from ml.spread.contract import SpreadForecast, SpreadModel, SpreadModelInput

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class HeuristicSpreadV0Config:
    """Configuration for the v0 heuristic spread model."""
    # Base spread rate in km per hour (radial spread without wind)
    base_spread_km_h: float = 0.05
    
    # Wind influence: how many km of additional downwind spread per m/s per hour
    wind_influence_km_h_per_ms: float = 0.1
    
    # Anisotropy: ratio of downwind spread to crosswind spread
    # 1.0 means circular spread (ignoring wind displacement)
    # > 1.0 means elongated downwind
    wind_elongation_factor: float = 1.5
    
    # Activation threshold for fire heatmap
    fire_threshold: float = 0.0
    
    # Decay factor: how quickly probability drops off (distance-based)
    # Higher means sharper footprint edges
    distance_decay_km: float = 2.0

    # Cap kernel size to avoid memory issues for very long horizons/high winds.
    # Must be an odd integer >= 7.
    max_kernel_size: int = 201

    # Optional terrain bias (upslope)
    #
    # Terrain conventions (from `api.terrain.features_math` / `api.terrain.window`):
    # - slope: degrees [0, 90]
    # - aspect: degrees [0, 360), clockwise from North, direction of steepest DESCENT (downslope)
    #
    # When enabled, we bias spread in the UPSLOPE direction (aspect + 180°). This is
    # implemented using window-mean slope/aspect so it remains compatible with a single
    # convolution kernel per horizon.
    enable_slope_bias: bool = False
    slope_influence: float = 0.35  # unitless strength (0 disables); typical 0.1–0.6
    slope_reference_deg: float = 30.0  # slope at which bias is near full strength

class HeuristicSpreadModelV0(SpreadModel):
    """Simple rule-based spread model using wind bias."""
    
    def __init__(self, config: HeuristicSpreadV0Config | None = None):
        self.config = config or HeuristicSpreadV0Config()

    @staticmethod
    def _circular_mean_deg(values: np.ndarray) -> float:
        """Compute the circular mean for angles in degrees.

        This is appropriate for azimuth-like quantities where 0° ≡ 360°.
        Returns NaN if there are no finite samples or the mean direction is undefined
        (e.g., perfectly opposing directions cancel out).
        """
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")

        # Normalize into [0, 360) so wrap-around is handled consistently.
        arr = np.mod(arr, 360.0)
        theta = np.deg2rad(arr)

        sin_mean = float(np.mean(np.sin(theta)))
        cos_mean = float(np.mean(np.cos(theta)))
        if np.hypot(sin_mean, cos_mean) < 1e-12:
            return float("nan")

        mean_rad = np.arctan2(sin_mean, cos_mean)
        return float(np.mod(np.rad2deg(mean_rad), 360.0))

    def predict(self, inputs: SpreadModelInput) -> SpreadForecast:
        """Predict fire spread probability over the requested horizons."""
        LOGGER.info(
            "Running heuristic spread v0",
            extra={
                "horizons_hours": list(inputs.horizons_hours),
                "window_shape": (
                    int(inputs.window.lat.size),
                    int(inputs.window.lon.size),
                ),
            },
        )

        horizons = list(inputs.horizons_hours)
        for h in horizons:
            if h <= 0:
                raise ValueError(f"All horizons_hours must be positive; got {horizons!r}")
        
        # 1. Prepare fire source
        # (lat, lon) array where 1.0 is active fire
        fire_mask = (inputs.active_fires.heatmap > self.config.fire_threshold).astype(np.float32)
        fire_sum = float(fire_mask.sum())
        
        # 2. Get grid resolution in km (approximate)
        # 0.01 degree is ~1.11 km at equator.
        # We'll use a slightly more accurate mean lat approximation.
        mean_lat = float(inputs.window.lat.mean())
        km_per_lat_deg = 111.0
        km_per_lon_deg = float(111.0 * np.cos(np.radians(mean_lat)))
        
        dy_km = max(float(inputs.grid.cell_size_deg) * km_per_lat_deg, 1e-6)
        dx_km = max(float(inputs.grid.cell_size_deg) * km_per_lon_deg, 1e-6)
        
        # Optional terrain bias uses window-mean slope/aspect (keeps kernel global).
        slope_deg = None
        aspect_deg = None
        if self.config.enable_slope_bias:
            slope = getattr(inputs.terrain, "slope", None)
            aspect = getattr(inputs.terrain, "aspect", None)
            if slope is not None and aspect is not None:
                slope_arr = np.asarray(slope, dtype=float)
                aspect_arr = np.asarray(aspect, dtype=float)
                slope_deg = float(np.nanmean(slope_arr)) if np.size(slope_arr) else None
                aspect_deg = self._circular_mean_deg(aspect_arr) if np.size(aspect_arr) else None
            else:
                LOGGER.warning(
                    "enable_slope_bias=True but terrain slope/aspect unavailable; ignoring slope bias"
                )

        if fire_sum == 0.0:
            forecast_grids = [np.zeros_like(fire_mask, dtype=np.float32) for _ in horizons]
            return self._package_forecast(inputs, horizons, forecast_grids)
        
        forecast_grids = []
        for horizon_h in horizons:
            # 3. Extract wind for this horizon
            # Simple approach: mean wind over the window at the nearest time
            target_time = inputs.forecast_reference_time + timedelta(hours=horizon_h)
            
            # Select nearest time in weather_cube
            if "time" in inputs.weather_cube.dims:
                weather_at_t = inputs.weather_cube.sel(time=self._as_datetime64_utc_naive(target_time), method="nearest")
            else:
                weather_at_t = inputs.weather_cube

            missing = [v for v in ("u10", "v10") if v not in weather_at_t.data_vars]
            if missing:
                raise ValueError(
                    "weather_cube missing required variable(s) for heuristic_v0: "
                    + ", ".join(missing)
                )

            u10 = float(weather_at_t["u10"].mean())
            v10 = float(weather_at_t["v10"].mean())
            
            # 4. Generate kernel for this horizon/wind
            kernel = self._generate_kernel(
                horizon_h,
                u10,
                v10,
                dy_km,
                dx_km,
                slope_deg=slope_deg,
                aspect_deg=aspect_deg,
            )
            
            # 5. Convolve
            # Use mode='same' to keep output size matching input
            # fftconvolve is generally faster for these kernel sizes
            prob_grid = fftconvolve(fire_mask, kernel, mode="same").astype(np.float32, copy=False)
                
            # 6. Apply masks and normalize
            # Ensure probability is in [0, 1]
            max_val = float(np.max(prob_grid)) if prob_grid.size else 0.0
            if max_val > 0:
                prob_grid = prob_grid / max_val

            # FFT convolution can introduce tiny negative values; clamp to contract range.
            prob_grid = np.clip(prob_grid, 0.0, 1.0, out=prob_grid)
            
            # Mask out invalid terrain if present
            if inputs.terrain.valid_data_mask is not None:
                prob_grid = prob_grid * inputs.terrain.valid_data_mask
            
            if inputs.terrain.aoi_mask is not None:
                prob_grid = prob_grid * inputs.terrain.aoi_mask
                
            forecast_grids.append(prob_grid)
            
        return self._package_forecast(inputs, horizons, forecast_grids)

    @staticmethod
    def _as_datetime64_utc_naive(dt) -> np.datetime64:
        # Weather ingest normalizes to tz-naive UTC datetime64[ns].
        # Avoid tz-aware datetime selection issues in xarray/pandas comparisons.
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(dt, "ns")

    @staticmethod
    def _package_forecast(
        inputs: SpreadModelInput,
        horizons: Sequence[int],
        forecast_grids: Sequence[np.ndarray],
    ) -> SpreadForecast:
        # 7. Package results
        times = [inputs.forecast_reference_time + timedelta(hours=h) for h in horizons]

        da = xr.DataArray(
            np.stack(forecast_grids).astype(np.float32, copy=False),
            coords={
                "time": times,
                "lat": inputs.window.lat,
                "lon": inputs.window.lon,
                "lead_time_hours": ("time", list(horizons)),
            },
            dims=("time", "lat", "lon"),
            name="spread_probability",
        )

        return SpreadForecast(
            probabilities=da,
            forecast_reference_time=inputs.forecast_reference_time,
            horizons_hours=horizons,
        )

    def _generate_kernel(
        self, 
        horizon_h: float, 
        u_ms: float, 
        v_ms: float, 
        dy_km: float, 
        dx_km: float,
        *,
        slope_deg: float | None = None,
        aspect_deg: float | None = None,
    ) -> np.ndarray:
        """Generate an anisotropic kernel centered at origin with downwind bias."""
        
        # Base spread distance based on horizon
        base_dist = self.config.base_spread_km_h * horizon_h
        # Wind speed magnitude
        wind_speed = np.sqrt(u_ms**2 + v_ms**2)
        
        # Kernel size: cover the spread distance
        # We'll use 4x the max spread distance in pixels to capture the decay
        max_dist_km = base_dist + (wind_speed * horizon_h * self.config.wind_influence_km_h_per_ms)
        max_dist_px = max(max_dist_km / dx_km, max_dist_km / dy_km)
        k_size = int(max(7, 2 * (max_dist_px * 3) + 1))
        if k_size % 2 == 0:
            k_size += 1
        # Cap k_size to avoid memory issues for very long horizons/high winds in large AOIs
        max_k = int(self.config.max_kernel_size)
        if max_k < 7:
            raise ValueError(f"max_kernel_size must be >= 7; got {max_k}")
        if max_k % 2 == 0:
            raise ValueError(f"max_kernel_size must be odd; got {max_k}")
        k_size = min(k_size, max_k)
            
        half = k_size // 2
        y, x = np.ogrid[-half:half+1, -half:half+1]
        
        y_km = y * dy_km
        x_km = x * dx_km
        
        dist = np.sqrt(x_km**2 + y_km**2)
        
        eff_dist = dist

        if wind_speed > 1e-6:
            # Angle of each pixel from origin
            angles = np.arctan2(y_km, x_km)
            # Angle of wind
            wind_angle = np.arctan2(v_ms, u_ms)
            
            # Difference from wind direction
            # cos(diff) is 1.0 downwind, -1.0 upwind
            cos_diff = np.cos(angles - wind_angle)
            
            # Effective distance: shorter downwind, longer upwind
            # We use the elongation factor to control the asymmetry
            # wind_bias factor in [0, 1)
            # 0.5 means downwind is 3x easier than upwind (0.5 vs 1.5)
            # We'll map elongation_factor to a bias
            bias = (self.config.wind_elongation_factor - 1) / (self.config.wind_elongation_factor + 1)
            bias = min(max(bias, 0.0), 0.9) # cap it
            
            eff_dist = eff_dist * (1.0 - bias * cos_diff)
            
            # Additionally, stretch the whole thing proportional to wind speed
            wind_scale = 1.0 + (wind_speed * self.config.wind_influence_km_h_per_ms / (self.config.base_spread_km_h + 1e-6))
            eff_dist = eff_dist / wind_scale

        # Optional upslope bias (terrain-driven). Uses window-mean slope/aspect.
        if self.config.enable_slope_bias and slope_deg is not None and aspect_deg is not None:
            if np.isfinite(slope_deg) and np.isfinite(aspect_deg) and slope_deg > 0:
                # Aspect is downslope azimuth; upslope is opposite.
                upslope_deg = (aspect_deg + 180.0) % 360.0
                upslope_angle = np.deg2rad(90.0 - upslope_deg)  # convert azimuth->math angle

                angles = np.arctan2(y_km, x_km)
                cos_up = np.cos(angles - upslope_angle)  # 1.0 upslope, -1.0 downslope

                # Strength scales with slope: saturate around slope_reference_deg.
                ref = max(float(self.config.slope_reference_deg), 1e-6)
                strength = float(self.config.slope_influence) * min(max(slope_deg / ref, 0.0), 1.0)
                strength = min(max(strength, 0.0), 0.9)

                # Make upslope "easier" (shorter effective distance), downslope harder.
                eff_dist = eff_dist * (1.0 - strength * cos_up)
            
        # Exponential decay probability
        # We include the decay parameter from config
        kernel = np.exp(-eff_dist / (base_dist + self.config.distance_decay_km))
        
        return kernel
