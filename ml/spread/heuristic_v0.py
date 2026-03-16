"""Baseline heuristic spread model (v0).

This model implements a simple rule-based spread using wind direction and speed
to produce a downwind-biased probability footprint.

See `docs/spread_model_design.md` for a higher-level description, assumptions, and limitations.
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

    # Hard cap on spread expansion per 24h to keep fallback forecasts operationally sane.
    max_daily_spread_km: float = 20.0

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

    # Iterative spread step size in hours.  The model propagates fire state in steps of
    # this size and snapshots at each requested horizon.  Smaller steps are more
    # physically accurate; 2h balances accuracy and compute.
    iterative_step_hours: int = 2

    # Moisture suppression: scale spread rate by dead fuel moisture content (DFMC).
    # When enabled, DFMC from the weather cube modulates the spread rate:
    #   DFMC < dfmc_min_pct  → rate multiplied by dfmc_max_factor (rapid spread)
    #   DFMC = dfmc_ref_pct  → rate multiplied by 1.0 (neutral)
    #   DFMC > dfmc_max_pct  → rate multiplied by dfmc_min_factor (suppressed)
    enable_moisture_suppression: bool = True
    dfmc_ref_pct: float = 8.0    # reference DFMC (fraction × 100) for neutral spread
    dfmc_min_pct: float = 3.0    # below this: maximum acceleration
    dfmc_max_pct: float = 25.0   # above this: maximum suppression
    dfmc_max_factor: float = 1.6  # spread multiplier at very dry conditions
    dfmc_min_factor: float = 0.15 # spread multiplier at very wet conditions

    def __post_init__(self):
        """Validate configuration constraints."""
        if self.max_kernel_size < 7:
            raise ValueError(f"max_kernel_size must be >= 7; got {self.max_kernel_size}")
        if self.max_kernel_size % 2 == 0:
            raise ValueError(f"max_kernel_size must be odd; got {self.max_kernel_size}")
        if self.base_spread_km_h <= 0:
            raise ValueError(f"base_spread_km_h must be positive; got {self.base_spread_km_h}")
        if self.wind_elongation_factor < 1.0:
            raise ValueError(f"wind_elongation_factor must be >= 1.0; got {self.wind_elongation_factor}")
        if not 0.0 <= self.slope_influence <= 1.0:
            raise ValueError(f"slope_influence must be in [0.0, 1.0]; got {self.slope_influence}")
        if self.distance_decay_km <= 0:
            raise ValueError(f"distance_decay_km must be positive; got {self.distance_decay_km}")
        if self.max_daily_spread_km <= 0:
            raise ValueError(f"max_daily_spread_km must be positive; got {self.max_daily_spread_km}")
        if self.iterative_step_hours <= 0:
            raise ValueError(f"iterative_step_hours must be positive; got {self.iterative_step_hours}")

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

    def _moisture_suppression_factor(self, dfmc_fraction: float) -> float:
        """Compute a spread-rate multiplier from dead fuel moisture content.

        Returns a value in [dfmc_min_factor, dfmc_max_factor]:
        - Near dfmc_ref_pct  → 1.0 (neutral)
        - Below dfmc_min_pct → dfmc_max_factor (dry, fast spread)
        - Above dfmc_max_pct → dfmc_min_factor (moist, suppressed spread)

        Uses a piecewise-linear interpolation between breakpoints.
        """
        cfg = self.config
        dfmc_pct = dfmc_fraction * 100.0

        if dfmc_pct <= cfg.dfmc_min_pct:
            return cfg.dfmc_max_factor
        if dfmc_pct <= cfg.dfmc_ref_pct:
            # linear from max_factor → 1.0
            t = (dfmc_pct - cfg.dfmc_min_pct) / max(cfg.dfmc_ref_pct - cfg.dfmc_min_pct, 1e-9)
            return cfg.dfmc_max_factor + t * (1.0 - cfg.dfmc_max_factor)
        if dfmc_pct <= cfg.dfmc_max_pct:
            # linear from 1.0 → min_factor
            t = (dfmc_pct - cfg.dfmc_ref_pct) / max(cfg.dfmc_max_pct - cfg.dfmc_ref_pct, 1e-9)
            return 1.0 + t * (cfg.dfmc_min_factor - 1.0)
        return cfg.dfmc_min_factor

    def _get_dfmc_at_time(self, weather_at_t: "xr.Dataset") -> float:
        """Extract window-mean DFMC (fraction) from a time-sliced weather dataset.

        Returns 1.0 (neutral / no suppression) when DFMC is unavailable or all-NaN.
        """
        if "dfmc" not in weather_at_t.data_vars:
            return float("nan")
        arr = np.asarray(weather_at_t["dfmc"].values, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite))

    def predict(self, inputs: SpreadModelInput) -> SpreadForecast:
        """Predict fire spread probability over the requested horizons.

        The model propagates fire state iteratively in steps of
        ``config.iterative_step_hours`` (default 2h) and snapshots the
        probability grid at each requested horizon.  Each step uses the
        wind and moisture conditions at that point in time, so longer
        horizons compound the directional and moisture effects of intermediate
        weather rather than computing each horizon independently from t=0.
        """
        LOGGER.info(
            "Running heuristic spread v0 (iterative)",
            extra={
                "horizons_hours": list(inputs.horizons_hours),
                "step_hours": self.config.iterative_step_hours,
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

        step_h = int(self.config.iterative_step_hours)
        max_horizon = max(horizons)

        # 1. Prepare fire source: soft probability seed from active fires.
        heatmap = np.asarray(inputs.active_fires.heatmap, dtype=np.float32)
        fire_mask = (heatmap > self.config.fire_threshold).astype(np.float32)
        fire_sum = float(fire_mask.sum())

        # 2. Grid resolution in km.
        mean_lat = float(inputs.window.lat.mean())
        dy_km = max(float(inputs.grid.cell_size_deg) * 111.0, 1e-6)
        dx_km = max(float(inputs.grid.cell_size_deg) * 111.0 * np.cos(np.radians(mean_lat)), 1e-6)

        # 3. Terrain bias parameters (window-mean, computed once).
        slope_deg: float | None = None
        aspect_deg: float | None = None
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
            empty = [np.zeros_like(fire_mask, dtype=np.float32) for _ in horizons]
            return self._package_forecast(inputs, horizons, empty)

        # 4. Iterative propagation.
        current_state = fire_mask.copy()
        horizon_set = set(horizons)
        snapshots: dict[int, np.ndarray] = {}

        elapsed_h = 0
        while elapsed_h < max_horizon:
            elapsed_h += step_h

            target_time = inputs.forecast_reference_time + timedelta(hours=elapsed_h)

            # Select weather at this step.
            if "time" in inputs.weather_cube.dims:
                weather_at_t = inputs.weather_cube.sel(
                    time=self._as_datetime64_utc_naive(target_time), method="nearest"
                )
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

            # Moisture suppression factor for this step.
            moisture_factor = 1.0
            if self.config.enable_moisture_suppression:
                dfmc_val = self._get_dfmc_at_time(weather_at_t)
                if np.isfinite(dfmc_val):
                    moisture_factor = self._moisture_suppression_factor(dfmc_val)
                    LOGGER.debug(
                        "Step +%dh: DFMC=%.3f → moisture_factor=%.3f",
                        elapsed_h, dfmc_val, moisture_factor,
                    )

            # Build kernel for one step (step_h hours, current wind).
            kernel = self._generate_kernel(
                step_h,
                u10,
                v10,
                dy_km,
                dx_km,
                slope_deg=slope_deg,
                aspect_deg=aspect_deg,
                moisture_factor=moisture_factor,
            )

            # Convolve current state with kernel.
            next_state = fftconvolve(current_state, kernel, mode="same").astype(np.float32, copy=False)

            # Normalize to [0, 1]; FFT can introduce tiny negatives.
            max_val = float(np.max(next_state)) if next_state.size else 0.0
            if max_val > 0.0:
                next_state = next_state / max_val
            next_state = np.clip(next_state, 0.0, 1.0, out=next_state)

            # Apply terrain masks.
            if inputs.terrain.valid_data_mask is not None:
                next_state = next_state * inputs.terrain.valid_data_mask
            if inputs.terrain.aoi_mask is not None:
                next_state = next_state * inputs.terrain.aoi_mask

            current_state = next_state

            # Snapshot at requested horizons (exact match or overshoot).
            for h in horizon_set:
                if h not in snapshots and elapsed_h >= h:
                    snapshots[h] = current_state.copy()

        # Collect in original horizon order; fill any missed with current_state.
        forecast_grids = [snapshots.get(h, current_state.copy()) for h in horizons]
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
        times_64 = [HeuristicSpreadModelV0._as_datetime64_utc_naive(t) for t in times]

        da = xr.DataArray(
            np.stack(forecast_grids).astype(np.float32, copy=False),
            coords={
                "time": times_64,
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
            model_name="HeuristicSpreadModelV0",
            model_version="0.1.0",
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
        moisture_factor: float = 1.0,
    ) -> np.ndarray:
        """Generate an anisotropic kernel centered at origin with downwind bias.

        Parameters
        ----------
        moisture_factor : float
            Multiplier on the spread rate derived from DFMC.  Values < 1 suppress
            spread (wet conditions); values > 1 accelerate it (dry conditions).
        """
        # Wind speed magnitude
        wind_speed = np.sqrt(u_ms**2 + v_ms**2)

        # Additive spread-rate model, modulated by moisture, with a hard daily cap.
        spread_rate_km_h = float(
            (self.config.base_spread_km_h + (wind_speed * self.config.wind_influence_km_h_per_ms))
            * float(moisture_factor)
        )
        max_dist_by_rate_km = float(spread_rate_km_h * horizon_h)
        max_dist_by_cap_km = float(self.config.max_daily_spread_km * (horizon_h / 24.0))
        max_dist_km = float(min(max_dist_by_rate_km, max_dist_by_cap_km))
        max_dist_km = max(max_dist_km, 1e-3)
        
        # Kernel size: cover the spread distance
        # We'll use 4x the max spread distance in pixels to capture the decay
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
        kernel = np.exp(-eff_dist / (max_dist_km + self.config.distance_decay_km))
        
        return kernel
