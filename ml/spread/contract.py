"""Spread model contract types and interface.

This module defines the primary Protocol and data structures for wildfire
spread forecasting models in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence, TYPE_CHECKING

import xarray as xr

# Use TYPE_CHECKING to avoid circular imports if any, or to keep dependencies clean.
if TYPE_CHECKING:
    from api.core.grid import GridSpec, GridWindow
    from api.fires.service import FireHeatmapWindow
    from api.terrain.window import TerrainWindow

# MVP horizons for v1 spread model.
# Tuple to keep this constant immutable and safe to reuse as a default.
DEFAULT_HORIZONS_HOURS: tuple[int, ...] = (24, 48, 72)

@dataclass(frozen=True, slots=True)
class SpreadModelInput:
    """Input package for a spread model prediction."""
    grid: GridSpec
    window: GridWindow
    active_fires: FireHeatmapWindow
    # Expected to be aligned to the analysis grid conventions (see `api.core.grid`):
    # - dims/coords include `time`, `lat`, `lon` (lat/lon increasing, cell centers)
    # - variables depend on the chosen weather model/features.
    weather_cube: xr.Dataset
    terrain: TerrainWindow
    forecast_reference_time: datetime
    horizons_hours: Sequence[int] = DEFAULT_HORIZONS_HOURS

@dataclass(frozen=True, slots=True)
class SpreadForecast:
    """Output package for a spread model prediction."""
    probabilities: xr.DataArray  # Dims: (time, lat, lon), Values: [0, 1]
    forecast_reference_time: datetime
    horizons_hours: Sequence[int]
    model_name: str = "unknown"  # Name of the model that produced this forecast
    model_version: str = ""      # Version string for model reproducibility
    
    def validate(self) -> None:
        """Enforce dimensions, coordinates, and value ranges."""
        # 1. Dimensions
        expected_dims = ("time", "lat", "lon")
        if self.probabilities.dims != expected_dims:
            raise ValueError(f"Expected dimensions {expected_dims}, got {self.probabilities.dims}")

        # 1b. Leading dimension length matches horizons
        if int(self.probabilities.sizes["time"]) != len(self.horizons_hours):
            raise ValueError(
                "Expected len(horizons_hours) to match probabilities.time length: "
                f"len(horizons_hours)={len(self.horizons_hours)} time={int(self.probabilities.sizes['time'])}"
            )
        
        # 2. Required coordinates
        #
        # Note: In xarray, an array can have named dimensions without explicit
        # coordinate variables (xarray will use implicit integer indices).
        # The spread forecast contract requires explicit cell-center coordinates.
        required_coords = ("time", "lat", "lon")
        missing_coords = [c for c in required_coords if c not in self.probabilities.coords]
        if missing_coords:
            missing_str = ", ".join(repr(c) for c in missing_coords)
            raise ValueError(
                f"Missing required coordinate(s): {missing_str}. "
                "SpreadForecast.probabilities must include explicit 'time', 'lat', and 'lon' "
                "cell-center coordinates matching the input window."
            )

        # 2b. Coordinate shape sanity (must align with their dimension)
        lat = self.probabilities.coords["lat"]
        lon = self.probabilities.coords["lon"]
        if tuple(lat.dims) != ("lat",):
            raise ValueError("Expected 'lat' coordinate to have dims ('lat',).")
        if tuple(lon.dims) != ("lon",):
            raise ValueError("Expected 'lon' coordinate to have dims ('lon',).")
        if int(lat.sizes["lat"]) != int(self.probabilities.sizes["lat"]):
            raise ValueError("Expected 'lat' coordinate length to match probabilities.lat length.")
        if int(lon.sizes["lon"]) != int(self.probabilities.sizes["lon"]):
            raise ValueError("Expected 'lon' coordinate length to match probabilities.lon length.")
        
        # 3. Lead time (optional but recommended)
        if "lead_time_hours" not in self.probabilities.coords:
            # We don't strictly require it here if the user handles it, 
            # but our standard outputs should have it.
            pass
        else:
            # If provided, it should be aligned with the time dimension.
            lead = self.probabilities.coords["lead_time_hours"]
            if tuple(lead.dims) != ("time",):
                raise ValueError("Expected 'lead_time_hours' coordinate to have dims ('time',).")
            if int(lead.sizes["time"]) != len(self.horizons_hours):
                raise ValueError("Expected 'lead_time_hours' length to match horizons_hours.")

        # 4. Values finite + range [0, 1]
        # We allow a small epsilon for floating point issues.
        if self.probabilities.size > 0:
            data = self.probabilities.to_numpy()
            if not (data.dtype.kind == "f"):
                raise ValueError(f"Expected float probabilities dtype, got {data.dtype}.")
            if not (data.size == 0 or (data == data).all()):  # NaN check without importing numpy
                raise ValueError("Probabilities contain NaN values.")
            # No +/- inf.
            if not (data.size == 0 or ((data != float("inf")) & (data != float("-inf"))).all()):
                raise ValueError("Probabilities contain non-finite values.")

            p_min = float(data.min())
            p_max = float(data.max())
            if p_min < -1e-6 or p_max > 1.0 + 1e-6:
                raise ValueError(f"Probabilities out of range [0, 1]: min={p_min}, max={p_max}")

class SpreadModel(Protocol):
    """Protocol for wildfire spread forecasting models."""
    
    def predict(self, inputs: SpreadModelInput) -> SpreadForecast:
        """Predict fire spread probability over the requested horizons.
        
        Parameters
        ----------
        inputs : SpreadModelInput
            The input features including fire state, weather, and terrain.
            
        Returns
        -------
        SpreadForecast
            Probability grids for each time horizon.
        """
        ...

