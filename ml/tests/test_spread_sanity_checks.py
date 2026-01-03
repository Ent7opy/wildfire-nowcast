import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timezone
from dataclasses import dataclass

from ml.spread.sanity_checks import (
    check_nonempty_when_fires_present,
    check_monotonic_footprint,
    check_wind_elongation,
)
from ml.spread.contract import SpreadForecast, SpreadModelInput

@dataclass(frozen=True)
class MockWindow:
    lat: np.ndarray
    lon: np.ndarray

@dataclass(frozen=True)
class MockFireHeatmap:
    heatmap: np.ndarray

def create_mock_forecast(horizons, lat, lon, data):
    da = xr.DataArray(
        data,
        coords={
            "time": [datetime(2025, 1, 1, tzinfo=timezone.utc)] * len(horizons),
            "lat": lat,
            "lon": lon,
            "lead_time_hours": ("time", horizons)
        },
        dims=("time", "lat", "lon")
    )
    return SpreadForecast(
        probabilities=da,
        forecast_reference_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        horizons_hours=horizons
    )

def test_check_nonempty_when_fires_present():
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    
    # Case: fires present, forecast empty -> fail
    inputs = MagicMock(spec=SpreadModelInput)
    inputs.active_fires = MockFireHeatmap(heatmap=np.array([[1.0, 0.0], [0.0, 0.0]]))
    forecast = create_mock_forecast([24], lat, lon, np.zeros((1, 2, 2)))
    
    with pytest.raises(ValueError, match="Spread forecast is empty"):
        check_nonempty_when_fires_present(inputs, forecast)
        
    # Case: fires present, forecast non-empty -> pass
    forecast = create_mock_forecast([24], lat, lon, np.array([[[0.5, 0.0], [0.0, 0.0]]]))
    check_nonempty_when_fires_present(inputs, forecast)

def test_check_monotonic_footprint():
    lat = np.arange(5)
    lon = np.arange(5)
    
    # Case: footprint grows -> pass
    data = np.zeros((2, 5, 5))
    data[0, 2, 2] = 0.5
    data[1, 1:4, 1:4] = 0.5
    forecast = create_mock_forecast([24, 48], lat, lon, data)
    check_monotonic_footprint(forecast)
    
    # Case: footprint shrinks -> fail
    data = np.zeros((2, 5, 5))
    data[0, 1:4, 1:4] = 0.5
    data[1, 2, 2] = 0.5
    forecast = create_mock_forecast([24, 48], lat, lon, data)
    with pytest.raises(ValueError, match="Fire footprint shrank significantly"):
        check_monotonic_footprint(forecast)

def test_check_wind_elongation():
    lat = np.linspace(40, 41, 21)
    lon = np.linspace(20, 21, 21)
    
    # Strong East wind (u=10, v=0)
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.ones((21, 21)) * 10.0),
            "v10": (("lat", "lon"), np.zeros((21, 21))),
        },
        coords={"lat": lat, "lon": lon}
    )
    inputs = MagicMock(spec=SpreadModelInput)
    inputs.weather_cube = weather
    
    # Case 1: Elongated along East-West -> pass
    # Probability distribution concentrated along the center lat, spread along lon
    data = np.zeros((1, 21, 21))
    data[0, 10, 5:16] = 0.5 # A line along lon
    forecast = create_mock_forecast([24], lat, lon, data)
    check_wind_elongation(inputs, forecast)
    
    # Case 2: Elongated along North-South (misaligned) -> fail
    data = np.zeros((1, 21, 21))
    data[0, 5:16, 10] = 0.5 # A line along lat
    forecast = create_mock_forecast([24], lat, lon, data)
    with pytest.raises(ValueError, match="Footprint major axis is misaligned"):
        check_wind_elongation(inputs, forecast)

    # Case 3: Circular (no anisotropy) -> returns early (pass)
    data = np.zeros((1, 21, 21))
    data[0, 9:12, 9:12] = 0.5
    forecast = create_mock_forecast([24], lat, lon, data)
    check_wind_elongation(inputs, forecast, min_anisotropy=1.5)

from unittest.mock import MagicMock

