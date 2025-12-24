import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Sequence

from ml.spread.contract import (
    SpreadForecast, 
    SpreadModel, 
    SpreadModelInput,
)

def create_mock_forecast(
    horizons: Sequence[int], 
    ref_time: datetime, 
    shape: tuple[int, int] = (10, 10),
    data: np.ndarray | None = None
) -> SpreadForecast:
    """Helper to create a valid SpreadForecast."""
    nt = len(horizons)
    ny, nx = shape
    
    if data is None:
        data = np.zeros((nt, ny, nx), dtype=np.float32)
    
    times = [ref_time + timedelta(hours=h) for h in horizons]
    
    da = xr.DataArray(
        data,
        coords={
            "time": times,
            "lat": np.arange(ny),
            "lon": np.arange(nx),
            "lead_time_hours": ("time", horizons)
        },
        dims=("time", "lat", "lon"),
        name="spread_probability"
    )
    
    return SpreadForecast(
        probabilities=da,
        forecast_reference_time=ref_time,
        horizons_hours=horizons
    )

def test_spread_forecast_validation_valid():
    horizons = [24, 48, 72]
    ref_time = datetime.now(timezone.utc)
    forecast = create_mock_forecast(horizons, ref_time)
    
    # Should not raise
    forecast.validate()

def test_spread_forecast_validation_invalid_dims():
    horizons = [24, 48, 72]
    ref_time = datetime.now(timezone.utc)
    forecast = create_mock_forecast(horizons, ref_time)
    
    # Force invalid dims using rename
    invalid_da = forecast.probabilities.rename({"time": "t", "lat": "y", "lon": "x"})
    
    invalid_forecast = SpreadForecast(
        probabilities=invalid_da,
        forecast_reference_time=forecast.forecast_reference_time,
        horizons_hours=forecast.horizons_hours
    )
    
    with pytest.raises(ValueError, match="Expected dimensions"):
        invalid_forecast.validate()

def test_spread_forecast_validation_invalid_values():
    horizons = [24]
    ref_time = datetime.now(timezone.utc)
    
    # Value > 1.0
    data = np.array([[[1.5]]], dtype=np.float32)
    forecast = create_mock_forecast(horizons, ref_time, shape=(1, 1), data=data)
    
    with pytest.raises(ValueError, match="Probabilities out of range"):
        forecast.validate()
        
    # Value < 0.0
    data = np.array([[[-0.1]]], dtype=np.float32)
    forecast = create_mock_forecast(horizons, ref_time, shape=(1, 1), data=data)
    
    with pytest.raises(ValueError, match="Probabilities out of range"):
        forecast.validate()

def test_spread_forecast_validation_requires_lat_lon_coords():
    """A forecast with lat/lon dims but no explicit coords must fail validation."""
    horizons = [24, 48]
    ref_time = datetime.now(timezone.utc)

    nt, ny, nx = len(horizons), 3, 4
    data = np.zeros((nt, ny, nx), dtype=np.float32)
    times = [ref_time + timedelta(hours=h) for h in horizons]

    # Intentionally omit lat/lon coords: xarray will treat them as implicit indices.
    da = xr.DataArray(
        data,
        coords={
            "time": times,
            "lead_time_hours": ("time", horizons),
        },
        dims=("time", "lat", "lon"),
        name="spread_probability",
    )

    forecast = SpreadForecast(
        probabilities=da,
        forecast_reference_time=ref_time,
        horizons_hours=horizons,
    )

    with pytest.raises(ValueError, match="Missing required coordinate\\(s\\).*'lat'.*'lon'"):
        forecast.validate()

def test_spread_model_protocol_dummy():
    """Verify that a dummy class satisfying the Protocol works."""
    class DummyModel:
        def predict(self, inputs: SpreadModelInput) -> SpreadForecast:
            return create_mock_forecast(
                inputs.horizons_hours, 
                inputs.forecast_reference_time,
                shape=(int(inputs.window.lat.size), int(inputs.window.lon.size))
            )

    # Check if DummyModel implements SpreadModel Protocol
    model: SpreadModel = DummyModel()
    
    # Create minimal concrete inputs without importing the full API package
    # (the dataclass does not enforce runtime types; annotations are deferred).
    @dataclass(frozen=True)
    class DummyWindow:
        lat: np.ndarray
        lon: np.ndarray

    window = DummyWindow(
        lat=np.arange(5, dtype=float),
        lon=np.arange(5, dtype=float),
    )

    weather = xr.Dataset(coords={"time": [datetime.now(timezone.utc)], "lat": window.lat, "lon": window.lon})
    inputs = SpreadModelInput(
        grid=object(),
        window=window,
        active_fires=object(),
        weather_cube=weather,
        terrain=object(),
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[24, 48],
    )
    
    forecast = model.predict(inputs)
    assert forecast.probabilities.shape == (2, 5, 5)
    forecast.validate()

def test_empty_window_behavior():
    """Test that the contract handles 0x0 windows gracefully."""
    horizons = [24, 48]
    ref_time = datetime.now(timezone.utc)
    
    # 0x0 spatial window
    forecast = create_mock_forecast(horizons, ref_time, shape=(0, 0))
    
    assert forecast.probabilities.shape == (2, 0, 0)
    # Validation should still pass as range check on empty array is fine
    forecast.validate()

