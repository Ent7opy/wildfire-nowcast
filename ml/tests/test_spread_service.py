import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import numpy as np

from ml.spread.service import run_spread_forecast, SpreadForecastRequest, MAX_AOI_CELLS
from ml.spread.contract import SpreadForecast, SpreadModelInput
from api.core.grid import GridSpec, GridWindow

@pytest.fixture
def mock_grid():
    return GridSpec(
        crs="EPSG:4326",
        cell_size_deg=0.01,
        origin_lat=40.0,
        origin_lon=20.0,
        n_lat=100,
        n_lon=100,
    )

@pytest.fixture
def mock_window():
    lat = np.array([40.05, 40.15])
    lon = np.array([20.05, 20.15])
    return GridWindow(i0=0, i1=2, j0=0, j1=2, lat=lat, lon=lon)

@pytest.fixture
def mock_spread_inputs(mock_grid, mock_window):
    mock = MagicMock()
    mock.grid = mock_grid
    mock.window = mock_window
    # Service code logs `float(inputs_package.active_fires.heatmap.sum())`.
    # Use a real numeric array to avoid `float(MagicMock)` TypeError.
    mock.active_fires = MagicMock()
    mock.active_fires.heatmap = np.zeros((1, 1), dtype=float)
    mock.to_model_input.return_value = MagicMock(spec=SpreadModelInput)
    return mock

def test_run_spread_forecast_success(mock_spread_inputs):
    # Setup
    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
    )
    
    mock_forecast = MagicMock(spec=SpreadForecast)
    # Service code logs `float(forecast.probabilities.min()/max())`.
    mock_forecast.probabilities = MagicMock()
    mock_forecast.probabilities.min.return_value = 0.0
    mock_forecast.probabilities.max.return_value = 0.0
    mock_model = MagicMock()
    mock_model.predict.return_value = mock_forecast
    
    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs):
        # Execute
        result = run_spread_forecast(request, model=mock_model)
        
        # Verify
        assert result == mock_forecast
        mock_model.predict.assert_called_once()
        mock_forecast.validate.assert_called_once()

def test_run_spread_forecast_aoi_too_large(mock_grid):
    # Setup - Window that exceeds MAX_AOI_CELLS
    # Assuming MAX_AOI_CELLS = 40000, 201x200 = 40200
    side = int(np.sqrt(MAX_AOI_CELLS)) + 10
    lat = np.arange(side)
    lon = np.arange(side)
    large_window = GridWindow(i0=0, i1=side, j0=0, j1=side, lat=lat, lon=lon)
    
    mock_inputs = MagicMock()
    mock_inputs.window = large_window
    # Service logs active fire count before AOI size check.
    mock_inputs.active_fires = MagicMock()
    mock_inputs.active_fires.heatmap = np.zeros((1, 1), dtype=float)
    
    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 25.0, 45.0),
        forecast_reference_time=ref_time,
    )
    
    with patch("ml.spread.service.build_spread_inputs", return_value=mock_inputs):
        # Execute & Verify
        with pytest.raises(ValueError, match="AOI too large"):
            run_spread_forecast(request)

def test_run_spread_forecast_not_implemented_cluster():
    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        fire_cluster_id="cluster_123"
    )
    
    with pytest.raises(NotImplementedError, match="fire_cluster_id is not yet supported"):
        run_spread_forecast(request)

def test_run_spread_forecast_default_model(mock_spread_inputs):
    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
    )
    
    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs):
        with patch("ml.spread.service.HeuristicSpreadModelV0") as mock_heuristic_cls:
            mock_model = mock_heuristic_cls.return_value
            mock_forecast = MagicMock(spec=SpreadForecast)
            # Service code logs `float(forecast.probabilities.min()/max())`.
            mock_forecast.probabilities = MagicMock()
            mock_forecast.probabilities.min.return_value = 0.0
            mock_forecast.probabilities.max.return_value = 0.0
            mock_model.predict.return_value = mock_forecast
            run_spread_forecast(request)
            mock_model.predict.assert_called_once()
            mock_forecast.validate.assert_called_once()


def test_run_spread_forecast_applies_service_calibration_when_available(monkeypatch, mock_spread_inputs):
    """If a calibrator run dir is available, the service should calibrate outputs."""
    # Avoid bias-corrector resolution affecting the call.
    monkeypatch.delenv("WEATHER_BIAS_CORRECTOR_PATH", raising=False)
    monkeypatch.delenv("WEATHER_BIAS_CORRECTOR_ROOT", raising=False)

    # Force calibrator resolution.
    monkeypatch.setenv("SPREAD_CALIBRATOR_RUN_DIR", "/fake/calibrator/run")

    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        horizons_hours=[24, 48],
    )

    # Model returns raw probabilities of 0.8 everywhere; calibrator will halve them to 0.4.
    probs = np.full((2, 2, 2), 0.8, dtype=np.float32)
    da = MagicMock()

    import xarray as xr
    da = xr.DataArray(
        probs.copy(),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.5, 1.5], "lon": [10.5, 11.5], "lead_time_hours": ("time", [24, 48])},
    )
    forecast = SpreadForecast(probabilities=da, forecast_reference_time=ref_time, horizons_hours=[24, 48])

    mock_model = MagicMock()
    mock_model.predict.return_value = forecast

    class DummyCalibrator:
        per_horizon_models = {24: object(), 48: object()}
        metadata = {"run_id": "cal-run-1", "method": "dummy"}

        def calibrate_probs(self, raw_probs: np.ndarray, horizon_hours: int) -> np.ndarray:
            return np.asarray(raw_probs) * 0.5

    with (
        patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs),
        patch("ml.spread.service.SpreadProbabilityCalibrator.load", return_value=DummyCalibrator()),
    ):
        out = run_spread_forecast(request, model=mock_model)

    assert np.allclose(out.probabilities.values, 0.4, rtol=0.0, atol=1e-6)
    assert out.probabilities.attrs.get("calibration_applied") is True
    assert out.probabilities.attrs.get("calibration_source") == "service"
    assert out.probabilities.attrs.get("calibration_run_id") == "cal-run-1"


def test_run_spread_forecast_passes_weather_bias_corrector_path(monkeypatch, mock_spread_inputs):
    """Service should pass a configured bias corrector path into build_spread_inputs."""
    monkeypatch.setenv("WEATHER_BIAS_CORRECTOR_PATH", "/fake/corrector.json")
    monkeypatch.delenv("SPREAD_CALIBRATOR_RUN_DIR", raising=False)
    monkeypatch.delenv("SPREAD_CALIBRATOR_ROOT", raising=False)

    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
    )

    mock_forecast = MagicMock(spec=SpreadForecast)
    mock_forecast.probabilities = MagicMock()
    mock_forecast.probabilities.min.return_value = 0.0
    mock_forecast.probabilities.max.return_value = 0.0
    mock_model = MagicMock()
    mock_model.predict.return_value = mock_forecast

    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs) as mock_build:
        run_spread_forecast(request, model=mock_model)

    assert mock_build.call_count == 1
    _, kwargs = mock_build.call_args
    assert str(kwargs["weather_bias_corrector_path"]) == "/fake/corrector.json"

