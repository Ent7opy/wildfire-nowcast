import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np

from ml.spread.service import (
    ForecastInputFallbackError,
    MAX_AOI_CELLS,
    SpreadForecastRequest,
    run_spread_forecast,
)
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
    mock.weather_fallback_used = False
    mock.terrain_fallback_used = False
    mock.weather_cube = MagicMock()
    mock.weather_cube.attrs = {}
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
    assert Path(kwargs["weather_bias_corrector_path"]) == Path("/fake/corrector.json")


def test_run_spread_forecast_bbox_only_no_region_name(monkeypatch, mock_spread_inputs):
    """Service should support bbox-only requests without region_name (JIT forecasting)."""
    monkeypatch.delenv("WEATHER_BIAS_CORRECTOR_PATH", raising=False)
    monkeypatch.delenv("WEATHER_BIAS_CORRECTOR_ROOT", raising=False)
    monkeypatch.delenv("SPREAD_CALIBRATOR_RUN_DIR", raising=False)
    monkeypatch.delenv("SPREAD_CALIBRATOR_ROOT", raising=False)

    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name=None,
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
        result = run_spread_forecast(request, model=mock_model)

    assert result == mock_forecast
    mock_model.predict.assert_called_once()
    mock_forecast.validate.assert_called_once()
    
    # Verify build_spread_inputs was called with region_name=None
    assert mock_build.call_count == 1
    _, kwargs = mock_build.call_args
    assert kwargs["region_name"] is None
    assert kwargs["bbox"] == (20.0, 40.0, 20.2, 40.2)
    # Weather bias corrector should be None for location-based forecasts
    assert kwargs["weather_bias_corrector_path"] is None


def test_run_spread_forecast_strict_raises_on_weather_fallback(mock_spread_inputs):
    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        strict_inputs=True,
    )

    mock_spread_inputs.weather_fallback_used = True
    mock_spread_inputs.weather_cube.attrs = {"weather_fallback_reason": "no_weather_run_found"}

    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs):
        with pytest.raises(ForecastInputFallbackError, match="weather fallback used"):
            run_spread_forecast(request)


def test_run_spread_forecast_strict_raises_on_region_terrain_fallback(mock_spread_inputs):
    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        strict_inputs=True,
    )

    mock_spread_inputs.terrain_fallback_used = True

    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs):
        with pytest.raises(ForecastInputFallbackError, match="terrain fallback used"):
            run_spread_forecast(request)


def test_run_spread_forecast_strict_allows_location_based_terrain_fallback(mock_spread_inputs):
    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name=None,
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        strict_inputs=True,
    )

    mock_spread_inputs.terrain_fallback_used = True
    mock_forecast = MagicMock(spec=SpreadForecast)
    mock_forecast.probabilities = MagicMock()
    mock_forecast.probabilities.min.return_value = 0.0
    mock_forecast.probabilities.max.return_value = 0.0
    mock_model = MagicMock()
    mock_model.predict.return_value = mock_forecast

    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs):
        out = run_spread_forecast(request, model=mock_model)

    assert out == mock_forecast


def test_run_spread_forecast_adds_confidence_and_staleness_attrs(monkeypatch, mock_spread_inputs):
    import xarray as xr

    monkeypatch.setenv("SPREAD_STALE_WARN_HOURS", "12")
    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    mock_spread_inputs.weather_cube.attrs = {"weather_run_time": "2025-12-26T00:00:00+00:00"}

    forecast = SpreadForecast(
        probabilities=xr.DataArray(
            np.full((1, 2, 2), 0.2, dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": [0.0, 1.0], "lon": [0.0, 1.0], "lead_time_hours": ("time", [24])},
        ),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        model_name="dummy",
        model_version="x",
    )
    model = MagicMock()
    model.predict.return_value = forecast

    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
    )

    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs):
        out = run_spread_forecast(request, model=model)

    assert out.probabilities.attrs["confidence_level"] == "low"
    assert float(out.probabilities.attrs["staleness_hours"]) == pytest.approx(12.0)
    assert out.probabilities.attrs["fallback_used"] is False


def test_run_spread_forecast_shadow_mode_sets_shadow_attrs(monkeypatch, mock_spread_inputs):
    import xarray as xr

    monkeypatch.setenv("SPREAD_SHADOW_ENABLED", "true")
    monkeypatch.delenv("SPREAD_CALIBRATOR_RUN_DIR", raising=False)
    monkeypatch.delenv("SPREAD_CALIBRATOR_ROOT", raising=False)

    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    base_probs = xr.DataArray(
        np.full((1, 2, 2), 0.2, dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": [0.0, 1.0], "lon": [0.0, 1.0], "lead_time_hours": ("time", [24])},
    )
    champion_forecast = SpreadForecast(
        probabilities=base_probs.copy(deep=True),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        model_name="champion",
        model_version="v1",
    )
    challenger_forecast = SpreadForecast(
        probabilities=xr.DataArray(
            np.full((1, 2, 2), 0.8, dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": [0.0, 1.0], "lon": [0.0, 1.0], "lead_time_hours": ("time", [24])},
        ),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        model_name="challenger",
        model_version="v2",
    )

    champion_model = MagicMock()
    champion_model.predict.return_value = champion_forecast
    challenger_model = MagicMock()
    challenger_model.predict.return_value = challenger_forecast

    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        shadow_model_name="LearnedSpreadModelV2",
        shadow_model_params={"model_run_dir": "/tmp/not-used"},
    )

    with (
        patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs),
        patch("ml.spread.factory.get_spread_model", return_value=challenger_model),
    ):
        out = run_spread_forecast(request, model=champion_model)

    assert out.probabilities.attrs.get("shadow_evaluated") is True
    summary = out.probabilities.attrs.get("shadow_metrics_summary")
    assert isinstance(summary, dict)
    assert "mean_abs_probability_delta" in summary


def test_run_spread_forecast_weather_fallback_sets_low_confidence(mock_spread_inputs):
    import xarray as xr

    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    mock_spread_inputs.weather_fallback_used = True
    mock_spread_inputs.weather_cube.attrs = {"weather_fallback_reason": "no_weather_run_found"}

    forecast = SpreadForecast(
        probabilities=xr.DataArray(
            np.full((1, 2, 2), 0.2, dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": [0.0, 1.0], "lon": [0.0, 1.0], "lead_time_hours": ("time", [24])},
        ),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        model_name="dummy",
        model_version="x",
    )
    model = MagicMock()
    model.predict.return_value = forecast

    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        strict_inputs=False,
    )

    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs):
        out = run_spread_forecast(request, model=model)

    assert out.probabilities.attrs["confidence_level"] == "low"
    assert out.probabilities.attrs["fallback_used"] is True


def test_run_spread_forecast_spatial_sanity_fallback_to_heuristic(monkeypatch, mock_grid, mock_window):
    import xarray as xr

    monkeypatch.setenv("SPREAD_SANITY_ENABLED", "true")
    monkeypatch.setenv("SPREAD_SANITY_HIGH_PROB_THRESHOLD", "0.3")
    monkeypatch.setenv("SPREAD_SANITY_SEED_MIN_PROB", "0.05")
    monkeypatch.setenv("SPREAD_SANITY_NEAR_SEED_PX", "0")
    monkeypatch.delenv("SPREAD_CALIBRATOR_RUN_DIR", raising=False)
    monkeypatch.delenv("SPREAD_CALIBRATOR_ROOT", raising=False)

    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
    )

    mock_inputs = MagicMock()
    mock_inputs.grid = mock_grid
    mock_inputs.window = mock_window
    mock_inputs.active_fires = MagicMock()
    # Single ignition seed at [0, 0].
    mock_inputs.active_fires.heatmap = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    mock_inputs.weather_fallback_used = False
    mock_inputs.terrain_fallback_used = False
    mock_inputs.weather_cube = MagicMock()
    mock_inputs.weather_cube.attrs = {}
    model_input = MagicMock(spec=SpreadModelInput)
    mock_inputs.to_model_input.return_value = model_input

    learned_forecast = SpreadForecast(
        probabilities=xr.DataArray(
            # Implausible: high prob only away from seed.
            np.array([[[0.0, 0.0], [0.0, 0.9]]], dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": [0.0, 1.0], "lon": [0.0, 1.0], "lead_time_hours": ("time", [24])},
        ),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        model_name="LearnedSpreadModelV3",
        model_version="v3",
    )
    learned_model = MagicMock()
    learned_model.predict.return_value = learned_forecast

    heuristic_forecast = SpreadForecast(
        probabilities=xr.DataArray(
            # Plausible: highest probability at seed.
            np.array([[[0.95, 0.1], [0.05, 0.02]]], dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": [0.0, 1.0], "lon": [0.0, 1.0], "lead_time_hours": ("time", [24])},
        ),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        model_name="HeuristicSpreadModelV0",
        model_version="v0",
    )
    heuristic_model = MagicMock()
    heuristic_model.predict.return_value = heuristic_forecast

    with (
        patch("ml.spread.service.build_spread_inputs", return_value=mock_inputs),
        patch("ml.spread.service.HeuristicSpreadModelV0", return_value=heuristic_model),
    ):
        out = run_spread_forecast(request, model=learned_model)

    assert out.model_name == "HeuristicSpreadModelV0"
    assert out.probabilities.attrs.get("sanity_fallback_used") is True
    assert "spatial_sanity_failed" in str(out.probabilities.attrs.get("sanity_fallback_reason"))
    assert out.probabilities.attrs.get("sanity_original_model_name") == learned_model.__class__.__name__
    learned_model.predict.assert_called_once_with(model_input)
    heuristic_model.predict.assert_called_once_with(model_input)


def test_run_spread_forecast_mvp_guardrail_warn_mode(monkeypatch, mock_spread_inputs):
    import xarray as xr

    monkeypatch.setenv("SPREAD_MVP_GUARD_ENABLED", "true")
    monkeypatch.setenv("SPREAD_MVP_GUARD_HORIZON_HOURS", "24")
    monkeypatch.setenv("SPREAD_MVP_GUARD_PROB_THRESHOLD", "0.7")
    monkeypatch.setenv("SPREAD_MVP_GUARD_MAX_COVERAGE", "0.60")
    monkeypatch.setenv("SPREAD_MVP_GUARD_MAX_SEED_CELLS", "1")

    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        strict_inputs=False,
    )

    mock_spread_inputs.active_fires.heatmap = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    forecast = SpreadForecast(
        probabilities=xr.DataArray(
            # 3/4 cells >= 0.7 => 0.75 coverage -> triggers warning.
            np.array([[[0.9, 0.8], [0.75, 0.2]]], dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": [0.0, 1.0], "lon": [0.0, 1.0], "lead_time_hours": ("time", [24])},
        ),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        model_name="HeuristicSpreadModelV0",
        model_version="v0",
    )
    model = MagicMock()
    model.predict.return_value = forecast

    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs):
        out = run_spread_forecast(request, model=model)

    assert out.probabilities.attrs.get("mvp_guardrail_triggered") is True
    assert out.probabilities.attrs.get("mvp_guardrail_mode") == "warn"
    assert "mvp_guardrail_oversized_footprint" in str(out.probabilities.attrs.get("mvp_guardrail_reason"))
    assert out.probabilities.attrs.get("confidence_level") == "low"


def test_run_spread_forecast_mvp_guardrail_strict_fails(monkeypatch, mock_spread_inputs):
    import xarray as xr

    monkeypatch.setenv("SPREAD_MVP_GUARD_ENABLED", "true")
    monkeypatch.setenv("SPREAD_MVP_GUARD_HORIZON_HOURS", "24")
    monkeypatch.setenv("SPREAD_MVP_GUARD_PROB_THRESHOLD", "0.7")
    monkeypatch.setenv("SPREAD_MVP_GUARD_MAX_COVERAGE", "0.60")
    monkeypatch.setenv("SPREAD_MVP_GUARD_MAX_SEED_CELLS", "1")

    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 20.2, 40.2),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        strict_inputs=True,
    )

    mock_spread_inputs.active_fires.heatmap = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    forecast = SpreadForecast(
        probabilities=xr.DataArray(
            np.array([[[0.9, 0.8], [0.75, 0.2]]], dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": [0.0, 1.0], "lon": [0.0, 1.0], "lead_time_hours": ("time", [24])},
        ),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
        model_name="HeuristicSpreadModelV0",
        model_version="v0",
    )
    model = MagicMock()
    model.predict.return_value = forecast

    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs):
        with pytest.raises(ValueError, match="MVP_GUARD_STOP"):
            run_spread_forecast(request, model=model)


def test_run_spread_forecast_cpu_latency_p95_40k_cells(monkeypatch):
    import xarray as xr

    monkeypatch.setenv("SPREAD_SHADOW_ENABLED", "false")
    monkeypatch.setenv("SPREAD_STALE_WARN_HOURS", "12")
    monkeypatch.setenv("SPREAD_SERVE_STALE", "true")

    side = 200
    lat = np.linspace(40.0, 41.99, side)
    lon = np.linspace(20.0, 21.99, side)

    mock_inputs = MagicMock()
    mock_inputs.grid = GridSpec(
        crs="EPSG:4326",
        cell_size_deg=0.01,
        origin_lat=40.0,
        origin_lon=20.0,
        n_lat=side,
        n_lon=side,
    )
    mock_inputs.window = GridWindow(i0=0, i1=side, j0=0, j1=side, lat=lat, lon=lon)
    mock_inputs.active_fires = MagicMock()
    mock_inputs.active_fires.heatmap = np.zeros((side, side), dtype=float)
    mock_inputs.weather_fallback_used = False
    mock_inputs.terrain_fallback_used = False
    mock_inputs.weather_cube = MagicMock()
    mock_inputs.weather_cube.attrs = {}
    mock_inputs.to_model_input.return_value = MagicMock(spec=SpreadModelInput)

    ref_time = datetime(2025, 12, 26, 12, 0, tzinfo=timezone.utc)
    request = SpreadForecastRequest(
        region_name="test_region",
        bbox=(20.0, 40.0, 22.0, 42.0),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
    )

    def _build_forecast() -> SpreadForecast:
        probs = xr.DataArray(
            np.full((1, side, side), 0.2, dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": lat, "lon": lon, "lead_time_hours": ("time", [24])},
        )
        return SpreadForecast(
            probabilities=probs,
            forecast_reference_time=ref_time,
            horizons_hours=[24],
            model_name="dummy",
            model_version="x",
        )

    model = MagicMock()
    model.predict.side_effect = lambda *_args, **_kwargs: _build_forecast()

    latencies = []
    with patch("ml.spread.service.build_spread_inputs", return_value=mock_inputs):
        for _ in range(5):
            start = time.perf_counter()
            result = run_spread_forecast(request, model=model)
            latencies.append(time.perf_counter() - start)
            assert result.probabilities.shape == (1, side, side)

    p95 = float(np.percentile(np.asarray(latencies, dtype=np.float64), 95))
    assert p95 <= 1.5
