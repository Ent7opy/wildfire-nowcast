import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import numpy as np
import xarray as xr

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
    mock_model = MagicMock()
    mock_model.predict.return_value = mock_forecast
    
    with patch("ml.spread.service.build_spread_inputs", return_value=mock_spread_inputs):
        # Execute
        result = run_spread_forecast(request, model=mock_model)
        
        # Verify
        assert result == mock_forecast
        mock_model.predict.assert_called_once()

def test_run_spread_forecast_aoi_too_large(mock_grid):
    # Setup - Window that exceeds MAX_AOI_CELLS
    # Assuming MAX_AOI_CELLS = 40000, 201x200 = 40200
    side = int(np.sqrt(MAX_AOI_CELLS)) + 10
    lat = np.arange(side)
    lon = np.arange(side)
    large_window = GridWindow(i0=0, i1=side, j0=0, j1=side, lat=lat, lon=lon)
    
    mock_inputs = MagicMock()
    mock_inputs.window = large_window
    
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
            run_spread_forecast(request)
            mock_model.predict.assert_called_once()

