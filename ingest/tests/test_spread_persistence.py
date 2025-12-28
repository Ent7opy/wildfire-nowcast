import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, shape

from ingest.spread_forecast import generate_contours, save_forecast_rasters
from api.core.grid import GridSpec


def test_generate_contours_simple_square():
    # 3x3 grid
    #  0 0 0
    #  0 1 0
    #  0 0 0
    data = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    # Grid: origin (0, 30), pixel size 10 deg.
    # Row 0: y=30..20
    # Row 1: y=20..10 -> Center pixel is x=10..20, y=10..20
    # Row 2: y=10..0
    transform = from_origin(0, 30, 10, 10)
    
    thresholds = [0.5]
    contours = generate_contours(data, transform, thresholds)
    
    assert len(contours) == 1
    c = contours[0]
    assert c["threshold"] == 0.5
    
    # Parse GeoJSON
    geom = shape(json.loads(c["geom_geojson"]))
    
    assert isinstance(geom, MultiPolygon)
    assert not geom.is_empty
    # Area of one 10x10 pixel is 100
    assert abs(geom.area - 100.0) < 1e-6
    
    # Check bounds
    minx, miny, maxx, maxy = geom.bounds
    assert minx == 10
    assert maxx == 20
    assert miny == 10
    assert maxy == 20


def test_generate_contours_empty():
    data = np.zeros((3, 3), dtype=np.float32)
    transform = from_origin(0, 30, 10, 10)
    
    thresholds = [0.5]
    contours = generate_contours(data, transform, thresholds)
    
    assert len(contours) == 1
    c = contours[0]
    geom = shape(json.loads(c["geom_geojson"]))
    assert geom.is_empty


@patch("rasterio.open")
@patch("ingest.spread_forecast.convert_to_cog")
def test_save_forecast_rasters(mock_convert, mock_open):
    # Mock inputs
    forecast = MagicMock()
    forecast.horizons_hours = [24]
    # Mock probability data
    forecast.probabilities.sel.return_value.values = np.zeros((10, 10))
    # We also need to mock the _select_probability_slice_by_horizon call
    # Since it's an internal function, we can mock it inside the test or rely on the mock of .sel/.isel
    # But wait, save_forecast_rasters calls _select_probability_slice_by_horizon.
    # Let's mock _select_probability_slice_by_horizon directly if possible, or just mock the forecast object enough.
    
    # The function uses:
    # data = _select_probability_slice_by_horizon(forecast, int(h))
    # which uses forecast.probabilities.coords...
    
    # Simpler: mock ingest.spread_forecast._select_probability_slice_by_horizon
    with patch("ingest.spread_forecast._select_probability_slice_by_horizon") as mock_select:
        mock_select.return_value = np.zeros((10, 10), dtype=np.float32)
        
        grid = GridSpec(
            origin_lat=0,
            origin_lon=0,
            n_lat=10,
            n_lon=10,
            cell_size_deg=1.0
        )
        
        # Use a Mock for run_dir so we don't hit the filesystem
        run_dir = MagicMock()
        
        # Mock the path division operator /
        out_path_mock = MagicMock()
        run_dir.__truediv__.return_value = out_path_mock
        
        # Mock unlink
        out_path_mock.unlink.return_value = None
        
        # Mock convert_to_cog return value
        # It should be a path-like object that supports relative_to
        final_path_mock = MagicMock()
        final_path_mock.relative_to.return_value = Path("data/forecasts/run_1/spread_h024_cog.tif")
        mock_convert.return_value = final_path_mock

        # Mock file writing
        mock_dst = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_dst
        
        results = save_forecast_rasters(forecast, grid, run_dir, emit_cog=True)
        
        assert len(results) == 1
        assert results[0]["horizon_hours"] == 24
        assert results[0]["file_format"] == "COG"
        assert str(results[0]["storage_path"]) == str(Path("data/forecasts/run_1/spread_h024_cog.tif"))
        
        mock_convert.assert_called_once()
        # Verify unlink was called on the intermediate file
        out_path_mock.unlink.assert_called_once()

