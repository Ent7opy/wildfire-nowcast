import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, shape
import xarray as xr

from ingest.spread_forecast import (
    _select_probability_slice_by_horizon,
    build_contour_records,
    generate_contours,
    save_forecast_rasters,
)
from api.core.grid import GridSpec, GridWindow


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

    # Provide minimal, contract-like lat/lon coordinates (cell centers) on the window.
    lat = (np.arange(10, dtype=float) + 0.5).astype(np.float64)
    lon = (np.arange(10, dtype=float) + 0.5).astype(np.float64)
    forecast.probabilities = xr.DataArray(
        np.zeros((1, 10, 10), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": lat, "lon": lon},
    )

    # Mock the internal selector so we don't depend on time/lead_time selection logic.
    with patch("ingest.spread_forecast._select_probability_slice_by_horizon") as mock_select:
        mock_select.return_value = np.zeros((10, 10), dtype=np.float32)
        
        grid = GridSpec(
            origin_lat=0,
            origin_lon=0,
            n_lat=10,
            n_lon=10,
            cell_size_deg=1.0
        )
        window = GridWindow(i0=0, i1=10, j0=0, j1=10, lat=lat, lon=lon)
        
        # Use a Mock for run_dir so we don't hit the filesystem
        run_dir = MagicMock()
        run_dir.mkdir.return_value = None
        
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
        
        results = save_forecast_rasters(forecast, grid, window, run_dir, emit_cog=True)
        
        assert len(results) == 1
        assert results[0]["horizon_hours"] == 24
        assert results[0]["file_format"] == "COG"
        assert str(results[0]["storage_path"]) == str(Path("data/forecasts/run_1/spread_h024_cog.tif"))
        
        mock_convert.assert_called_once()
        # Verify unlink was called on the intermediate file
        out_path_mock.unlink.assert_called_once()


@patch("rasterio.open")
@patch("ingest.spread_forecast.convert_to_cog")
def test_save_forecast_rasters_uses_forecast_coord_transform_fallback(mock_convert, mock_open):
    """Regression: if raster saving falls back to forecast-coord transform, contours must too.

    We assert the *raster* path uses the fallback transform when forecast coords differ
    from the requested window coords, which is the precondition for the raster/contour
    misalignment bug.
    """
    forecast = MagicMock()
    forecast.horizons_hours = [24]

    # Window coordinates (cell centers)
    window_lat = (np.arange(10, dtype=float) + 0.5).astype(np.float64)
    window_lon = (np.arange(10, dtype=float) + 0.5).astype(np.float64)

    # Forecast coordinates are slightly shifted (same shape, but not allclose at atol=1e-12)
    eps = 1e-6
    forecast_lat = window_lat + eps
    forecast_lon = window_lon + eps

    forecast.probabilities = xr.DataArray(
        np.zeros((1, 10, 10), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": forecast_lat, "lon": forecast_lon},
    )

    with patch("ingest.spread_forecast._select_probability_slice_by_horizon") as mock_select:
        mock_select.return_value = np.zeros((10, 10), dtype=np.float32)

        grid = GridSpec(origin_lat=0, origin_lon=0, n_lat=10, n_lon=10, cell_size_deg=1.0)
        window = GridWindow(i0=0, i1=10, j0=0, j1=10, lat=window_lat, lon=window_lon)

        run_dir = MagicMock()
        run_dir.mkdir.return_value = None
        out_path_mock = MagicMock()
        run_dir.__truediv__.return_value = out_path_mock
        out_path_mock.unlink.return_value = None

        final_path_mock = MagicMock()
        final_path_mock.relative_to.return_value = Path("data/forecasts/run_1/spread_h024_cog.tif")
        mock_convert.return_value = final_path_mock

        save_forecast_rasters(forecast, grid, window, run_dir, emit_cog=True)

        # The call is: rasterio.open(out_path, "w", **profile)
        _, _, kwargs = mock_open.mock_calls[0]
        transform = kwargs["transform"]

        # Expected fallback transform derived from forecast coords:
        # dx=dy=1.0, west_edge=lon.min()-0.5, north_edge=lat.max()+0.5
        expected = from_origin(west=float(forecast_lon.min() - 0.5), north=float(forecast_lat.max() + 0.5), xsize=1.0, ysize=1.0)
        assert transform == expected


def test_select_probability_slice_by_horizon_fallback_exact_match():
    """Fallback path (time-based) should select an exact time match."""
    ref_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # time coordinate is naive UTC (matches helper behavior)
    t24 = np.datetime64(datetime(2025, 1, 2), "ns")

    forecast = MagicMock()
    forecast.forecast_reference_time = ref_time
    forecast.probabilities = xr.DataArray(
        np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": [t24], "lat": [0.5, 1.5], "lon": [10.5, 11.5]},
    )

    out = _select_probability_slice_by_horizon(forecast, 24)
    assert out.shape == (2, 2)
    assert np.allclose(out, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_select_probability_slice_by_horizon_fallback_raises_on_missing_horizon():
    """Regression: never silently snap to nearest time for missing horizons."""
    ref_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t24 = np.datetime64(datetime(2025, 1, 2), "ns")

    forecast = MagicMock()
    forecast.forecast_reference_time = ref_time
    forecast.probabilities = xr.DataArray(
        np.zeros((1, 2, 2), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": [t24], "lat": [0.5, 1.5], "lon": [10.5, 11.5]},
    )

    with pytest.raises(ValueError, match=r"Requested horizon_hours=25"):
        _select_probability_slice_by_horizon(forecast, 25)


def test_build_contour_records_uses_forecast_horizons_only():
    forecast = MagicMock()
    forecast.horizons_hours = [24]  # model output horizons
    forecast.forecast_reference_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    lat = (np.arange(2, dtype=float) + 0.5).astype(np.float64)
    lon = (np.arange(2, dtype=float) + 0.5).astype(np.float64)
    forecast.probabilities = xr.DataArray(
        np.zeros((1, 2, 2), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": lat, "lon": lon},
    )

    grid = GridSpec(origin_lat=0, origin_lon=0, n_lat=2, n_lon=2, cell_size_deg=1.0)
    window = GridWindow(i0=0, i1=2, j0=0, j1=2, lat=lat, lon=lon)

    with patch("ingest.spread_forecast._select_probability_slice_by_horizon") as mock_select, patch(
        "ingest.spread_forecast.generate_contours"
    ) as mock_gen:
        mock_select.return_value = np.zeros((2, 2), dtype=np.float32)
        mock_gen.return_value = [{"threshold": 0.5, "geom_geojson": '{"type":"MultiPolygon","coordinates":[]}'}]

        records = build_contour_records(
            forecast=forecast, grid=grid, window=window, thresholds=[0.5]
        )

    assert len(records) == 1
    assert records[0]["horizon_hours"] == 24
    assert records[0]["threshold"] == 0.5

