from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from api.core.grid import GridSpec, GridWindow
from api.fires.service import FireHeatmapWindow
from api.terrain.window import TerrainWindow
from ml.spread_features import (
    SpreadInputs,
    _create_fallback_weather,
    _load_weather_cube,
    build_spread_inputs,
)

@pytest.fixture
def mock_grid():
    return GridSpec(
        crs="EPSG:4326",
        cell_size_deg=0.01,
        origin_lat=35.0,
        origin_lon=5.0,
        n_lat=100,
        n_lon=100
    )

@pytest.fixture
def mock_window():
    # Use linspace to be precise about the number of elements (10 cells)
    lat = np.linspace(35.055, 35.145, 10)
    lon = np.linspace(5.055, 5.145, 10)
    return GridWindow(i0=5, i1=15, j0=5, j1=15, lat=lat, lon=lon)

def test_spread_inputs_to_model_input(mock_grid, mock_window):
    """Verify conversion to the canonical model input contract."""
    fires = MagicMock(spec=FireHeatmapWindow)
    weather = xr.Dataset()
    terrain = MagicMock(spec=TerrainWindow)
    ref_time = datetime.now(timezone.utc)

    inputs = SpreadInputs(
        grid=mock_grid,
        window=mock_window,
        active_fires=fires,
        weather_cube=weather,
        terrain=terrain,
        forecast_reference_time=ref_time,
        horizons_hours=[24, 48]
    )

    model_input = inputs.to_model_input()
    assert model_input.grid == mock_grid
    assert model_input.window == mock_window
    assert model_input.active_fires == fires
    assert model_input.weather_cube is weather
    assert model_input.terrain == terrain
    assert model_input.forecast_reference_time == ref_time
    assert model_input.horizons_hours == [24, 48]

def test_fallback_weather_creation(mock_window):
    """Verify fallback weather has correct shapes and coordinates."""
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
    horizons = [24, 48]

    ds = _create_fallback_weather(mock_window, ref_time, horizons)

    assert dict(ds.sizes) == {"time": 2, "lat": 10, "lon": 10}
    assert "u10" in ds.data_vars
    assert "v10" in ds.data_vars
    assert (ds.u10.values == 0).all()
    assert (ds.v10.values == 0).all()
    assert np.isnan(ds.t2m.values).all()

    # Check coords
    assert len(ds.time) == 2
    assert np.array_equal(ds.lat.values, mock_window.lat)
    assert np.array_equal(ds.lon.values, mock_window.lon)

def _assert_window_equal(a: GridWindow, b: GridWindow) -> None:
    assert (a.i0, a.i1, a.j0, a.j1) == (b.i0, b.i1, b.j0, b.j1)
    assert np.allclose(a.lat, b.lat, rtol=0.0, atol=1e-12)
    assert np.allclose(a.lon, b.lon, rtol=0.0, atol=1e-12)


@patch("ml.spread_features.get_region_grid_spec")
@patch("ml.spread_features.get_grid_window_for_bbox")
@patch("ml.spread_features.get_fire_cells_heatmap")
@patch("ml.spread_features.load_terrain_window")
@patch("ml.spread_features._load_weather_cube")
def test_build_spread_inputs_aoi_to_window_extraction(
    mock_load_weather,
    mock_load_terrain,
    mock_get_fires,
    mock_get_window,
    mock_get_spec,
    mock_grid,
    mock_window,
):
    """AOI/bbox should be converted to a GridWindow and used consistently across components."""
    region = "test_region"
    bbox = (5.055, 35.055, 5.145, 35.145)
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)

    mock_get_spec.return_value = mock_grid
    mock_get_window.return_value = mock_window

    mock_get_fires.return_value = FireHeatmapWindow(
        grid=mock_grid,
        window=mock_window,
        heatmap=np.zeros((10, 10), dtype=np.float32),
    )
    mock_load_terrain.return_value = TerrainWindow(
        window=mock_window,
        slope=np.zeros((10, 10), dtype=np.float32),
        aspect=np.zeros((10, 10), dtype=np.float32),
    )
    mock_load_weather.return_value = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), np.zeros((2, 10, 10), dtype=np.float32)),
            "v10": (("time", "lat", "lon"), np.zeros((2, 10, 10), dtype=np.float32)),
        },
        coords={
            "time": np.array(
                [np.datetime64("2025-12-27T12:00:00", "ns"), np.datetime64("2025-12-28T12:00:00", "ns")]
            ),
            "lat": mock_window.lat,
            "lon": mock_window.lon,
            "lead_time_hours": ("time", [24, 48]),
        },
    )

    inputs = build_spread_inputs(region, bbox, ref_time, horizons_hours=[24, 48])

    mock_get_window.assert_called_once_with(mock_grid, bbox, clip=True)
    _assert_window_equal(inputs.window, mock_window)

    assert inputs.active_fires.heatmap.shape == (10, 10)
    assert inputs.terrain.slope.shape == (10, 10)
    assert dict(inputs.weather_cube.sizes) == {"time": 2, "lat": 10, "lon": 10}
    assert np.allclose(inputs.weather_cube["lat"].values, mock_window.lat, rtol=0.0, atol=1e-12)
    assert np.allclose(inputs.weather_cube["lon"].values, mock_window.lon, rtol=0.0, atol=1e-12)

@patch("ml.spread_features._get_latest_weather_run")
def test_build_spread_inputs_fallback_on_no_weather(mock_get_run, mock_grid, mock_window):
    """Verify fallback behavior when no weather run is found."""
    mock_get_run.return_value = None

    with (
        patch("ml.spread_features.get_region_grid_spec", return_value=mock_grid),
        patch("ml.spread_features.get_grid_window_for_bbox", return_value=mock_window),
        patch("ml.spread_features.get_fire_cells_heatmap") as mock_fires,
        patch("ml.spread_features.load_terrain_window") as mock_terrain,
    ):

        mock_fires.return_value = FireHeatmapWindow(mock_grid, mock_window, np.zeros((10,10)))
        mock_terrain.return_value = TerrainWindow(mock_window, np.zeros((10,10)), np.zeros((10,10)))

        ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
        inputs = build_spread_inputs("region", (5.055, 35.055, 5.145, 35.145), ref_time)

        assert inputs.weather_cube.u10.shape == (3, 10, 10) # default horizons: 24, 48, 72
        assert (inputs.weather_cube.u10.values == 0).all()


@patch("ml.spread_features._get_latest_weather_run")
def test_load_weather_cube_uses_fallback_when_no_run(mock_get_run, mock_window):
    """_load_weather_cube should gracefully fall back when no run exists."""
    mock_get_run.return_value = None
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
    ds = _load_weather_cube(ref_time=ref_time, window=mock_window, horizons_hours=[24, 48], bbox=(0, 0, 1, 1))

    assert dict(ds.sizes) == {"time": 2, "lat": 10, "lon": 10}
    assert (ds["u10"].values == 0).all()
    assert (ds["v10"].values == 0).all()


@patch("ml.spread_features.Path.exists", return_value=True)
@patch("ml.spread_features._get_latest_weather_run")
@patch("xarray.open_dataset")
def test_load_weather_cube_aligns_and_sets_coords(mock_open, mock_get_run, _mock_exists, mock_window):
    """Weather load should align lat/lon/time to the requested window and target horizons."""
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
    mock_get_run.return_value = {"id": 123, "storage_path": "fake.nc", "run_time": ref_time}

    target_times = [
        np.datetime64("2025-12-27T12:00:00", "ns"),
        np.datetime64("2025-12-28T12:00:00", "ns"),
    ]

    w_ds = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), np.ones((2, 10, 10), dtype=np.float32)),
            "v10": (("time", "lat", "lon"), np.ones((2, 10, 10), dtype=np.float32)),
        },
        coords={"time": target_times, "lat": mock_window.lat, "lon": mock_window.lon},
    )
    mock_open.return_value = w_ds

    out = _load_weather_cube(
        ref_time=ref_time,
        window=mock_window,
        horizons_hours=[24, 48],
        bbox=(0, 0, 1, 1),
    )

    assert dict(out.sizes) == {"time": 2, "lat": 10, "lon": 10}
    assert np.array_equal(out["time"].values, np.asarray(target_times))
    assert np.array_equal(out["lead_time_hours"].values, np.asarray([24, 48]))


@patch("ml.spread_features._get_latest_weather_run")
@patch("ml.spread_features.list_fire_detections_bbox_time")
def test_build_spread_inputs_with_region_name_none(mock_list_fires, mock_get_weather_run):
    """Verify build_spread_inputs works with region_name=None (bbox-only mode)."""
    bbox = (20.0, 40.0, 21.0, 41.0)
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)

    mock_list_fires.return_value = []
    mock_get_weather_run.return_value = None

    inputs = build_spread_inputs(
        region_name=None,
        bbox=bbox,
        forecast_reference_time=ref_time,
        horizons_hours=[24, 48],
    )

    assert inputs.grid.crs == "EPSG:4326"
    assert inputs.grid.cell_size_deg == 0.01
    assert inputs.grid.origin_lat == 40.0
    assert inputs.grid.origin_lon == 20.0
    assert inputs.grid.n_lat == 100
    assert inputs.grid.n_lon == 100

    assert inputs.terrain.slope.shape == inputs.window.lat.shape + inputs.window.lon.shape
    assert (inputs.terrain.slope == 0).all()
    assert (inputs.terrain.aspect == 0).all()

    assert inputs.active_fires.heatmap.shape == (
        len(inputs.window.lat),
        len(inputs.window.lon),
    )

    assert "u10" in inputs.weather_cube.data_vars
    assert "v10" in inputs.weather_cube.data_vars

