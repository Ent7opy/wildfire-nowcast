"""Tests for hindcast builder."""

import json
import warnings
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from api.core.grid import GridSpec, GridWindow
from api.fires.service import FireHeatmapWindow
from api.terrain.window import TerrainWindow
from ml.spread.contract import SpreadForecast
from ml.spread.hindcast_builder import (
    build_hindcast_case,
    group_reference_times_into_events,
    run_hindcast_builder,
)
from ml.spread_features import SpreadInputs


@pytest.fixture
def mock_grid():
    return GridSpec(
        crs="EPSG:4326",
        cell_size_deg=0.01,
        origin_lat=35.0,
        origin_lon=5.0,
        n_lat=100,
        n_lon=100,
    )


@pytest.fixture
def mock_window():
    lat = np.linspace(35.05, 35.14, 10)
    lon = np.linspace(5.05, 5.14, 10)
    return GridWindow(i0=5, i1=15, j0=5, j1=15, lat=lat, lon=lon)


def test_group_reference_times_into_events():
    t0 = datetime(2025, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Contiguous
    times = [t0, t0 + timedelta(hours=6), t0 + timedelta(hours=12)]
    # Gap
    times.append(t0 + timedelta(hours=36))
    times.append(t0 + timedelta(hours=42))

    events = group_reference_times_into_events(times, interval_hours=6)
    assert len(events) == 2
    assert len(events[0]) == 3
    assert len(events[1]) == 2
    assert events[0][-1] == t0 + timedelta(hours=12)
    assert events[1][0] == t0 + timedelta(hours=36)


@patch("ml.spread.hindcast_builder.build_spread_inputs")
@patch("ml.spread.hindcast_builder.get_fire_cells_heatmap")
def test_build_hindcast_case(mock_get_heatmap, mock_build_inputs, mock_grid, mock_window):
    ref_time = datetime(2025, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    horizons = [24, 48]

    # Mock SpreadInputs
    fires_t0 = FireHeatmapWindow(mock_grid, mock_window, np.ones((10, 10)))
    weather = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), np.ones((2, 10, 10))),
            "v10": (("time", "lat", "lon"), np.zeros((2, 10, 10))),
        },
        coords={
            "time": [ref_time + timedelta(hours=h) for h in horizons],
            "lat": mock_window.lat,
            "lon": mock_window.lon,
        },
    )
    terrain = TerrainWindow(
        window=mock_window,
        slope=np.zeros((10, 10)),
        aspect=np.zeros((10, 10)),
        elevation=np.zeros((10, 10)),
    )

    mock_build_inputs.return_value = SpreadInputs(
        grid=mock_grid,
        window=mock_window,
        active_fires=fires_t0,
        weather_cube=weather,
        terrain=terrain,
        forecast_reference_time=ref_time,
        horizons_hours=horizons,
    )

    # Mock Model
    mock_model = MagicMock()
    probs = xr.DataArray(
        np.zeros((2, 10, 10)),
        dims=("time", "lat", "lon"),
        coords={
            "time": [ref_time + timedelta(hours=h) for h in horizons],
            "lat": mock_window.lat,
            "lon": mock_window.lon,
        },
    )
    mock_model.predict.return_value = SpreadForecast(
        probabilities=probs,
        forecast_reference_time=ref_time,
        horizons_hours=horizons,
    )

    # Mock Observations
    mock_get_heatmap.return_value = FireHeatmapWindow(
        mock_grid, mock_window, np.ones((10, 10))
    )

    ds = build_hindcast_case(
        "region", (0, 0, 1, 1), ref_time, horizons, mock_model, label_window_hours=3
    )

    assert "y_pred" in ds.data_vars
    assert "y_obs" in ds.data_vars
    assert ds.y_pred.shape == (2, 10, 10)
    assert ds.y_obs.shape == (2, 10, 10)
    assert (ds.y_obs == 1.0).all()


@patch("ml.spread.hindcast_builder.sample_fire_reference_times")
@patch("ml.spread.hindcast_builder.build_hindcast_case")
@patch("ml.spread.hindcast_builder.get_engine")
def test_run_hindcast_builder(
    mock_get_engine, mock_build_case, mock_sample, tmp_path, mock_window
):
    ref_time = datetime(2025, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_sample.return_value = [ref_time]

    # Suppress the DeprecationWarning about parsing timezone aware datetimes
    warnings.filterwarnings("ignore", category=DeprecationWarning, message="parsing timezone aware datetimes is deprecated")

    # Mock Case
    ds = xr.Dataset(
        data_vars={
            "y_pred": (["time", "lat", "lon"], np.zeros((1, 10, 10))),
            "y_obs": (["time", "lat", "lon"], np.zeros((1, 10, 10))),
            "fire_t0": (["lat", "lon"], np.ones((10, 10))),
            "slope": (["lat", "lon"], np.zeros((10, 10))),
            "aspect": (["lat", "lon"], np.zeros((10, 10))),
        },
        coords={
            "time": [np.datetime64(ref_time + timedelta(hours=24), "ns")],
            "lat": mock_window.lat,
            "lon": mock_window.lon,
        },
    )
    mock_build_case.return_value = ds

    config = {
        "region_name": "smoke_grid",
        "bbox": [22.0, 40.0, 24.0, 42.0],
        "start_time": "2025-07-01",
        "end_time": "2025-08-31",
        "horizons_hours": [24],
        "output_root": str(tmp_path),
        "min_detections": 1,
        "interval_hours": 6,
        "min_active_cells_t0": 1,
        "min_event_buckets": 1,
    }

    run_hindcast_builder(config)

    # Check output
    runs = list(tmp_path.glob("run_*"))
    assert len(runs) == 1
    run_dir = runs[0]
    
    event_dirs = list(run_dir.glob("event_*"))
    assert len(event_dirs) == 1
    event_dir = event_dirs[0]

    nc_files = list(event_dir.glob("*.nc"))
    assert len(nc_files) == 1
    
    manifest_path = run_dir / "index.json"
    assert manifest_path.exists()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        assert len(manifest["cases"]) == 1
        assert manifest["cases"][0]["event_id"] == "event_000"


@patch("ml.spread.hindcast_builder.sample_fire_reference_times")
@patch("ml.spread.hindcast_builder.build_hindcast_case")
@patch("ml.spread.hindcast_builder.get_engine")
def test_run_hindcast_builder_relative_output_root(
    mock_get_engine, mock_build_case, mock_sample, tmp_path, mock_window, monkeypatch
):
    """
    Regression: when output_root is relative, manifest path computation must not raise
    ValueError by mixing relative/absolute paths in Path.relative_to().
    """
    # Run from an absolute cwd, but use a relative output_root.
    monkeypatch.chdir(tmp_path)

    ref_time = datetime(2025, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_sample.return_value = [ref_time]

    # Suppress the DeprecationWarning about parsing timezone aware datetimes
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="parsing timezone aware datetimes is deprecated",
    )

    # Mock Case
    ds = xr.Dataset(
        data_vars={
            "y_pred": (["time", "lat", "lon"], np.zeros((1, 10, 10))),
            "y_obs": (["time", "lat", "lon"], np.zeros((1, 10, 10))),
            "fire_t0": (["lat", "lon"], np.ones((10, 10))),
            "slope": (["lat", "lon"], np.zeros((10, 10))),
            "aspect": (["lat", "lon"], np.zeros((10, 10))),
        },
        coords={
            "time": [np.datetime64(ref_time + timedelta(hours=24), "ns")],
            "lat": mock_window.lat,
            "lon": mock_window.lon,
        },
    )
    mock_build_case.return_value = ds

    config = {
        "region_name": "smoke_grid",
        "bbox": [22.0, 40.0, 24.0, 42.0],
        "start_time": "2025-07-01",
        "end_time": "2025-08-31",
        "horizons_hours": [24],
        # Intentionally relative (matches real default patterns).
        "output_root": "data/hindcasts/smoke_grid",
        "min_detections": 1,
        "interval_hours": 6,
        "min_active_cells_t0": 1,
        "min_event_buckets": 1,
    }

    run_hindcast_builder(config)

    # Check output exists under tmp_path/cwd
    runs = list((tmp_path / "data" / "hindcasts" / "smoke_grid").glob("run_*"))
    assert len(runs) == 1
    run_dir = runs[0]

    manifest_path = run_dir / "index.json"
    assert manifest_path.exists()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    assert len(manifest["cases"]) == 1
    # Manifest path should be a relative path string (not absolute).
    assert manifest["cases"][0]["path"].startswith("data/hindcasts/smoke_grid/")


@patch("ml.spread.hindcast_builder.get_spread_model")
@patch("ml.spread.hindcast_builder.sample_fire_reference_times")
@patch("ml.spread.hindcast_builder.build_hindcast_case")
@patch("ml.spread.hindcast_builder.get_engine")
def test_run_hindcast_builder_with_params(
    mock_get_engine, mock_build_case, mock_sample, mock_get_model, tmp_path, mock_window
):
    ref_time = datetime(2025, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_sample.return_value = [ref_time]

    # Mock Model Instance
    mock_model_instance = MagicMock()
    mock_get_model.return_value = mock_model_instance

    config = {
        "region_name": "smoke_grid",
        "bbox": [22.0, 40.0, 24.0, 42.0],
        "start_time": "2025-07-01",
        "end_time": "2025-08-31",
        "horizons_hours": [24],
        "output_root": str(tmp_path),
        "min_detections": 1,
        "interval_hours": 6,
        "model_params": {
            "base_spread_km_h": 0.1,
            "wind_influence_km_h_per_ms": 0.2
        }
    }

    run_hindcast_builder(config)

    # Verify factory call
    mock_get_model.assert_called_once()
    args = mock_get_model.call_args[0]
    assert args[0] == "HeuristicSpreadModelV0"
    assert args[1]["base_spread_km_h"] == 0.1
    assert args[1]["wind_influence_km_h_per_ms"] == 0.2