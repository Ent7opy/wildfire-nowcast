"""Tests for hindcast dataset builder."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from ml.spread.hindcast_dataset import sample_fire_reference_times, _flatten_features, build_hindcast_dataset
from ml.spread_features import SpreadInputs
from api.fires.service import FireHeatmapWindow
from api.terrain.window import TerrainWindow
from api.core.grid import GridSpec, GridWindow
import xarray as xr

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
    lat = np.linspace(35.05, 35.14, 10)
    lon = np.linspace(5.05, 5.14, 10)
    return GridWindow(i0=5, i1=15, j0=5, j1=15, lat=lat, lon=lon)

@patch("ml.spread.hindcast_dataset.sa.text")
def test_sample_fire_reference_times(mock_text):
    mock_engine = MagicMock()
    mock_conn = mock_engine.connect.return_value.__enter__.return_value
    
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
    mock_conn.execute.return_value.mappings.return_value.all.return_value = [
        {"ref_time": ref_time, "detection_count": 10}
    ]
    
    bbox = (5.0, 35.0, 6.0, 36.0)
    times = sample_fire_reference_times(
        mock_engine, bbox, ref_time - timedelta(days=1), ref_time, min_detections=5
    )
    
    assert len(times) == 1
    assert times[0] == ref_time
    assert times[0].tzinfo == timezone.utc

@patch("ml.spread.hindcast_dataset.build_spread_inputs")
@patch("ml.spread.hindcast_dataset.get_fire_cells_heatmap")
def test_flatten_features(mock_get_heatmap, mock_build_inputs, mock_grid, mock_window):
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
    horizons = [24]
    
    # Mock SpreadInputs
    fires_t0 = FireHeatmapWindow(mock_grid, mock_window, np.zeros((10, 10)))
    weather = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), np.ones((1, 10, 10))),
            "v10": (("time", "lat", "lon"), np.zeros((1, 10, 10))),
        },
        coords={"time": [ref_time + timedelta(hours=24)], "lat": mock_window.lat, "lon": mock_window.lon}
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
        horizons_hours=horizons
    )
    
    # Mock future heatmap (label)
    label_heatmap = np.zeros((10, 10))
    label_heatmap[5, 5] = 1.0
    mock_get_heatmap.return_value = FireHeatmapWindow(mock_grid, mock_window, label_heatmap)
    
    dfs = _flatten_features("region", (5.05, 35.05, 5.14, 35.14), ref_time, horizons)
    
    assert len(dfs) == 1
    df = dfs[0]
    assert len(df) == 100
    assert "label" in df.columns
    assert df["label"].sum() == 1
    assert "u10" in df.columns
    assert (df["u10"] == 1.0).all()

@patch("ml.spread.hindcast_dataset.sample_fire_reference_times")
@patch("ml.spread.hindcast_dataset._flatten_features")
@patch("ml.spread.hindcast_dataset.get_engine")
def test_build_hindcast_dataset_with_negative_sampling(mock_get_engine, mock_flatten, mock_sample):
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
    mock_sample.return_value = [ref_time]
    
    # Create a dummy DF with 2 positive and 98 negative samples
    data = {
        "ref_time": [ref_time] * 100,
        "horizon_h": [24] * 100,
        "fire_t0": [0] * 100,
        "label": [1] * 2 + [0] * 98,
    }
    mock_flatten.return_value = [pd.DataFrame(data)]
    
    # negative_ratio = 5.0 -> 2 pos * 5 = 10 neg expected
    df = build_hindcast_dataset(
        "region", (0, 0, 1, 1), ref_time, ref_time, [24], negative_ratio=5.0
    )
    
    assert len(df) == 12 # 2 pos + 10 neg
    assert df["label"].sum() == 2

