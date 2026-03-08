"""Tests for hindcast dataset builder."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from api.core.grid import GridSpec, GridWindow
from api.fires.service import FireHeatmapWindow
from api.terrain.window import TerrainWindow
from ml.spread.hindcast_dataset import (
    _flatten_features,
    build_hindcast_dataset,
    sample_fire_reference_times,
    split_hindcast_dataset,
)
from ml.spread_features import SpreadInputs


@pytest.fixture
def mock_grid() -> GridSpec:
    return GridSpec(
        crs="EPSG:4326",
        cell_size_deg=0.01,
        origin_lat=35.0,
        origin_lon=5.0,
        n_lat=100,
        n_lon=100,
    )


@pytest.fixture
def mock_window() -> GridWindow:
    lat = np.linspace(35.05, 35.14, 10)
    lon = np.linspace(5.05, 5.14, 10)
    return GridWindow(i0=5, i1=15, j0=5, j1=15, lat=lat, lon=lon)


@patch("ml.spread.hindcast_dataset.sa.text")
def test_sample_fire_reference_times(mock_text: MagicMock):
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
def test_flatten_features(mock_get_heatmap: MagicMock, mock_build_inputs: MagicMock, mock_grid: GridSpec, mock_window: GridWindow):
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
    horizons = [24]

    fires_t0 = FireHeatmapWindow(mock_grid, mock_window, np.zeros((10, 10), dtype=np.float32))
    weather = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), np.ones((1, 10, 10), dtype=np.float32)),
            "v10": (("time", "lat", "lon"), np.zeros((1, 10, 10), dtype=np.float32)),
            "t2m": (("time", "lat", "lon"), np.full((1, 10, 10), 290.0, dtype=np.float32)),
            "rh2m": (("time", "lat", "lon"), np.full((1, 10, 10), 40.0, dtype=np.float32)),
        },
        coords={"time": [ref_time + timedelta(hours=24)], "lat": mock_window.lat, "lon": mock_window.lon},
    )
    terrain = TerrainWindow(
        window=mock_window,
        slope=np.zeros((10, 10), dtype=np.float32),
        aspect=np.zeros((10, 10), dtype=np.float32),
        elevation=np.zeros((10, 10), dtype=np.float32),
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

    label_heatmap = np.zeros((10, 10), dtype=np.float32)
    label_heatmap[5, 5] = 1.0
    mock_get_heatmap.return_value = FireHeatmapWindow(mock_grid, mock_window, label_heatmap)

    dfs = _flatten_features("region", (5.05, 35.05, 5.14, 35.14), ref_time, horizons)

    assert len(dfs) == 1
    df = dfs[0]
    assert len(df) == 100
    assert "label" in df.columns
    assert int(df["label"].sum()) == 1
    assert "u10" in df.columns
    assert (df["u10"] == 1.0).all()
    assert "fire_t-6h" in df.columns
    assert "region_id_embedding_input" in df.columns


@patch("ml.spread.hindcast_dataset.sample_fire_reference_times")
@patch("ml.spread.hindcast_dataset._flatten_features")
@patch("ml.spread.hindcast_dataset.get_engine")
def test_build_hindcast_dataset_with_negative_sampling(
    mock_get_engine: MagicMock,
    mock_flatten: MagicMock,
    mock_sample: MagicMock,
):
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
    mock_sample.return_value = [ref_time]

    data = {
        "ref_time": [ref_time] * 100,
        "horizon_h": [24] * 100,
        "lat": [1.0] * 100,
        "lon": [1.0] * 100,
        "fire_t0": [0] * 100,
        "label": [1] * 2 + [0] * 98,
        "region_bucket": [0] * 100,
        "ref_year": [2025] * 100,
    }
    mock_flatten.return_value = [pd.DataFrame(data)]

    df = build_hindcast_dataset("region", (0, 0, 1, 1), ref_time, ref_time, [24], negative_ratio=5.0)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert int(df["label"].sum()) == 2


@patch("ml.spread.hindcast_dataset.sample_fire_reference_times")
@patch("ml.spread.hindcast_dataset._flatten_features")
@patch("ml.spread.hindcast_dataset.get_engine")
def test_build_hindcast_dataset_tensor_mode(
    mock_get_engine: MagicMock,
    mock_flatten: MagicMock,
    mock_sample: MagicMock,
):
    ref_time = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
    mock_sample.return_value = [ref_time]

    lat_vals = np.array([35.0, 35.1], dtype=float)
    lon_vals = np.array([5.0, 5.1], dtype=float)
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")

    base = {
        "ref_time": [ref_time] * 4,
        "horizon_h": [24] * 4,
        "lat": lat_grid.ravel(),
        "lon": lon_grid.ravel(),
        "label": np.array([1, 0, 0, 1], dtype=np.int8),
        "region_name": ["region"] * 4,
        "region_bucket": [0] * 4,
        "fire_t0": np.ones(4),
        "fire_t-6h": np.ones(4),
        "fire_t-12h": np.ones(4),
        "u10": np.ones(4),
        "v10": np.zeros(4),
        "t2m": np.ones(4),
        "rh2m": np.ones(4),
        "precip_24h": np.zeros(4),
        "slope_deg": np.zeros(4),
        "aspect_sin": np.zeros(4),
        "aspect_cos": np.ones(4),
        "elevation_m": np.zeros(4),
        "ruggedness": np.zeros(4),
        "tpi": np.zeros(4),
        "ndvi": np.zeros(4),
        "lfmc": np.zeros(4),
        "dfmc": np.zeros(4),
        "region_id_embedding_input": np.zeros(4),
        "ref_year": [2025] * 4,
    }
    mock_flatten.return_value = [pd.DataFrame(base)]

    out = build_hindcast_dataset(
        "region",
        (0, 0, 1, 1),
        ref_time,
        ref_time,
        [24],
        negative_ratio=None,
        output_mode="tensor",
    )
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["x_tensor"].ndim == 3
    assert out[0]["y_tensor"].shape == (2, 2)


def test_split_hindcast_dataset_year_and_region_bucket():
    df = pd.DataFrame(
        {
            "ref_time": pd.to_datetime(
                [
                    "2024-07-01T00:00:00Z",
                    "2024-08-01T00:00:00Z",
                    "2025-07-01T00:00:00Z",
                    "2025-08-01T00:00:00Z",
                ],
                utc=True,
            ),
            "ref_year": [2024, 2024, 2025, 2025],
            "region_bucket": [1, 0, 1, 0],
            "lat": [1, 2, 3, 4],
            "lon": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
        }
    )

    train_df, eval_df = split_hindcast_dataset(
        df,
        split_year=2025,
        validation_region_buckets={0},
    )

    assert not train_df.empty
    assert not eval_df.empty
    assert (eval_df["ref_year"] >= 2025).all()
    assert (eval_df["region_bucket"] == 0).all()
