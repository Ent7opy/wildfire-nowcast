import sys
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr

from api.core.grid import GridSpec, GridWindow
from api.fires.service import FireHeatmapWindow
from api.terrain.window import TerrainWindow
from ml.spread.contract import SpreadModelInput
from ml.spread.learned_v3 import LearnedSpreadModelV3


def test_v3_uses_real_lag_features_not_fire_t0_copies(tmp_path, monkeypatch):
    run_dir = tmp_path / "spread_v3_run"
    run_dir.mkdir(parents=True)
    (run_dir / "model.onnx").write_bytes(b"dummy")
    (run_dir / "feature_schema.json").write_text(
        '{"channels":["fire_t0","fire_t-6h","fire_t-12h","region_id_embedding_input"],"weather_aggregation":"horizon_weighted_mean"}',
        encoding="utf-8",
    )

    class _DummySession:
        def get_inputs(self):
            return [types.SimpleNamespace(name="x")]

        def run(self, *_args, **_kwargs):
            return [np.zeros((1, 1, 2, 2), dtype=np.float32)]

    fake_ort = types.SimpleNamespace(
        SessionOptions=lambda: types.SimpleNamespace(intra_op_num_threads=1),
        InferenceSession=lambda *_args, **_kwargs: _DummySession(),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    grid = GridSpec.from_bbox((20.0, 40.0, 20.02, 40.02))
    lat = np.array([40.005, 40.015], dtype=np.float64)
    lon = np.array([20.005, 20.015], dtype=np.float64)
    window = GridWindow(i0=0, i1=2, j0=0, j1=2, lat=lat, lon=lon)
    fire_t0 = np.full((2, 2), 0.9, dtype=np.float32)
    terrain = TerrainWindow(
        window=window,
        slope=np.zeros((2, 2), dtype=np.float32),
        aspect=np.zeros((2, 2), dtype=np.float32),
        elevation=np.zeros((2, 2), dtype=np.float32),
    )
    weather = xr.Dataset(
        data_vars={},
        coords={
            "time": [datetime(2026, 2, 11, tzinfo=timezone.utc) + timedelta(hours=24)],
            "lat": lat,
            "lon": lon,
        },
    )
    model_input = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=FireHeatmapWindow(grid=grid, window=window, heatmap=fire_t0),
        weather_cube=weather,
        terrain=terrain,
        forecast_reference_time=datetime(2026, 2, 11, tzinfo=timezone.utc),
        horizons_hours=[24],
    )

    def _mock_lag(*_args, **kwargs):
        span_h = int((kwargs["end_time"] - kwargs["start_time"]).total_seconds() / 3600.0)
        value = 0.6 if span_h == 6 else 0.2
        return FireHeatmapWindow(grid=grid, window=window, heatmap=np.full((2, 2), value, dtype=np.float32))

    monkeypatch.setattr("ml.spread.learned_v3.get_fire_cells_heatmap", _mock_lag)

    model = LearnedSpreadModelV3(model_run_dir=str(run_dir))
    x = model._build_feature_tensor(model_input)

    assert np.allclose(x[0, 0], 0.9)
    assert np.allclose(x[0, 1], 0.6)
    assert np.allclose(x[0, 2], 0.2)


def test_v3_predict_pads_odd_spatial_shapes_for_onnx(tmp_path, monkeypatch):
    run_dir = tmp_path / "spread_v3_run"
    run_dir.mkdir(parents=True)
    (run_dir / "model.onnx").write_bytes(b"dummy")
    (run_dir / "feature_schema.json").write_text(
        '{"channels":["fire_t0","fire_t-6h","fire_t-12h","region_id_embedding_input"],"weather_aggregation":"horizon_weighted_mean"}',
        encoding="utf-8",
    )

    class _DummySession:
        def __init__(self) -> None:
            self.last_input_shape: tuple[int, ...] | None = None

        def get_inputs(self):
            return [types.SimpleNamespace(name="x")]

        def run(self, _output_names, feed):
            x = np.asarray(feed["x"], dtype=np.float32)
            self.last_input_shape = tuple(int(v) for v in x.shape)
            return [np.full((1, 1, x.shape[2], x.shape[3]), 0.5, dtype=np.float32)]

    dummy_session = _DummySession()
    fake_ort = types.SimpleNamespace(
        SessionOptions=lambda: types.SimpleNamespace(intra_op_num_threads=1),
        InferenceSession=lambda *_args, **_kwargs: dummy_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    grid = GridSpec(
        crs="EPSG:4326",
        cell_size_deg=0.01,
        origin_lat=40.0,
        origin_lon=20.0,
        n_lat=23,
        n_lon=22,
    )
    lat = np.linspace(40.005, 40.225, 23, dtype=np.float64)
    lon = np.linspace(20.005, 20.215, 22, dtype=np.float64)
    window = GridWindow(i0=0, i1=23, j0=0, j1=22, lat=lat, lon=lon)
    fire = np.full((23, 22), 0.3, dtype=np.float32)
    terrain = TerrainWindow(
        window=window,
        slope=np.zeros((23, 22), dtype=np.float32),
        aspect=np.zeros((23, 22), dtype=np.float32),
        elevation=np.zeros((23, 22), dtype=np.float32),
    )
    weather = xr.Dataset(
        data_vars={},
        coords={
            "time": [datetime(2026, 2, 11, tzinfo=timezone.utc) + timedelta(hours=24)],
            "lat": lat,
            "lon": lon,
        },
    )
    model_input = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=FireHeatmapWindow(grid=grid, window=window, heatmap=fire),
        weather_cube=weather,
        terrain=terrain,
        forecast_reference_time=datetime(2026, 2, 11, tzinfo=timezone.utc),
        horizons_hours=[24],
    )

    monkeypatch.setattr(
        "ml.spread.learned_v3.get_fire_cells_heatmap",
        lambda *_args, **_kwargs: FireHeatmapWindow(
            grid=grid, window=window, heatmap=np.zeros((23, 22), dtype=np.float32)
        ),
    )

    model = LearnedSpreadModelV3(model_run_dir=str(run_dir))
    forecast = model.predict(model_input)

    assert dummy_session.last_input_shape is not None
    assert dummy_session.last_input_shape[2] % 16 == 0
    assert dummy_session.last_input_shape[3] % 16 == 0
    assert dummy_session.last_input_shape[2] >= 23
    assert dummy_session.last_input_shape[3] >= 22
    assert tuple(forecast.probabilities.shape) == (1, 23, 22)
