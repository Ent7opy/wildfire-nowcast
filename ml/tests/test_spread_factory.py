"""Tests for the spread model factory."""

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml.spread.factory import get_spread_model, normalize_model_selection
from ml.spread.heuristic_v0 import HeuristicSpreadModelV0
from ml.spread.learned_v1 import LearnedSpreadModelV1
from ml.spread.learned_v2 import LearnedSpreadModelV2


def test_get_spread_model_learned_v1_success():
    """Verify that we can instantiate the learned v1 model."""
    params = {"model_run_dir": "/tmp/test_run"}

    with (
        patch("ml.spread.learned_v1.joblib.load", return_value={}),
        patch("ml.spread.learned_v1.os.path.exists", return_value=True),
        patch("builtins.open", MagicMock()),
        patch("json.load", return_value=["feature1", "feature2"]),
    ):
        model = get_spread_model("LearnedSpreadModelV1", params=params)
        assert isinstance(model, LearnedSpreadModelV1)
        assert model.model_run_dir == "/tmp/test_run"


def test_get_spread_model_learned_v2_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that v2 model can be instantiated via factory."""
    run_dir = tmp_path / "spread_v2_run"
    run_dir.mkdir(parents=True)
    (run_dir / "model.onnx").write_bytes(b"dummy")
    (run_dir / "feature_schema.json").write_text('{"channels": ["fire_t0"]}', encoding="utf-8")

    class _DummySession:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(name="x")]

        def run(self, *_args, **_kwargs):
            raise RuntimeError("not used")

    fake_ort = types.SimpleNamespace(
        SessionOptions=lambda: types.SimpleNamespace(intra_op_num_threads=1),
        InferenceSession=lambda *_args, **_kwargs: _DummySession(),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    model = get_spread_model("LearnedSpreadModelV2", params={"model_run_dir": str(run_dir)})
    assert isinstance(model, LearnedSpreadModelV2)
    assert model.model_run_dir == str(run_dir)


def test_get_spread_model_success():
    """Verify that we can instantiate the known MVP model."""
    model = get_spread_model("HeuristicSpreadModelV0")
    assert isinstance(model, HeuristicSpreadModelV0)
    assert model.config.base_spread_km_h == 0.05


def test_get_spread_model_with_params():
    """Verify that parameters are passed to the model config."""
    params = {"base_spread_km_h": 0.5, "wind_influence_km_h_per_ms": 1.0}
    model = get_spread_model("HeuristicSpreadModelV0", params=params)
    assert isinstance(model, HeuristicSpreadModelV0)
    assert model.config.base_spread_km_h == 0.5
    assert model.config.wind_influence_km_h_per_ms == 1.0


def test_get_spread_model_unknown_model():
    """Verify that asking for a non-existent model raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model: UnknownModel"):
        get_spread_model("UnknownModel")


def test_normalize_model_selection_requires_learned_run_dir():
    with pytest.raises(ValueError, match="model_params.model_run_dir is required"):
        normalize_model_selection("LearnedSpreadModelV1", {})


def test_normalize_model_selection_requires_v2_run_dir():
    with pytest.raises(ValueError, match="model_params.model_run_dir is required"):
        normalize_model_selection("LearnedSpreadModelV2", {})


def test_get_spread_model_filters_unknown_params(caplog: pytest.LogCaptureFixture):
    """Verify that unknown parameters are filtered and warned about."""
    params = {"base_spread_km_h": 0.1, "this_does_not_exist": 123}

    with caplog.at_level(logging.WARNING):
        model = get_spread_model("HeuristicSpreadModelV0", params=params)

    assert isinstance(model, HeuristicSpreadModelV0)
    assert model.config.base_spread_km_h == 0.1
    assert "Ignoring unknown model_params for HeuristicSpreadModelV0" in caplog.text
    assert "this_does_not_exist" in caplog.text
