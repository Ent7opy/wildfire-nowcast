"""Integration tests: spread feature contract enforcement (train vs. infer).

Done criteria:
  - Contract schema exists (runtime_contract.py)
  - Integration test passes (this file)
  - Gate report rejects on mismatch (STOP-CONTRACT-001)

Tests in this file use no DB, no ONNX session, and no external services.
They operate purely on channel lists and tensor shapes.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from ml.spread.runtime_contract import (
    CANONICAL_V2_CHANNELS,
    ContractViolationError,
    SpreadRuntimeContract,
    load_contract,
    validate_channel_alignment,
    validate_feature_tensor,
    write_contract,
)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeGridWindow:
    lat: np.ndarray
    lon: np.ndarray


@dataclass(frozen=True)
class _FakeFireHeatmap:
    heatmap: np.ndarray


@dataclass(frozen=True)
class _FakeTerrain:
    slope: np.ndarray
    aspect: np.ndarray
    elevation: np.ndarray | None = None


def _make_spread_model_input(ny: int = 8, nx: int = 8):
    """Minimal SpreadModelInput-compatible object for _build_feature_tensor."""
    lat = np.linspace(30.0, 31.0, ny, dtype=np.float32)
    lon = np.linspace(-120.0, -119.0, nx, dtype=np.float32)
    zeros = np.zeros((ny, nx), dtype=np.float32)
    weather_cube = xr.Dataset(
        {
            "u10": (["lat", "lon"], zeros.copy()),
            "v10": (["lat", "lon"], zeros.copy()),
            "t2m": (["lat", "lon"], zeros.copy()),
            "rh2m": (["lat", "lon"], zeros.copy()),
            "precip_24h": (["lat", "lon"], zeros.copy()),
            "ndvi": (["lat", "lon"], zeros.copy()),
            "lfmc": (["lat", "lon"], zeros.copy()),
            "dfmc": (["lat", "lon"], zeros.copy()),
        },
        coords={"lat": lat, "lon": lon},
    )
    return types.SimpleNamespace(
        active_fires=_FakeFireHeatmap(heatmap=zeros.copy()),
        terrain=_FakeTerrain(slope=zeros.copy(), aspect=zeros.copy(), elevation=zeros.copy()),
        weather_cube=weather_cube,
        horizons_hours=[2, 6, 12],
        window=_FakeGridWindow(lat=lat, lon=lon),
        forecast_reference_time=datetime.now(timezone.utc),
        fire_history_t6h=None,
        fire_history_t12h=None,
    )


class _DummyOrt:
    """Minimal onnxruntime stand-in for tests that exercise model init but not inference."""

    class SessionOptions:
        intra_op_num_threads = 1

    class InferenceSession:
        def __init__(self, *a, **kw):
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(name="x")]


# ---------------------------------------------------------------------------
# Test 1: Channel list identity — hindcast dataset must mirror canonical contract
# ---------------------------------------------------------------------------


def test_hindcast_channels_match_canonical():
    """V2_TENSOR_CHANNELS in hindcast_dataset.py must equal CANONICAL_V2_CHANNELS.

    If this fails, someone redefined the channel list locally instead of
    importing from runtime_contract.py.
    """
    from ml.spread.hindcast_dataset import V2_TENSOR_CHANNELS

    assert V2_TENSOR_CHANNELS == CANONICAL_V2_CHANNELS, (
        "V2_TENSOR_CHANNELS in hindcast_dataset.py has diverged from CANONICAL_V2_CHANNELS. "
        "Edit hindcast_dataset.py to re-import from ml.spread.runtime_contract."
    )


def test_hindcast_v3_channels_match_canonical():
    from ml.spread.hindcast_dataset import V3_TENSOR_CHANNELS
    from ml.spread.runtime_contract import CANONICAL_V3_CHANNELS

    assert V3_TENSOR_CHANNELS == CANONICAL_V3_CHANNELS


# ---------------------------------------------------------------------------
# Test 2: Inference builder produces a tensor matching CANONICAL_V2_CHANNELS
# ---------------------------------------------------------------------------


def test_learned_v2_build_feature_tensor_matches_canonical(monkeypatch):
    """LearnedSpreadModelV2._build_feature_tensor must produce exactly len(CANONICAL_V2_CHANNELS) channels.

    We instantiate the model via __new__ to skip ONNX loading, then call
    _build_feature_tensor directly. No DB, no ONNX session needed.
    """
    from ml.spread.learned_v2 import LearnedSpreadModelV2

    model = LearnedSpreadModelV2.__new__(LearnedSpreadModelV2)
    model.channel_names = list(CANONICAL_V2_CHANNELS)

    inp = _make_spread_model_input()
    tensor = model._build_feature_tensor(inp)

    # Shape: (1, C, H, W)
    assert tensor.ndim == 4, f"Expected 4-D (NCHW) tensor, got ndim={tensor.ndim}"
    n, c, h, w = tensor.shape
    assert n == 1
    assert c == len(CANONICAL_V2_CHANNELS), (
        f"Tensor has {c} channels but CANONICAL_V2_CHANNELS has {len(CANONICAL_V2_CHANNELS)}. "
        "Update _build_feature_tensor to match the canonical channel list."
    )
    assert h == 8 and w == 8

    # Contract-level tensor validation must not raise.
    contract = SpreadRuntimeContract(channels=CANONICAL_V2_CHANNELS)
    validate_feature_tensor(tensor, contract)


def test_learned_v2_rejects_unknown_channel_in_schema(tmp_path, monkeypatch):
    """If feature_schema.json lists a channel the builder cannot produce,
    model init must raise ContractViolationError — not silently fail at inference.
    """
    from ml.spread.learned_v2 import LearnedSpreadModelV2

    run_dir = tmp_path / "model_run"
    run_dir.mkdir()
    bad_channels = list(CANONICAL_V2_CHANNELS[:-1]) + ["UNKNOWN_CHANNEL_XYZ"]
    (run_dir / "feature_schema.json").write_text(
        json.dumps({"channels": bad_channels}), encoding="utf-8"
    )
    (run_dir / "model.onnx").write_bytes(b"dummy")
    monkeypatch.setitem(sys.modules, "onnxruntime", _DummyOrt())

    with pytest.raises(ContractViolationError, match="UNKNOWN_CHANNEL_XYZ"):
        LearnedSpreadModelV2(model_run_dir=str(run_dir))


def test_learned_v2_passes_when_runtime_contract_matches(tmp_path, monkeypatch):
    """When runtime_contract.json matches channel_names, init must succeed."""
    from ml.spread.learned_v2 import LearnedSpreadModelV2

    run_dir = tmp_path / "model_run"
    run_dir.mkdir()
    (run_dir / "feature_schema.json").write_text(
        json.dumps({"channels": list(CANONICAL_V2_CHANNELS)}), encoding="utf-8"
    )
    write_contract(run_dir / "runtime_contract.json", SpreadRuntimeContract(channels=CANONICAL_V2_CHANNELS))
    (run_dir / "model.onnx").write_bytes(b"dummy")
    monkeypatch.setitem(sys.modules, "onnxruntime", _DummyOrt())

    model = LearnedSpreadModelV2(model_run_dir=str(run_dir))
    assert model.channel_names == list(CANONICAL_V2_CHANNELS)


def test_learned_v2_rejects_when_runtime_contract_mismatches_schema(tmp_path, monkeypatch):
    """If runtime_contract.json and feature_schema.json disagree, init must raise."""
    from ml.spread.learned_v2 import LearnedSpreadModelV2

    run_dir = tmp_path / "model_run"
    run_dir.mkdir()
    (run_dir / "feature_schema.json").write_text(
        json.dumps({"channels": list(CANONICAL_V2_CHANNELS)}), encoding="utf-8"
    )
    wrong_channels = ("fire_t0", "fire_t-6h")  # severely truncated
    write_contract(run_dir / "runtime_contract.json", SpreadRuntimeContract(channels=wrong_channels))
    (run_dir / "model.onnx").write_bytes(b"dummy")
    monkeypatch.setitem(sys.modules, "onnxruntime", _DummyOrt())

    with pytest.raises(ContractViolationError, match="STOP"):
        LearnedSpreadModelV2(model_run_dir=str(run_dir))


# ---------------------------------------------------------------------------
# Test 3: Gate report rejects on channel mismatch (STOP-CONTRACT-001)
# ---------------------------------------------------------------------------


_GOOD_DECISION = {
    "pass": True,
    "weighted_bss_improvement": 0.05,
    "recommend_challenger": True,
    "reasons": [],
}


def test_gate_stop_contract_001_fires_on_wrong_feature_schema(tmp_path):
    """STOP-CONTRACT-001 must appear when challenger feature_schema has wrong channels."""
    from ml.eval_spread_champion_challenger import _build_stage_governance

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bad_channels = list(CANONICAL_V2_CHANNELS[:-1]) + ["RENAMED_CHANNEL"]
    (run_dir / "feature_schema.json").write_text(
        json.dumps({"channels": bad_channels}), encoding="utf-8"
    )

    config = {
        "gate": {"maturity_stage": "mvp_operational"},
        "challenger": {
            "model_name": "LearnedSpreadModelV2",
            "model_params": {"model_run_dir": str(run_dir)},
        },
    }
    governance = _build_stage_governance(config=config, decision=_GOOD_DECISION, summary_rows=[])
    stop_ids = [s["id"] for s in governance["hard_stops"]]
    assert "STOP-CONTRACT-001" in stop_ids, (
        f"Expected STOP-CONTRACT-001 in hard_stops, got: {stop_ids}"
    )
    assert governance["promotion_decision"] == "hold_challenger"


def test_gate_stop_contract_001_fires_on_missing_schema(tmp_path):
    """STOP-CONTRACT-001 must appear when model_run_dir has no schema file at all."""
    from ml.eval_spread_champion_challenger import _build_stage_governance

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()  # no feature_schema.json or runtime_contract.json

    config = {
        "gate": {"maturity_stage": "mvp_operational"},
        "challenger": {
            "model_name": "LearnedSpreadModelV2",
            "model_params": {"model_run_dir": str(run_dir)},
        },
    }
    governance = _build_stage_governance(config=config, decision=_GOOD_DECISION, summary_rows=[])
    stop_ids = [s["id"] for s in governance["hard_stops"]]
    assert "STOP-CONTRACT-001" in stop_ids


def test_gate_stop_contract_001_fires_when_model_run_dir_missing(tmp_path):
    """STOP-CONTRACT-001 must appear when model_params has no model_run_dir."""
    from ml.eval_spread_champion_challenger import _build_stage_governance

    config = {
        "gate": {"maturity_stage": "mvp_operational"},
        "challenger": {
            "model_name": "LearnedSpreadModelV2",
            "model_params": {},
        },
    }
    governance = _build_stage_governance(config=config, decision=_GOOD_DECISION, summary_rows=[])
    stop_ids = [s["id"] for s in governance["hard_stops"]]
    assert "STOP-CONTRACT-001" in stop_ids


def test_gate_passes_when_runtime_contract_matches_canonical(tmp_path):
    """Gate must NOT fire STOP-CONTRACT-001 when runtime_contract.json is correct."""
    from ml.eval_spread_champion_challenger import _build_stage_governance

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_contract(
        run_dir / "runtime_contract.json",
        SpreadRuntimeContract(channels=CANONICAL_V2_CHANNELS),
    )

    config = {
        "gate": {"maturity_stage": "mvp_operational"},
        "challenger": {
            "model_name": "LearnedSpreadModelV2",
            "model_params": {"model_run_dir": str(run_dir)},
        },
    }
    governance = _build_stage_governance(config=config, decision=_GOOD_DECISION, summary_rows=[])
    stop_ids = [s["id"] for s in governance["hard_stops"]]
    assert "STOP-CONTRACT-001" not in stop_ids, (
        f"STOP-CONTRACT-001 fired unexpectedly; hard_stops={governance['hard_stops']}"
    )


def test_gate_passes_when_feature_schema_matches_canonical(tmp_path):
    """Gate must NOT fire STOP-CONTRACT-001 when feature_schema.json has correct channels."""
    from ml.eval_spread_champion_challenger import _build_stage_governance

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "feature_schema.json").write_text(
        json.dumps({"channels": list(CANONICAL_V2_CHANNELS)}), encoding="utf-8"
    )

    config = {
        "gate": {"maturity_stage": "mvp_operational"},
        "challenger": {
            "model_name": "LearnedSpreadModelV2",
            "model_params": {"model_run_dir": str(run_dir)},
        },
    }
    governance = _build_stage_governance(config=config, decision=_GOOD_DECISION, summary_rows=[])
    stop_ids = [s["id"] for s in governance["hard_stops"]]
    assert "STOP-CONTRACT-001" not in stop_ids


def test_gate_skips_contract_check_for_heuristic_model():
    """Heuristic/tabular models have no tensor schema — gate must not fire STOP-CONTRACT-001."""
    from ml.eval_spread_champion_challenger import _build_stage_governance

    config = {
        "gate": {"maturity_stage": "mvp_operational"},
        "challenger": {
            "model_name": "HeuristicSpreadModelV0",
            "model_params": {},
        },
    }
    governance = _build_stage_governance(config=config, decision=_GOOD_DECISION, summary_rows=[])
    stop_ids = [s["id"] for s in governance["hard_stops"]]
    assert "STOP-CONTRACT-001" not in stop_ids


# ---------------------------------------------------------------------------
# Test 4: runtime_contract.py utility round-trip
# ---------------------------------------------------------------------------


def test_write_and_load_contract_roundtrip(tmp_path):
    contract = SpreadRuntimeContract(channels=CANONICAL_V2_CHANNELS)
    path = tmp_path / "runtime_contract.json"
    write_contract(path, contract)
    loaded = load_contract(path)
    assert loaded.channels == CANONICAL_V2_CHANNELS
    assert loaded.dtype == "float32"
    assert loaded.layout == "CHW"


def test_load_contract_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="STOP"):
        load_contract(tmp_path / "does_not_exist.json")


def test_validate_channel_alignment_passes_on_match():
    validate_channel_alignment(CANONICAL_V2_CHANNELS, CANONICAL_V2_CHANNELS)


def test_validate_channel_alignment_raises_on_missing_channel():
    truncated = CANONICAL_V2_CHANNELS[:-1]
    with pytest.raises(ContractViolationError, match="missing from inference"):
        validate_channel_alignment(truncated, CANONICAL_V2_CHANNELS)


def test_validate_channel_alignment_raises_on_wrong_order():
    reordered = list(CANONICAL_V2_CHANNELS)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ContractViolationError, match="order mismatch"):
        validate_channel_alignment(reordered, list(CANONICAL_V2_CHANNELS))


def test_validate_feature_tensor_passes_correct_shape():
    contract = SpreadRuntimeContract(channels=CANONICAL_V2_CHANNELS)
    c = len(CANONICAL_V2_CHANNELS)
    tensor_chw = np.zeros((c, 8, 8), dtype=np.float32)
    validate_feature_tensor(tensor_chw, contract)  # must not raise

    tensor_nchw = np.zeros((1, c, 8, 8), dtype=np.float32)
    validate_feature_tensor(tensor_nchw, contract)  # must not raise


def test_validate_feature_tensor_raises_on_wrong_channel_count():
    contract = SpreadRuntimeContract(channels=CANONICAL_V2_CHANNELS)
    wrong = np.zeros((5, 8, 8), dtype=np.float32)  # only 5 channels
    with pytest.raises(ContractViolationError, match="channels"):
        validate_feature_tensor(wrong, contract)
