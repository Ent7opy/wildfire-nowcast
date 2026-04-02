"""Unit tests for ml.ignition_inference — contract validation and inference logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import tempfile

import numpy as np
import pandas as pd
import pytest

from ml.ignition_inference import (
    _categorise,
    _validate_contract,
    _run_onnx_session,
    IgnitionInferenceEngine,
)


# ── Categorisation thresholds ─────────────────────────────────────────────────

class TestCategorise:
    def test_low_category(self):
        assert _categorise(0.0, 0.25, 0.50, 0.75) == "low"
        assert _categorise(0.24, 0.25, 0.50, 0.75) == "low"

    def test_elevated_category(self):
        assert _categorise(0.25, 0.25, 0.50, 0.75) == "elevated"
        assert _categorise(0.49, 0.25, 0.50, 0.75) == "elevated"

    def test_high_category(self):
        assert _categorise(0.50, 0.25, 0.50, 0.75) == "high"
        assert _categorise(0.74, 0.25, 0.50, 0.75) == "high"

    def test_critical_category(self):
        assert _categorise(0.75, 0.25, 0.50, 0.75) == "critical"
        assert _categorise(1.0, 0.25, 0.50, 0.75) == "critical"

    def test_boundary_at_zero(self):
        assert _categorise(0.0, 0.25, 0.50, 0.75) == "low"

    def test_boundary_at_one(self):
        assert _categorise(1.0, 0.25, 0.50, 0.75) == "critical"


# ── Contract validation ───────────────────────────────────────────────────────

class TestValidateContract:
    def _contract(self, features: list[str]) -> dict:
        return {
            "required_features": features,
            "feature_dtypes": {f: "float32" for f in features},
            "missing_feature_policy": "BLOCKER",
        }

    def test_all_features_present_passes(self):
        df = pd.DataFrame({"fuel_moisture": [0.5], "temperature_c": [25.0]})
        contract = self._contract(["fuel_moisture", "temperature_c"])
        # Should not raise.
        _validate_contract(contract, df)

    def test_missing_required_feature_raises_blocker(self):
        df = pd.DataFrame({"fuel_moisture": [0.5]})
        contract = self._contract(["fuel_moisture", "temperature_c"])
        with pytest.raises(RuntimeError, match="BLOCKER"):
            _validate_contract(contract, df)

    def test_missing_multiple_features_lists_all(self):
        df = pd.DataFrame({"col_a": [1.0]})
        contract = self._contract(["col_a", "col_b", "col_c"])
        with pytest.raises(RuntimeError) as exc_info:
            _validate_contract(contract, df)
        msg = str(exc_info.value)
        assert "col_b" in msg
        assert "col_c" in msg

    def test_extra_columns_in_df_are_allowed(self):
        """DataFrame may have extra columns beyond the required set."""
        df = pd.DataFrame({
            "fuel_moisture": [0.5],
            "temperature_c": [25.0],
            "extra_column": [999.0],
        })
        contract = self._contract(["fuel_moisture", "temperature_c"])
        # Should not raise.
        _validate_contract(contract, df)

    def test_empty_required_features_passes(self):
        df = pd.DataFrame({"col": [1.0]})
        _validate_contract({"required_features": []}, df)


# ── ONNX session output handling ─────────────────────────────────────────────

class TestRunOnnxSession:
    """Tests for the probability extraction from ONNX session outputs."""

    def _make_session(self, outputs) -> MagicMock:
        sess = MagicMock()
        input_spec = MagicMock()
        input_spec.name = "float_input"
        sess.get_inputs.return_value = [input_spec]
        sess.run.return_value = outputs
        return sess

    def test_array_output_shape_n_2(self):
        """Standard (n, 2) probability array — return column 1."""
        proba = np.array([[0.8, 0.2], [0.3, 0.7]])
        sess = self._make_session([np.array([0, 1]), proba])
        X = np.zeros((2, 3), dtype=np.float32)
        result = _run_onnx_session(sess, X)
        np.testing.assert_allclose(result, [0.2, 0.7])

    def test_zipmap_dict_output(self):
        """ZipMap output (list of dicts) — extract probability for class 1."""
        zipmap = [{0: 0.8, 1: 0.2}, {0: 0.3, 1: 0.7}]
        sess = self._make_session([np.array([0, 1]), zipmap])
        X = np.zeros((2, 3), dtype=np.float32)
        result = _run_onnx_session(sess, X)
        np.testing.assert_allclose(result, [0.2, 0.7])

    def test_single_output_raises_blocker(self):
        """ONNX model with fewer than 2 outputs → BLOCKER."""
        sess = self._make_session([np.array([0, 1])])
        X = np.zeros((2, 3), dtype=np.float32)
        with pytest.raises(RuntimeError, match="BLOCKER"):
            _run_onnx_session(sess, X)


# ── IgnitionInferenceEngine ───────────────────────────────────────────────────

class TestIgnitionInferenceEngine:
    """Integration-level tests for the inference engine using mocked ONNX."""

    _FEATURES = ["fuel_moisture", "temperature_c", "wind_speed_kmh",
                  "relative_humidity", "precip_last_7d_mm", "drought_index",
                  "thunderstorm_active", "days_since_last_burn", "lulc_flammability"]

    def _make_contract(self) -> dict:
        return {
            "required_features": self._FEATURES,
            "feature_dtypes": {f: "float32" for f in self._FEATURES},
            "input_shape": [None, len(self._FEATURES)],
            "missing_feature_policy": "BLOCKER",
            "thresholds": {"low_max": 0.25, "elevated_max": 0.5, "high_max": 0.75},
        }

    def _make_feature_df(self, n: int = 3) -> pd.DataFrame:
        return pd.DataFrame({
            "fuel_moisture": np.random.uniform(0.1, 0.9, n),
            "temperature_c": np.random.uniform(15, 40, n),
            "wind_speed_kmh": np.random.uniform(0, 50, n),
            "relative_humidity": np.random.uniform(10, 90, n),
            "precip_last_7d_mm": np.random.uniform(0, 20, n),
            "drought_index": np.random.uniform(0, 1, n),
            "thunderstorm_active": np.zeros(n, dtype=bool),
            "days_since_last_burn": np.random.uniform(30, 3650, n),
            "lulc_flammability": np.random.uniform(0.1, 1.0, n),
        })

    def test_predict_returns_probability_and_category(self):
        """Engine.predict returns required output columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Write a fake ONNX file (content doesn't matter; session is mocked).
            (tmp / "model.onnx").write_bytes(b"fake_onnx")
            (tmp / "contract.json").write_text(json.dumps(self._make_contract()))

            engine = IgnitionInferenceEngine.from_model_path(tmp)

            df = self._make_feature_df(3)
            fake_proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7]])

            with patch("ml.ignition_inference._load_onnx_session") as mock_load, \
                 patch("ml.ignition_inference._run_onnx_session") as mock_run:
                mock_run.return_value = fake_proba[:, 1]
                result = engine.predict(df)

        assert "ignition_probability" in result.columns
        assert "ignition_category" in result.columns
        assert len(result) == 3
        np.testing.assert_allclose(result["ignition_probability"].values, [0.1, 0.6, 0.7])

    def test_predict_categories_match_thresholds(self):
        """Categorical classifications must use the configured thresholds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "model.onnx").write_bytes(b"fake_onnx")
            (tmp / "contract.json").write_text(json.dumps(self._make_contract()))

            engine = IgnitionInferenceEngine.from_model_path(tmp)
            df = self._make_feature_df(4)
            probs = np.array([0.10, 0.35, 0.60, 0.80])

            with patch("ml.ignition_inference._load_onnx_session"), \
                 patch("ml.ignition_inference._run_onnx_session", return_value=probs):
                result = engine.predict(df)

        cats = result["ignition_category"].tolist()
        assert cats[0] == "low"
        assert cats[1] == "elevated"
        assert cats[2] == "high"
        assert cats[3] == "critical"

    def test_missing_required_feature_raises_blocker(self):
        """Missing required feature in input → BLOCKER RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "model.onnx").write_bytes(b"fake_onnx")
            (tmp / "contract.json").write_text(json.dumps(self._make_contract()))

            engine = IgnitionInferenceEngine.from_model_path(tmp)

            # DataFrame missing "drought_index".
            df = self._make_feature_df(2).drop(columns=["drought_index"])

            with pytest.raises(RuntimeError, match="BLOCKER"):
                engine.predict(df)

    def test_predict_empty_dataframe_returns_empty(self):
        """Empty input → empty output (no crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "model.onnx").write_bytes(b"fake_onnx")
            (tmp / "contract.json").write_text(json.dumps(self._make_contract()))

            engine = IgnitionInferenceEngine.from_model_path(tmp)
            empty_df = pd.DataFrame(columns=self._FEATURES)
            result = engine.predict(empty_df)

        assert len(result) == 0
        assert "ignition_probability" in result.columns
        assert "ignition_category" in result.columns

    def test_missing_model_file_raises(self):
        """Missing ONNX file at load time → FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # No model.onnx created.
            (tmp / "contract.json").write_text(json.dumps(self._make_contract()))
            with pytest.raises(FileNotFoundError, match="BLOCKER"):
                IgnitionInferenceEngine.from_model_path(tmp)

    def test_missing_contract_file_raises(self):
        """Missing contract.json at load time → FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "model.onnx").write_bytes(b"fake_onnx")
            # No contract.json created.
            with pytest.raises(FileNotFoundError, match="BLOCKER"):
                IgnitionInferenceEngine.from_model_path(tmp)

    def test_no_promoted_model_required_raises_blocker(self):
        """IGNITION_REQUIRED=true + no registry entry → BLOCKER RuntimeError."""
        mock_db = MagicMock()

        # resolve_active_model is a local import inside from_registry; patch at source.
        with patch("api.model_registry.resolve_active_model", return_value=None), \
             patch("ml.ignition_inference._IGNITION_REQUIRED", True):
            with pytest.raises(RuntimeError, match="BLOCKER"):
                IgnitionInferenceEngine.from_registry(mock_db)

    def test_probabilities_clipped_to_unit_interval(self):
        """Output probabilities must be in [0, 1] regardless of raw model output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "model.onnx").write_bytes(b"fake_onnx")
            (tmp / "contract.json").write_text(json.dumps(self._make_contract()))

            engine = IgnitionInferenceEngine.from_model_path(tmp)
            df = self._make_feature_df(3)

            # Raw model returns out-of-range values.
            raw_probs = np.array([-0.1, 1.5, 0.5])

            with patch("ml.ignition_inference._load_onnx_session"), \
                 patch("ml.ignition_inference._run_onnx_session", return_value=raw_probs):
                result = engine.predict(df)

        probs = result["ignition_probability"].values
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
