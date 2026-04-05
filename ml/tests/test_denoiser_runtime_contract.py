"""Tests for denoiser v2 runtime contract validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ml.denoiser.runtime_contract import (
    ContractViolationError,
    DenoiserRuntimeContract,
    load_contract,
    validate_feature_alignment,
    write_contract,
)


class TestDenoiserRuntimeContract:
    """Test contract dataclass and serialization."""

    def test_contract_creation(self):
        """Test basic contract creation with tuple features."""
        features = ("frp_max", "confidence_max", "scan_angle")
        contract = DenoiserRuntimeContract(features=features)

        assert contract.features == features
        assert contract.n_features == 3
        assert contract.dtype == "float32"

    def test_contract_to_dict(self):
        """Test contract serialization to dict."""
        features = ("frp_max", "confidence_max")
        contract = DenoiserRuntimeContract(features=features, dtype="float32")

        d = contract.to_dict()
        assert d["features"] == ["frp_max", "confidence_max"]
        assert d["dtype"] == "float32"

    def test_contract_from_dict(self):
        """Test contract deserialization from dict."""
        d = {
            "features": ["frp_max", "confidence_max", "scan_angle"],
            "dtype": "float32",
        }
        contract = DenoiserRuntimeContract.from_dict(d)

        assert contract.features == ("frp_max", "confidence_max", "scan_angle")
        assert contract.dtype == "float32"

    def test_contract_immutability(self):
        """Test that contract dataclass is frozen."""
        contract = DenoiserRuntimeContract(features=("a", "b"))
        with pytest.raises(Exception):  # FrozenInstanceError
            contract.features = ("c", "d")


class TestValidateFeatureAlignment:
    """Test feature validation logic."""

    def test_exact_match(self):
        """Test validation passes when features match exactly."""
        infer = ["frp_max", "confidence_max", "scan_angle"]
        contract = ["frp_max", "confidence_max", "scan_angle"]
        # Should not raise
        validate_feature_alignment(infer, contract)

    def test_order_mismatch(self):
        """Test that feature reordering raises ContractViolationError."""
        infer = ["confidence_max", "frp_max", "scan_angle"]
        contract = ["frp_max", "confidence_max", "scan_angle"]

        with pytest.raises(ContractViolationError) as exc_info:
            validate_feature_alignment(infer, contract)

        error_msg = str(exc_info.value)
        assert "STOP" in error_msg
        assert "feature order mismatch" in error_msg or "order mismatch" in error_msg
        assert "index 0" in error_msg

    def test_missing_features(self):
        """Test that missing features raise ContractViolationError."""
        infer = ["frp_max", "scan_angle"]  # missing confidence_max
        contract = ["frp_max", "confidence_max", "scan_angle"]

        with pytest.raises(ContractViolationError) as exc_info:
            validate_feature_alignment(infer, contract)

        error_msg = str(exc_info.value)
        assert "STOP" in error_msg
        assert "missing from inference" in error_msg
        assert "confidence_max" in error_msg

    def test_extra_features(self):
        """Test that extra inference features raise ContractViolationError."""
        infer = ["frp_max", "confidence_max", "scan_angle", "extra_feature"]
        contract = ["frp_max", "confidence_max", "scan_angle"]

        with pytest.raises(ContractViolationError) as exc_info:
            validate_feature_alignment(infer, contract)

        error_msg = str(exc_info.value)
        assert "STOP" in error_msg
        assert "absent from contract" in error_msg
        assert "extra_feature" in error_msg

    def test_completely_different_features(self):
        """Test error message when feature sets are completely different."""
        infer = ["a", "b", "c"]
        contract = ["x", "y", "z"]

        with pytest.raises(ContractViolationError) as exc_info:
            validate_feature_alignment(infer, contract)

        error_msg = str(exc_info.value)
        assert "STOP" in error_msg

    def test_empty_features(self):
        """Test validation with empty feature lists."""
        validate_feature_alignment([], [])  # Should not raise

    def test_single_feature_match(self):
        """Test validation with single feature."""
        validate_feature_alignment(["frp_max"], ["frp_max"])  # Should not raise

    def test_single_feature_mismatch(self):
        """Test validation fails with single mismatched feature."""
        with pytest.raises(ContractViolationError):
            validate_feature_alignment(["frp_max"], ["confidence_max"])


class TestPersistence:
    """Test writing and loading contracts from disk."""

    def test_write_and_load_contract(self):
        """Test round-trip serialization of contract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = Path(tmpdir) / "runtime_contract.json"

            original = DenoiserRuntimeContract(
                features=("frp_max", "confidence_max", "scan_angle"),
                dtype="float32",
            )
            write_contract(contract_path, original)

            assert contract_path.exists()
            loaded = load_contract(contract_path)

            assert loaded.features == original.features
            assert loaded.dtype == original.dtype

    def test_write_contract_creates_parent_dir(self):
        """Test that write_contract creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = Path(tmpdir) / "nested" / "dir" / "runtime_contract.json"

            contract = DenoiserRuntimeContract(features=("a", "b"))
            write_contract(contract_path, contract)

            assert contract_path.exists()
            assert contract_path.parent.exists()

    def test_contract_json_format(self):
        """Test that contract JSON has expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = Path(tmpdir) / "runtime_contract.json"

            contract = DenoiserRuntimeContract(
                features=("frp_max", "confidence_max"),
                dtype="float32",
            )
            write_contract(contract_path, contract)

            # Read raw JSON to verify format
            data = json.loads(contract_path.read_text())
            assert "features" in data
            assert "dtype" in data
            assert data["features"] == ["frp_max", "confidence_max"]
            assert data["dtype"] == "float32"

    def test_load_contract_missing_file(self):
        """Test that loading missing contract raises FileNotFoundError."""
        missing_path = Path("/nonexistent/path/runtime_contract.json")
        with pytest.raises(FileNotFoundError) as exc_info:
            load_contract(missing_path)

        error_msg = str(exc_info.value)
        assert "STOP" in error_msg
        assert "runtime_contract.json not found" in error_msg

    def test_load_contract_invalid_json(self):
        """Test that loading invalid JSON raises appropriate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = Path(tmpdir) / "runtime_contract.json"
            contract_path.write_text("{ invalid json }")

            with pytest.raises(Exception):  # json.JSONDecodeError or similar
                load_contract(contract_path)

    def test_load_contract_backward_compatibility(self):
        """Test loading contract without optional fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = Path(tmpdir) / "runtime_contract.json"

            # Write minimal contract
            minimal = {"features": ["a", "b", "c"]}
            contract_path.write_text(json.dumps(minimal))

            loaded = load_contract(contract_path)
            assert loaded.features == ("a", "b", "c")
            assert loaded.dtype == "float32"  # default


class TestIntegrationWithTrainingArtifacts:
    """Test contract validation in realistic scenarios."""

    def test_realistic_denoiser_contract(self):
        """Test contract with realistic denoiser feature set."""
        # These are representative denoiser v2 features
        features = (
            "confidence_mean",
            "frp_max",
            "frp_mean",
            "brightness_mean",
            "scan_mean",
            "scan_angle_mean",
            "landcover_mean",
            "persistence_mean",
            "weather_mean",
            "lfmc_mean",
            "dfmc_10hr_mean",
            "wind_speed_mean",
            "rh2m_mean",
            "hour_of_day",
            "day_of_year",
            "sin_hour",
            "cos_hour",
            "sin_doy",
            "cos_doy",
        )

        contract = DenoiserRuntimeContract(features=features)
        assert contract.n_features == 19

        # Round-trip test
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "contract.json"
            write_contract(path, contract)
            loaded = load_contract(path)
            assert loaded.features == contract.features

    def test_reordering_detection_realistic(self):
        """Test that realistic feature reordering is caught."""
        train_features = [
            "confidence_mean",
            "frp_max",
            "frp_mean",
            "brightness_mean",
        ]

        # Inference accidentally swaps first two features
        infer_features = [
            "frp_max",  # SWAPPED
            "confidence_mean",  # SWAPPED
            "frp_mean",
            "brightness_mean",
        ]

        with pytest.raises(ContractViolationError) as exc_info:
            validate_feature_alignment(infer_features, train_features)

        error_msg = str(exc_info.value)
        assert "order mismatch" in error_msg or "index 0" in error_msg


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unicode_feature_names(self):
        """Test contract with unicode feature names (if supported)."""
        features = ("frp_max", "confidence_μ", "angle°")
        contract = DenoiserRuntimeContract(features=features)
        assert contract.n_features == 3

    def test_long_feature_list(self):
        """Test contract with large number of features."""
        features = tuple(f"feature_{i}" for i in range(1000))
        contract = DenoiserRuntimeContract(features=features)
        assert contract.n_features == 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "contract.json"
            write_contract(path, contract)
            loaded = load_contract(path)
            assert len(loaded.features) == 1000

    def test_duplicate_features_in_list(self):
        """Test behavior with duplicate feature names (should not crash)."""
        features = ("frp_max", "frp_max", "confidence_max")
        # Validation should flag this as mismatch if order differs
        validate_feature_alignment(features, features)  # No-op if identical
