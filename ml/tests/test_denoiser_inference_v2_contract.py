"""Integration tests for denoiser v2 inference with runtime contract validation.

Note: These tests are designed to validate the contract checking logic
in _load_bundle. They use mocks to avoid external dependencies that
may not be available in test environments.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml.denoiser.runtime_contract import (
    ContractViolationError,
    DenoiserRuntimeContract,
    validate_feature_alignment,
    write_contract,
)


class TestLoadBundleContractValidation:
    """Test contract validation logic in _load_bundle."""

    def test_validate_feature_alignment_exact_match(self):
        """Test validation passes for exact feature match."""
        train_features = ["frp_max", "confidence_max", "scan_angle"]
        infer_features = ["frp_max", "confidence_max", "scan_angle"]

        validate_feature_alignment(infer_features, train_features)

    def test_validate_feature_alignment_reordered(self):
        """Test validation catches feature reordering."""
        train_features = ["frp_max", "confidence_max", "scan_angle"]
        infer_features = ["confidence_max", "frp_max", "scan_angle"]

        with pytest.raises(ContractViolationError) as exc_info:
            validate_feature_alignment(infer_features, train_features)

        assert "STOP" in str(exc_info.value)
        assert "order mismatch" in str(exc_info.value) or "index 0" in str(exc_info.value)

    def test_validate_feature_alignment_missing(self):
        """Test validation catches missing features."""
        train_features = ["frp_max", "confidence_max", "scan_angle"]
        infer_features = ["frp_max", "scan_angle"]

        with pytest.raises(ContractViolationError) as exc_info:
            validate_feature_alignment(infer_features, train_features)

        assert "STOP" in str(exc_info.value)
        assert "missing from inference" in str(exc_info.value)

    def test_validate_feature_alignment_extra(self):
        """Test validation catches extra features."""
        train_features = ["frp_max", "confidence_max"]
        infer_features = ["frp_max", "confidence_max", "scan_angle", "extra"]

        with pytest.raises(ContractViolationError) as exc_info:
            validate_feature_alignment(infer_features, train_features)

        assert "STOP" in str(exc_info.value)
        assert "absent from contract" in str(exc_info.value)


class TestContractRoundTrip:
    """Test contract writing and loading."""

    def test_contract_write_and_load(self):
        """Test round-trip serialization."""
        features = ["frp_max", "confidence_max", "scan_angle"]

        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = os.path.join(tmpdir, "runtime_contract.json")

            # Write
            contract = DenoiserRuntimeContract(features=tuple(features))
            write_contract(contract_path, contract)

            # Load and verify
            with open(contract_path) as f:
                data = json.load(f)

            assert data["features"] == features
            assert data["dtype"] == "float32"

    def test_contract_backward_compatibility(self):
        """Test loading contract without dtype field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = os.path.join(tmpdir, "runtime_contract.json")

            # Write minimal contract (no dtype)
            data = {"features": ["a", "b", "c"]}
            Path(contract_path).write_text(json.dumps(data))

            # Load should work with default dtype
            loaded = DenoiserRuntimeContract.from_dict(data)
            assert loaded.features == ("a", "b", "c")
            assert loaded.dtype == "float32"


class TestContractIntegration:
    """Test contract in realistic denoiser workflow scenarios."""

    def test_realistic_feature_set(self):
        """Test contract with realistic denoiser feature set."""
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
        )

        contract = DenoiserRuntimeContract(features=features)
        assert contract.n_features == 11

        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = os.path.join(tmpdir, "runtime_contract.json")
            write_contract(contract_path, contract)

            with open(contract_path) as f:
                data = json.load(f)

            assert len(data["features"]) == 11
            assert data["features"][0] == "confidence_mean"
            assert data["features"][-1] == "dfmc_10hr_mean"

    def test_feature_reordering_detection(self):
        """Test detection of silent feature reordering bug."""
        train_order = [
            "confidence_mean",
            "frp_max",
            "brightness_mean",
        ]

        # Simulated bug: features get reordered in inference
        infer_order = [
            "frp_max",
            "confidence_mean",
            "brightness_mean",
        ]

        with pytest.raises(ContractViolationError) as exc_info:
            validate_feature_alignment(infer_order, train_order)

        error_msg = str(exc_info.value)
        assert "STOP" in error_msg
        assert "feature mismatch between inference" in error_msg
