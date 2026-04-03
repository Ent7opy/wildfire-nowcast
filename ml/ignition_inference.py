"""Runtime inference for the ignition probability model.

Loads the promoted ONNX model from the model registry, validates the runtime
contract, and returns per-cell ignition probabilities with categorical
classification.

Categorical thresholds (configurable via env vars):
  IGNITION_THRESHOLD_LOW      default 0.25  → "low"
  IGNITION_THRESHOLD_ELEVATED default 0.50  → "elevated"
  IGNITION_THRESHOLD_HIGH     default 0.75  → "high"
  scores >= HIGH threshold    → "critical"

Contract validation:
  If any required feature listed in the runtime contract is missing from the
  input DataFrame, this module raises a hard BLOCKER RuntimeError.  Silently
  substituting zeros for missing required features is explicitly forbidden.

Usage (programmatic):
    from ml.ignition_inference import IgnitionInferenceEngine

    engine = IgnitionInferenceEngine.from_registry(db_engine)
    result = engine.predict(feature_df)
    # result["ignition_probability"] → float [0, 1]
    # result["ignition_category"]    → "low" | "elevated" | "high" | "critical"
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ignition_inference")

# Env-configurable threshold profile.
_THRESHOLD_LOW = float(os.getenv("IGNITION_THRESHOLD_LOW", "0.25"))
_THRESHOLD_ELEVATED = float(os.getenv("IGNITION_THRESHOLD_ELEVATED", "0.50"))
_THRESHOLD_HIGH = float(os.getenv("IGNITION_THRESHOLD_HIGH", "0.75"))

# Env gates.
_IGNITION_REQUIRED = os.getenv("IGNITION_REQUIRED", "true").strip().lower() == "true"
_IGNITION_MODEL_PATH = os.getenv("IGNITION_MODEL_PATH", "")


def _categorise(prob: float, low: float, elevated: float, high: float) -> str:
    if prob < low:
        return "low"
    if prob < elevated:
        return "elevated"
    if prob < high:
        return "high"
    return "critical"


def _load_contract(contract_path: Path) -> dict[str, Any]:
    with open(contract_path) as f:
        return json.load(f)


def _validate_contract(contract: dict[str, Any], feature_df: pd.DataFrame) -> None:
    """Validate that all required features are present in the DataFrame.

    Raises RuntimeError (BLOCKER) if any required feature is missing.
    This is the hard-stop required by AGENTS.md: missing required features at
    runtime must never be silently substituted.
    """
    required = contract.get("required_features", [])
    missing = [f for f in required if f not in feature_df.columns]
    if missing:
        raise RuntimeError(
            f"BLOCKER [ignition_inference] Required features missing from input: {missing}. "
            "Cannot run ignition inference without all required features. "
            "Check that the upstream feature pipeline has populated these columns. "
            "If the model was trained without these features, retrain and re-register. "
            f"Full required list: {required}"
        )


def _load_onnx_session(model_path: Path):
    """Load an ONNX inference session."""
    import onnxruntime as rt  # noqa: PLC0415

    # Disable GPU; CPU is sufficient for per-cell inference on a forecast grid.
    opts = rt.SessionOptions()
    opts.log_severity_level = 3  # Suppress verbose ONNX logging.
    sess = rt.InferenceSession(str(model_path), sess_options=opts)
    return sess


def _run_onnx_session(
    sess,
    X: np.ndarray,
) -> np.ndarray:
    """Run ONNX session and extract positive-class probabilities.

    ONNXRuntime classifiers can return probabilities as either:
    - A float32 array of shape (n, 2) — direct probability matrix.
    - A list of dicts [{0: p0, 1: p1}, ...] — ZipMap output.

    This function normalises both forms to a 1-D float array of positive
    class probabilities.
    """
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X.astype(np.float32)})

    # outputs[0] = labels, outputs[1] = probabilities (ZipMap or array).
    if len(outputs) < 2:
        raise RuntimeError("BLOCKER [ignition_inference] ONNX model returned fewer than 2 outputs.")

    proba_output = outputs[1]

    if isinstance(proba_output, np.ndarray):
        if proba_output.ndim == 2 and proba_output.shape[1] == 2:
            return proba_output[:, 1].astype(float)
        if proba_output.ndim == 1:
            return proba_output.astype(float)
        raise RuntimeError(
            f"BLOCKER [ignition_inference] Unexpected ONNX probability output shape: {proba_output.shape}"
        )

    if isinstance(proba_output, list) and len(proba_output) > 0:
        # ZipMap output: list of dicts {class_id: probability}.
        if isinstance(proba_output[0], dict):
            return np.array([d.get(1, d.get(1.0, 0.0)) for d in proba_output], dtype=float)
        if isinstance(proba_output[0], (list, np.ndarray)):
            arr = np.array(proba_output, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr[:, 1]
            return arr.ravel().astype(float)

    raise RuntimeError(
        f"BLOCKER [ignition_inference] Cannot extract probabilities from ONNX output: "
        f"type={type(proba_output)}"
    )


class IgnitionInferenceEngine:
    """Stateful inference engine for ignition probability prediction.

    Attributes:
        model_path: Path to the ONNX model file.
        contract: Runtime contract dict (required features, thresholds).
        thresholds: Categorical probability thresholds.
    """

    def __init__(
        self,
        model_path: Path,
        contract: dict[str, Any],
        *,
        threshold_low: float = _THRESHOLD_LOW,
        threshold_elevated: float = _THRESHOLD_ELEVATED,
        threshold_high: float = _THRESHOLD_HIGH,
    ) -> None:
        self.model_path = model_path
        self.contract = contract
        self.threshold_low = threshold_low
        self.threshold_elevated = threshold_elevated
        self.threshold_high = threshold_high
        self._sess = None  # Lazy-load the ONNX session.

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        *,
        threshold_low: float | None = None,
        threshold_elevated: float | None = None,
        threshold_high: float | None = None,
    ) -> "IgnitionInferenceEngine":
        """Load from an explicit model directory or ONNX file path.

        Expects a contract.json file alongside the model.onnx in the same
        directory (as written by ml.train_ignition).

        Threshold priority: explicit arg > contract.json thresholds > env defaults.
        """
        path = Path(model_path)
        if path.is_dir():
            onnx_path = path / "model.onnx"
            contract_path = path / "contract.json"
        else:
            onnx_path = path
            contract_path = path.parent / "contract.json"

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"BLOCKER [ignition_inference] ONNX model not found: {onnx_path}"
            )
        if not contract_path.exists():
            raise FileNotFoundError(
                f"BLOCKER [ignition_inference] Runtime contract not found: {contract_path}. "
                "Every ignition model must have a contract.json alongside it."
            )

        contract = _load_contract(contract_path)
        ct = contract.get("thresholds", {})
        return cls(
            model_path=onnx_path,
            contract=contract,
            threshold_low=threshold_low if threshold_low is not None else ct.get("low_max", _THRESHOLD_LOW),
            threshold_elevated=threshold_elevated if threshold_elevated is not None else ct.get("elevated_max", _THRESHOLD_ELEVATED),
            threshold_high=threshold_high if threshold_high is not None else ct.get("high_max", _THRESHOLD_HIGH),
        )

    @classmethod
    def from_registry(
        cls,
        db_engine,
        *,
        threshold_low: float | None = None,
        threshold_elevated: float | None = None,
        threshold_high: float | None = None,
    ) -> "IgnitionInferenceEngine":
        """Load the currently promoted ignition model from the model registry.

        Raises RuntimeError with a 503-equivalent message if IGNITION_REQUIRED=true
        and no promoted model exists.

        Threshold priority: explicit arg > contract.json thresholds > env defaults.
        """
        from api.model_registry import resolve_active_model  # noqa: PLC0415

        model_row = resolve_active_model(family="ignition", engine=db_engine)
        if model_row is None:
            if _IGNITION_REQUIRED:
                raise RuntimeError(
                    "BLOCKER [ignition_inference] No promoted ignition model found in the registry. "
                    "Run: make ignition-train && make model-register FAMILY=ignition ... "
                    "&& make model-promote FAMILY=ignition MODEL_ID=... "
                    "Or set IGNITION_REQUIRED=false to return 503 gracefully."
                )
            raise RuntimeError(
                "503 [ignition_inference] No promoted ignition model available. "
                "IGNITION_REQUIRED=false; endpoint should return HTTP 503."
            )

        artifact_uri = model_row["artifact_uri"]

        # Prefer contract embedded in registry metrics_json, fall back to file.
        metrics_json = model_row.get("metrics_json") or {}
        contract = metrics_json.get("runtime_contract") or {}

        if not contract:
            contract_path = Path(artifact_uri).parent / "contract.json"
            if not contract_path.exists():
                raise RuntimeError(
                    f"BLOCKER [ignition_inference] No runtime contract in registry or at {contract_path}. "
                    "Re-register the model with RUNTIME_CONTRACT=@path/contract.json."
                )
            contract = _load_contract(contract_path)

        ct = contract.get("thresholds", {})
        return cls(
            model_path=Path(artifact_uri),
            contract=contract,
            threshold_low=threshold_low if threshold_low is not None else ct.get("low_max", _THRESHOLD_LOW),
            threshold_elevated=threshold_elevated if threshold_elevated is not None else ct.get("elevated_max", _THRESHOLD_ELEVATED),
            threshold_high=threshold_high if threshold_high is not None else ct.get("high_max", _THRESHOLD_HIGH),
        )

    def _get_session(self):
        if self._sess is None:
            self._sess = _load_onnx_session(self.model_path)
        return self._sess

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Run ignition probability inference on a feature DataFrame.

        Args:
            feature_df: DataFrame with one row per grid cell.  Must contain all
                features listed in the runtime contract (BLOCKER if any are missing).

        Returns:
            A copy of feature_df with two new columns:
            - ``ignition_probability``: calibrated probability in [0, 1]
            - ``ignition_category``: one of "low" | "elevated" | "high" | "critical"

        Raises:
            RuntimeError: BLOCKER if required features are missing.
        """
        if feature_df.empty:
            result = feature_df.copy()
            result["ignition_probability"] = pd.Series([], dtype=float)
            result["ignition_category"] = pd.Series([], dtype=str)
            return result

        # Hard-stop on missing required features.
        if self.contract:
            _validate_contract(self.contract, feature_df)

        required_features: list[str] = self.contract.get("required_features", [])
        if not required_features:
            LOGGER.warning(
                "WARNING [ignition_inference] No required_features in contract. "
                "Using all numeric columns. TARGET_STAGE: science_grade"
            )
            required_features = [
                c for c in feature_df.columns
                if feature_df[c].dtype in (np.float32, np.float64, float, int, bool)
            ]

        df = feature_df.copy()
        if "thunderstorm_active" in df.columns:
            df["thunderstorm_active"] = df["thunderstorm_active"].astype(float)

        X = df[required_features].values.astype(np.float32)

        sess = self._get_session()
        probabilities = _run_onnx_session(sess, X)
        probabilities = np.clip(probabilities, 0.0, 1.0)

        result = feature_df.copy()
        result["ignition_probability"] = probabilities
        result["ignition_category"] = [
            _categorise(p, self.threshold_low, self.threshold_elevated, self.threshold_high)
            for p in probabilities
        ]
        return result


def load_engine_from_env() -> IgnitionInferenceEngine:
    """Load inference engine from IGNITION_MODEL_PATH env var or model registry.

    Intended for use in the API startup path (FastAPI lifespan).
    """
    if _IGNITION_MODEL_PATH:
        LOGGER.info("Loading ignition model from IGNITION_MODEL_PATH=%s", _IGNITION_MODEL_PATH)
        return IgnitionInferenceEngine.from_model_path(_IGNITION_MODEL_PATH)

    # Resolve from registry.
    from api.db import get_engine as get_db_engine  # noqa: PLC0415

    db_engine = get_db_engine()
    LOGGER.info("Loading ignition model from model registry")
    return IgnitionInferenceEngine.from_registry(db_engine)
