"""Training pipeline for the ignition probability model.

Workflow:
  1. Load training snapshot (parquet produced by ml.ignition.snapshot).
  2. Train an XGBoost binary classifier.
  3. Apply probability calibration (isotonic or Platt).
  4. Evaluate: AUC-ROC, Brier score, calibration curve.
  5. Export to ONNX.
  6. Write metrics.json and gate_report.json to models/ignition/<run_id>/.

Usage:
  python -m ml.train_ignition --config configs/ignition_train.yaml

Gate:
  gate_report.json field "pass": true iff auc_roc >= auc_roc_gate_threshold.
  Model promotion requires "pass": true.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

from ml.calibration import (
    _maybe_git_sha,
    apply_binary_probability_calibrator,
    fit_binary_probability_calibrator,
)
from ml.parquet_io import read_parquet_with_fallback

LOGGER = logging.getLogger("train_ignition")

_DEFAULT_FEATURES = [
    "fuel_moisture",
    "lulc_flammability",
    "relative_humidity",
    "temperature_c",
    "wind_speed_kmh",
    "precip_last_7d_mm",
    "drought_index",
    "thunderstorm_active",
    "days_since_last_burn",
]

_DEFAULT_CONFIG: dict[str, Any] = {
    "snapshot_path": None,  # Required
    "out_root": "models/ignition",
    "features": _DEFAULT_FEATURES,
    "label_col": "ignition_label",
    "calibration_method": "isotonic",
    "calibration_holdout_fraction": 0.5,
    "auc_roc_gate_threshold": 0.65,
    "model_params": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 4,
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": 42,
    },
    # Negative downsampling: keep at most neg_pos_ratio * n_positive negatives.
    "neg_pos_ratio": 20,
    # Minimum positive examples required for training.
    "min_positives": 50,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(_DEFAULT_CONFIG, user_cfg)


def _load_snapshot(snapshot_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and eval parquet splits.

    Accepts either:
    - A directory with train.parquet / eval.parquet
    - A single parquet file (used as both train and eval)
    """
    path = Path(snapshot_path)
    try:
        train = read_parquet_with_fallback(str(path / "train.parquet"))
        eval_ = read_parquet_with_fallback(str(path / "eval.parquet"))
    except (FileNotFoundError, OSError):
        full = read_parquet_with_fallback(str(path))
        n_train = int(len(full) * 0.8)
        train = full.iloc[:n_train].reset_index(drop=True)
        eval_ = full.iloc[n_train:].reset_index(drop=True)
    return train, eval_


def _downsample_negatives(
    df: pd.DataFrame, label_col: str, neg_pos_ratio: int
) -> pd.DataFrame:
    """Randomly downsample negatives to at most neg_pos_ratio × n_positive."""
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    n_pos = len(pos)
    if n_pos == 0:
        return df
    max_neg = neg_pos_ratio * n_pos
    if len(neg) > max_neg:
        neg = neg.sample(n=max_neg, random_state=42)
    return pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)


def _prepare_matrix(
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix and labels."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(
            f"BLOCKER [train_ignition] Required features missing from snapshot: {missing}. "
            "Re-run the snapshot pipeline to include all required features. "
            f"Full required list: {features}"
        )

    df = df.copy()
    # thunderstorm_active is bool → cast to float.
    if "thunderstorm_active" in df.columns:
        df["thunderstorm_active"] = df["thunderstorm_active"].astype(float)

    X = df[features].fillna(0.0).values.astype(np.float32)
    y = df[label_col].fillna(0).astype(int).values
    return X, y, features


def _export_onnx(
    clf: xgb.XGBClassifier,
    feature_names: list[str],
    out_path: Path,
) -> None:
    """Export XGBoost classifier to ONNX format.

    Uses onnxmltools (recommended) with skl2onnx fallback for the XGBoost
    tree-ensemble converter.  Both packages convert the model to the standard
    ONNX TreeEnsembleClassifier operator, which ONNXRuntime executes efficiently.
    """
    n_features = len(feature_names)

    try:
        from onnxmltools.convert import convert_xgboost  # noqa: PLC0415
        from onnxmltools.convert.common.data_types import FloatTensorType  # noqa: PLC0415

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        model_onnx = convert_xgboost(clf.get_booster(), initial_types=initial_type)
        with open(out_path, "wb") as f:
            f.write(model_onnx.SerializeToString())
        LOGGER.info("ONNX export (onnxmltools) → %s", out_path)
        return
    except ImportError:
        pass

    # Fallback: skl2onnx with XGBoost sklearn wrapper.
    try:
        from skl2onnx import to_onnx as skl2onnx_to_onnx  # noqa: PLC0415
        from skl2onnx.common.data_types import FloatTensorType  # noqa: PLC0415

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        model_onnx = skl2onnx_to_onnx(clf, initial_types=initial_type, target_opset=17)
        with open(out_path, "wb") as f:
            f.write(model_onnx.SerializeToString())
        LOGGER.info("ONNX export (skl2onnx) → %s", out_path)
        return
    except ImportError:
        pass

    raise RuntimeError(
        "ONNX export requires onnxmltools or skl2onnx. "
        "Install: pip install onnxmltools  (preferred)  or  pip install skl2onnx"
    )


def _build_runtime_contract(
    feature_names: list[str],
    thresholds: dict[str, float],
    n_features: int,
) -> dict[str, Any]:
    """Build the runtime contract JSON stored alongside the model."""
    return {
        "schema_version": "1.0",
        "required_features": feature_names,
        "feature_dtypes": {f: "float32" for f in feature_names},
        "input_shape": [None, n_features],
        "output_probabilities_index": 1,
        "thresholds": thresholds,
        "missing_feature_policy": "BLOCKER",
    }


def train_ignition(config: dict[str, Any]) -> Path:
    """Run the full ignition model training pipeline.

    Returns the run directory path.
    """
    snapshot_path = config.get("snapshot_path")
    if not snapshot_path:
        raise ValueError("snapshot_path is required in config")

    features: list[str] = list(config.get("features", _DEFAULT_FEATURES))
    label_col: str = config.get("label_col", "ignition_label")
    neg_pos_ratio: int = int(config.get("neg_pos_ratio", 20))
    min_positives: int = int(config.get("min_positives", 50))
    calib_method: str = config.get("calibration_method", "isotonic")
    calib_holdout: float = float(config.get("calibration_holdout_fraction", 0.5))
    auc_gate: float = float(config.get("auc_roc_gate_threshold", 0.65))
    model_params: dict[str, Any] = dict(config.get("model_params", {}))

    # Build run directory.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_root = Path(config.get("out_root", "models/ignition"))
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Run directory: %s", run_dir)

    # Load data.
    train_df, eval_df = _load_snapshot(snapshot_path)
    LOGGER.info(
        "Snapshot: train=%d rows, eval=%d rows", len(train_df), len(eval_df)
    )

    # Downsample negatives.
    train_df = _downsample_negatives(train_df, label_col, neg_pos_ratio)
    n_pos = int((train_df[label_col] == 1).sum())
    n_neg = int((train_df[label_col] == 0).sum())
    LOGGER.info("After downsampling: %d positive, %d negative", n_pos, n_neg)

    if n_pos < min_positives:
        raise RuntimeError(
            f"BLOCKER [train_ignition] Insufficient positive examples: {n_pos} < {min_positives}. "
            "Extend the snapshot date range or lower min_positives in config."
        )

    X_train, y_train, _ = _prepare_matrix(train_df, features, label_col)
    X_eval, y_eval, _ = _prepare_matrix(eval_df, features, label_col)

    # Split training data for calibration (hold-out the latest calib_holdout fraction).
    n_calib = max(1, int(len(X_train) * calib_holdout))
    X_fit, y_fit = X_train[:-n_calib], y_train[:-n_calib]
    X_calib, y_calib = X_train[-n_calib:], y_train[-n_calib:]

    # Scale positive weight for class imbalance.
    n_neg_fit = int(np.sum(y_fit == 0))
    n_pos_fit = int(np.sum(y_fit == 1))
    scale_pos_weight = max(1.0, n_neg_fit / max(1, n_pos_fit))
    LOGGER.info("scale_pos_weight=%.2f", scale_pos_weight)

    # Train.
    clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **model_params)
    clf.fit(
        X_fit,
        y_fit,
        eval_set=[(X_eval, y_eval)],
        verbose=False,
    )

    # Raw scores on calibration set and eval set.
    raw_calib = clf.predict_proba(X_calib)[:, 1]
    raw_eval = clf.predict_proba(X_eval)[:, 1]

    # Calibration.
    calibrator = fit_binary_probability_calibrator(raw_calib, y_calib, method=calib_method)
    cal_eval = apply_binary_probability_calibrator(calibrator, raw_eval)

    # Evaluation metrics.
    auc_roc_raw = float(roc_auc_score(y_eval, raw_eval)) if len(np.unique(y_eval)) > 1 else 0.0
    auc_roc_cal = float(roc_auc_score(y_eval, cal_eval)) if len(np.unique(y_eval)) > 1 else 0.0
    brier_raw = float(brier_score_loss(y_eval, raw_eval)) if len(np.unique(y_eval)) > 1 else 1.0
    brier_cal = float(brier_score_loss(y_eval, cal_eval)) if len(np.unique(y_eval)) > 1 else 1.0

    LOGGER.info(
        "Eval — AUC-ROC (raw/cal): %.4f / %.4f  |  Brier (raw/cal): %.4f / %.4f",
        auc_roc_raw, auc_roc_cal, brier_raw, brier_cal,
    )

    # Calibration curve.
    cal_curve_bins = 10
    try:
        prob_true_raw, prob_pred_raw = calibration_curve(y_eval, raw_eval, n_bins=cal_curve_bins)
        prob_true_cal, prob_pred_cal = calibration_curve(y_eval, cal_eval, n_bins=cal_curve_bins)
    except ValueError:
        prob_true_raw = prob_pred_raw = prob_true_cal = prob_pred_cal = np.array([])

    # Threshold defaults (configurable via IGNITION_THRESHOLD_PROFILE env if needed).
    thresholds = {
        "low_max": 0.25,
        "elevated_max": 0.50,
        "high_max": 0.75,
        # >= high_max → "critical"
    }

    # Write metrics.json.
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _maybe_git_sha(),
        "n_train": int(len(X_train)),
        "n_eval": int(len(X_eval)),
        "n_train_positive": n_pos,
        "n_eval_positive": int(np.sum(y_eval == 1)),
        "auc_roc_raw": auc_roc_raw,
        "auc_roc_calibrated": auc_roc_cal,
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "calibration_method": calib_method,
        "features": features,
        "calibration_curve": {
            "raw": {
                "prob_true": prob_true_raw.tolist(),
                "prob_pred": prob_pred_raw.tolist(),
            },
            "calibrated": {
                "prob_true": prob_true_cal.tolist(),
                "prob_pred": prob_pred_cal.tolist(),
            },
        },
        "model_params": model_params,
        "thresholds": thresholds,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "xgboost": xgb.__version__,
        },
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Write gate_report.json.
    gate_passed = auc_roc_cal >= auc_gate
    gate_report: dict[str, Any] = {
        "pass": gate_passed,
        "auc_roc_calibrated": auc_roc_cal,
        "auc_roc_gate_threshold": auc_gate,
        "brier_calibrated": brier_cal,
        "n_eval_positive": int(np.sum(y_eval == 1)),
        "note": (
            "Gate passed: calibrated AUC-ROC meets threshold."
            if gate_passed
            else f"Gate FAILED: calibrated AUC-ROC {auc_roc_cal:.4f} < threshold {auc_gate:.4f}. "
                 "Do not promote this model."
        ),
    }
    with open(run_dir / "gate_report.json", "w") as f:
        json.dump(gate_report, f, indent=2)
    LOGGER.info("Gate: pass=%s (auc_roc=%.4f, threshold=%.4f)", gate_passed, auc_roc_cal, auc_gate)

    # Export ONNX model.
    onnx_path = run_dir / "model.onnx"
    _export_onnx(clf, features, onnx_path)

    # Write runtime contract.
    contract = _build_runtime_contract(features, thresholds, len(features))
    with open(run_dir / "contract.json", "w") as f:
        json.dump(contract, f, indent=2)

    # Write feature list (for quick inspection).
    with open(run_dir / "feature_list.json", "w") as f:
        json.dump(features, f, indent=2)

    LOGGER.info("Artifacts written to %s", run_dir)
    return run_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train ignition probability model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ignition_train.yaml",
        help="Path to YAML training config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = train_ignition(config)
    LOGGER.info("Training complete. Run directory: %s", run_dir)

    # Print model-register command hint.
    print(
        f"\nTo register this model run:\n"
        f"  make model-register FAMILY=ignition "
        f"ARTIFACT={run_dir / 'model.onnx'} "
        f"METRICS=@{run_dir / 'metrics.json'} "
        f"RUNTIME_CONTRACT=@{run_dir / 'contract.json'}"
    )


if __name__ == "__main__":
    main()
