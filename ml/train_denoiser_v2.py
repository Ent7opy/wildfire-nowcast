"""Train event-level denoiser v2 with PU bootstrapping + slice calibration."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional fallback
    XGBClassifier = None
    _HAS_XGBOOST = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("train_denoiser_v2")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _maybe_git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            .strip()
            or None
        )
    except Exception:
        return None


def _load_snapshot(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.isdir(path):
        train = pd.read_parquet(os.path.join(path, "train.parquet"))
        eval_df = pd.read_parquet(os.path.join(path, "eval.parquet"))
        return train, eval_df
    full = pd.read_parquet(path)
    if "start_time" in full.columns:
        full = full.sort_values("start_time")
    split_dt = full["start_time"].quantile(0.8) if "start_time" in full.columns else None
    if split_dt is not None:
        return full[full["start_time"] < split_dt].copy(), full[full["start_time"] >= split_dt].copy()
    n = len(full)
    cut = int(n * 0.8)
    return full.iloc[:cut].copy(), full.iloc[cut:].copy()


def _map_labels(df: pd.DataFrame, label_col: str = "event_label") -> np.ndarray:
    mapping = {"POSITIVE": 1, "NEGATIVE": 0}
    labels = df[label_col].map(mapping).fillna(-1).astype(int).to_numpy()
    return labels


def _build_model(config: Dict[str, Any]) -> Any:
    model_params = dict(config.get("model_params", {}))
    if _HAS_XGBOOST and config.get("model_backend", "xgboost") == "xgboost":
        defaults = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": int(config.get("seed", 42)),
            "n_jobs": 4,
        }
        defaults.update(model_params)
        return XGBClassifier(**defaults)

    defaults = {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "l2_regularization": 0.01,
        "random_state": int(config.get("seed", 42)),
    }
    defaults.update(model_params)
    return HistGradientBoostingClassifier(**defaults)


def _predict_raw(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    pred = np.asarray(model.predict(x), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def _fit_calibrator(scores: np.ndarray, y: np.ndarray, method: str) -> Dict[str, Any]:
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores, y)
        return {"type": "isotonic", "model": iso}

    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(scores.reshape(-1, 1), y)
    return {"type": "platt", "model": lr}


def _apply_calibrator(cal: Dict[str, Any], scores: np.ndarray) -> np.ndarray:
    if cal["type"] == "isotonic":
        return np.asarray(cal["model"].predict(scores), dtype=float)
    if cal["type"] == "platt":
        return np.asarray(cal["model"].predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)
    return np.asarray(scores, dtype=float)


def _slice_key(row: pd.Series, slice_cols: list[str]) -> str:
    return "|".join(f"{col}={row.get(col, 'unknown')}" for col in slice_cols)


def _metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else None
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def train_denoiser_v2(config: Dict[str, Any]) -> str:
    seed = int(config.get("seed", 42))
    np.random.seed(seed)

    train_df, eval_df = _load_snapshot(config["snapshot_path"])
    features = list(config["features"])
    label_col = str(config.get("label_column", "event_label"))
    slice_cols = list(config.get("slice_columns", ["sensor", "biome_slice"]))

    for col in features:
        if col not in train_df.columns:
            train_df[col] = np.nan
        if col not in eval_df.columns:
            eval_df[col] = np.nan

    y_train = _map_labels(train_df, label_col)
    y_eval = _map_labels(eval_df, label_col)

    known_train = y_train >= 0
    known_eval = y_eval >= 0

    if not known_train.any():
        raise ValueError("No known labels (POSITIVE/NEGATIVE) available in train split.")

    model_stage1 = _build_model(config)
    x_train_known = train_df.loc[known_train, features]
    y_train_known = y_train[known_train]

    class_weight_pos = float(config.get("pos_class_weight", 1.0))
    sample_weight = np.where(y_train_known == 1, class_weight_pos, 1.0)
    model_stage1.fit(x_train_known, y_train_known, sample_weight=sample_weight)

    unknown_mask = y_train == -1
    reliable_negative_max_prob = float(config.get("pu_reliable_negative_max_prob", 0.15))
    reliable_neg_mask = np.zeros(len(train_df), dtype=bool)
    if unknown_mask.any():
        p_unknown = _predict_raw(model_stage1, train_df.loc[unknown_mask, features])
        reliable_idx = train_df.loc[unknown_mask].index[p_unknown <= reliable_negative_max_prob]
        reliable_neg_mask[reliable_idx.to_numpy()] = True

    y_stage2 = y_train.copy()
    y_stage2[reliable_neg_mask] = 0
    stage2_known = y_stage2 >= 0

    model = _build_model(config)
    x_stage2 = train_df.loc[stage2_known, features]
    y_stage2_known = y_stage2[stage2_known]
    sample_weight_stage2 = np.where(y_stage2_known == 1, class_weight_pos, 1.0)
    model.fit(x_stage2, y_stage2_known, sample_weight=sample_weight_stage2)

    # Fit calibration on eval known labels.
    eval_known_df = eval_df.loc[known_eval].copy()
    y_eval_known = y_eval[known_eval]
    if len(eval_known_df) == 0:
        raise ValueError("No known labels in eval split; cannot calibrate or compute promotion gates.")
    raw_eval = _predict_raw(model, eval_known_df[features])

    global_method = str(config.get("global_calibration", "platt"))
    if np.unique(y_eval_known).size < 2:
        LOGGER.warning(
            "Eval known labels contain a single class. Falling back to identity calibration."
        )
        global_calibrator = {"type": "identity", "model": None}
    else:
        global_calibrator = _fit_calibrator(raw_eval, y_eval_known, method=global_method)

    slice_min_samples = int(config.get("slice_calibration_min_samples", 50))
    slice_calibrators: dict[str, Dict[str, Any]] = {}

    if not eval_known_df.empty:
        eval_known_df = eval_known_df.copy()
        eval_known_df["_raw_score"] = raw_eval
        eval_known_df["_y"] = y_eval_known
        for key, g in eval_known_df.groupby(slice_cols, dropna=False):
            if len(g) < slice_min_samples or g["_y"].nunique() < 2:
                continue
            method = "isotonic" if len(g) >= int(config.get("isotonic_min_samples", 150)) else "platt"
            key_label = "|".join(
                f"{col}={val}"
                for col, val in zip(slice_cols, key if isinstance(key, tuple) else (key,), strict=False)
            )
            slice_calibrators[key_label] = _fit_calibrator(
                g["_raw_score"].to_numpy(dtype=float),
                g["_y"].to_numpy(dtype=int),
                method=method,
            )

    # Evaluate using slice calibrators when available.
    calibrated_scores = np.zeros(len(eval_known_df), dtype=float)
    for idx, row in enumerate(eval_known_df.itertuples(index=False)):
        row_series = pd.Series(row._asdict())
        key = _slice_key(row_series, slice_cols)
        cal = slice_calibrators.get(key, global_calibrator)
        calibrated_scores[idx] = float(_apply_calibrator(cal, np.asarray([row_series["_raw_score"]]))[0])

    threshold = float(config.get("decision_threshold", 0.5))
    metrics = _metrics(y_eval_known, calibrated_scores, threshold=threshold)

    # Operational latency estimate: extrapolate per 10k events from eval prediction speed.
    start = time.perf_counter()
    _ = _predict_raw(model, eval_df[features])
    elapsed = max(1e-6, time.perf_counter() - start)
    latency_per_10k = float(elapsed * (10000.0 / max(1, len(eval_df))))

    sensor_bias_pct = None
    if "sensor" in eval_known_df.columns and len(eval_known_df["sensor"].dropna().unique()) >= 2:
        sensor_means = eval_known_df.assign(score=calibrated_scores).groupby("sensor")["score"].mean()
        if not sensor_means.empty:
            sensor_bias_pct = float((sensor_means.max() - sensor_means.min()) * 100.0)

    gates = {
        "event_recall_min": 0.92,
        "event_precision_min": 0.75,
        "global_f1_min": 0.85,
        "roc_auc_min": 0.95,
        "latency_per_10k_max_seconds": 300.0,
        "min_event_positives": int(config.get("min_event_positives", 50)),
        "min_event_negatives": int(config.get("min_event_negatives", 50)),
        "sensor_bias_pct_max": 5.0,
    }
    n_pos = int((y_eval_known == 1).sum())
    n_neg = int((y_eval_known == 0).sum())
    gate_results = {
        "event_recall": {"value": metrics["recall"], "pass": metrics["recall"] >= gates["event_recall_min"]},
        "event_precision": {
            "value": metrics["precision"],
            "pass": metrics["precision"] >= gates["event_precision_min"],
        },
        "global_f1": {"value": metrics["f1"], "pass": metrics["f1"] >= gates["global_f1_min"]},
        "roc_auc": {
            "value": metrics["roc_auc"],
            "pass": (metrics["roc_auc"] is not None and metrics["roc_auc"] >= gates["roc_auc_min"]),
        },
        "latency_per_10k_seconds": {
            "value": latency_per_10k,
            "pass": latency_per_10k <= gates["latency_per_10k_max_seconds"],
        },
        "min_event_positives": {
            "value": n_pos,
            "pass": n_pos >= gates["min_event_positives"],
        },
        "min_event_negatives": {
            "value": n_neg,
            "pass": n_neg >= gates["min_event_negatives"],
        },
        "sensor_bias_pct": {
            "value": sensor_bias_pct,
            "pass": (sensor_bias_pct is None) or (sensor_bias_pct <= gates["sensor_bias_pct_max"]),
        },
    }
    gate_pass = all(bool(v["pass"]) for v in gate_results.values())

    git_sha = _maybe_git_sha() or "unknown"
    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{git_sha}"
    out_root = config.get("model_output_root", "models/denoiser_v2")
    run_dir = os.path.join(out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    bundle = {
        "model": model,
        "features": features,
        "slice_cols": slice_cols,
        "global_calibrator": global_calibrator,
        "slice_calibrators": slice_calibrators,
        "thresholds": {
            "decision": threshold,
            "strong_filter": float(config.get("strong_filter_threshold", 0.5)),
            "downweight": float(config.get("downweight_threshold", 0.7)),
            "uncertainty_band_low": float(config.get("uncertainty_band_low", 0.45)),
            "uncertainty_band_high": float(config.get("uncertainty_band_high", 0.55)),
        },
        "latency_per_10k_seconds": latency_per_10k,
        "run_id": run_name,
    }

    joblib.dump(bundle, os.path.join(run_dir, "model_bundle.pkl"))
    joblib.dump(model, os.path.join(run_dir, "model.pkl"))

    with open(os.path.join(run_dir, "feature_list.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    training_summary = {
        "run_id": run_name,
        "model_backend": "xgboost" if _HAS_XGBOOST else "hist_gradient_boosting",
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_known_rows": int(known_train.sum()),
        "train_unknown_rows": int((y_train == -1).sum()),
        "reliable_negative_rows": int(reliable_neg_mask.sum()),
        "eval_known_rows": int(known_eval.sum()),
        "metrics": metrics,
        "latency_per_10k_seconds": latency_per_10k,
        "sensor_bias_pct": sensor_bias_pct,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    gate_report = {
        "run_id": run_name,
        "pass": gate_pass,
        "thresholds": gates,
        "results": gate_results,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "gate_report.json"), "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=2)

    with open(os.path.join(run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    metadata = {
        "run_id": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "config": config,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "gate_pass": gate_pass,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train denoiser v2 (event-level PU + calibration).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = train_denoiser_v2(config)
    print(run_dir)


if __name__ == "__main__":
    main()
