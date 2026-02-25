"""Train event-level denoiser v2 with covered-first and leakage-safe PU options."""

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


def _load_snapshot(path: str, *, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    read_columns = list(dict.fromkeys(columns))
    def _read_with_fallback(parquet_path: str) -> pd.DataFrame:
        try:
            return pd.read_parquet(parquet_path, columns=read_columns)
        except Exception:
            full = pd.read_parquet(parquet_path)
            keep = [c for c in read_columns if c in full.columns]
            return full[keep]

    if os.path.isdir(path):
        train_path = os.path.join(path, "train.parquet")
        eval_path = os.path.join(path, "eval.parquet")
        train = _read_with_fallback(train_path)
        eval_df = _read_with_fallback(eval_path)
        return train, eval_df
    full_columns = list(dict.fromkeys(read_columns + ["start_time"]))
    try:
        full = pd.read_parquet(path, columns=full_columns)
    except Exception:
        full = pd.read_parquet(path)
        keep = [c for c in full_columns if c in full.columns]
        full = full[keep]
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


def _feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].astype(np.float32)


def _build_model(
    config: Dict[str, Any],
    *,
    backend_override: str | None = None,
    model_params_override: dict[str, Any] | None = None,
) -> Any:
    backend = str(backend_override or config.get("model_backend", "xgboost")).strip().lower()
    model_params = dict(config.get("model_params", {}))
    if model_params_override:
        model_params.update(model_params_override)

    if backend in {"xgboost", "xgboost_pu_bagging"}:
        if not _HAS_XGBOOST:
            raise RuntimeError(
                "XGBoost backend selected but XGBoost is unavailable. "
                "Install/fix runtime dependencies (e.g., libomp on macOS)."
            )
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

    if backend in {"hist_gradient_boosting", "hgb"}:
        defaults = {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "l2_regularization": 0.01,
            "random_state": int(config.get("seed", 42)),
        }
        defaults.update(model_params)
        return HistGradientBoostingClassifier(**defaults)

    raise ValueError(
        f"Unsupported model_backend={backend!r}. "
        "Use 'xgboost', 'xgboost_pu_bagging', or 'hist_gradient_boosting'."
    )


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
    ctype = cal.get("type")
    if ctype == "isotonic" and cal.get("model") is not None:
        return np.asarray(cal["model"].predict(scores), dtype=float)
    if ctype == "platt" and cal.get("model") is not None:
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


def _select_scope(df: pd.DataFrame, coverage_scope: str) -> pd.DataFrame:
    scope = str(coverage_scope).strip().lower()
    if scope != "covered":
        return df.copy()
    if "truth_covered_mask" not in df.columns:
        raise ValueError("coverage_scope=covered requires truth_covered_mask column in snapshot")
    return df[df["truth_covered_mask"].fillna(False).astype(bool)].copy()


def _split_calibration_eval_holdout(
    eval_known_df: pd.DataFrame,
    *,
    calibration_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if eval_known_df.empty:
        return eval_known_df.copy(), eval_known_df.copy()

    if "start_time" in eval_known_df.columns:
        ordered = eval_known_df.sort_values("start_time").reset_index(drop=True)
    else:
        ordered = eval_known_df.reset_index(drop=True)

    n = len(ordered)
    if n < 4:
        return ordered.copy(), ordered.copy()

    cut = int(n * float(calibration_fraction))
    cut = max(1, min(n - 1, cut))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def _fit_legacy_two_stage(
    config: Dict[str, Any],
    *,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
) -> tuple[Any, dict[str, Any]]:
    known_train = y_train >= 0
    if not known_train.any():
        raise ValueError("No known labels (POSITIVE/NEGATIVE) available in train split.")

    model_stage1 = _build_model(config)
    x_train_known = _feature_matrix(train_df.loc[known_train], features)
    y_train_known = y_train[known_train]

    class_weight_pos = float(config.get("pos_class_weight", 1.0))
    sample_weight = np.where(y_train_known == 1, class_weight_pos, 1.0)
    model_stage1.fit(x_train_known, y_train_known, sample_weight=sample_weight)

    unknown_mask = y_train == -1
    reliable_negative_max_prob = float(config.get("pu_reliable_negative_max_prob", 0.15))
    reliable_neg_mask = np.zeros(len(train_df), dtype=bool)
    if unknown_mask.any():
        p_unknown = _predict_raw(model_stage1, _feature_matrix(train_df.loc[unknown_mask], features))
        unknown_positions = np.flatnonzero(unknown_mask)
        reliable_neg_mask[unknown_positions[p_unknown <= reliable_negative_max_prob]] = True

    y_stage2 = y_train.copy()
    y_stage2[reliable_neg_mask] = 0
    stage2_known = y_stage2 >= 0

    model = _build_model(config)
    x_stage2 = _feature_matrix(train_df.loc[stage2_known], features)
    y_stage2_known = y_stage2[stage2_known]
    sample_weight_stage2 = np.where(y_stage2_known == 1, class_weight_pos, 1.0)
    model.fit(x_stage2, y_stage2_known, sample_weight=sample_weight_stage2)

    stats = {
        "method": "legacy_two_stage",
        "reliable_negative_rows": int(reliable_neg_mask.sum()),
        "stage2_known_rows": int(stage2_known.sum()),
    }
    return model, stats


def _fit_teacher_student_pu(
    config: Dict[str, Any],
    *,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    rng: np.random.Generator,
) -> tuple[Any, dict[str, Any]]:
    known_mask = y_train >= 0
    pos_mask = y_train == 1
    neg_mask = y_train == 0
    unknown_mask = y_train == -1

    if int(pos_mask.sum()) == 0 or int(neg_mask.sum()) == 0:
        raise ValueError("xgboost_pu_bagging requires both known POSITIVE and known NEGATIVE in train_core.")

    x_pos = _feature_matrix(train_df.loc[pos_mask], features)
    x_neg = _feature_matrix(train_df.loc[neg_mask], features)
    x_unknown = _feature_matrix(train_df.loc[unknown_mask], features)

    pu_cfg = dict(config.get("pu_bagging", {}))
    num_bags = max(1, int(pu_cfg.get("num_bags", 15)))
    unlabeled_multiplier = max(1.0, float(pu_cfg.get("unlabeled_multiplier", 4)))
    min_oob_votes = max(1, int(pu_cfg.get("min_oob_votes", 3)))
    pos_threshold = float(pu_cfg.get("pos_threshold", 0.70))
    neg_threshold = float(pu_cfg.get("neg_threshold", 0.30))

    class_weight_pos = float(config.get("pos_class_weight", 1.0))

    unknown_count = len(x_unknown)
    score_sums = np.zeros(unknown_count, dtype=float)
    vote_counts = np.zeros(unknown_count, dtype=np.int32)

    teacher_params = dict(config.get("model_params", {}))
    teacher_params["max_depth"] = int(pu_cfg.get("teacher_max_depth", 4))
    teacher_params["colsample_bytree"] = float(pu_cfg.get("teacher_colsample_bytree", 0.8))

    teacher_sample_size = 0
    if unknown_count > 0:
        teacher_sample_size = min(unknown_count, max(1, int(unlabeled_multiplier * len(x_pos))))

    for _ in range(num_bags):
        sampled_unknown: np.ndarray
        if unknown_count > 0:
            sampled_unknown = rng.choice(unknown_count, size=teacher_sample_size, replace=False)
        else:
            sampled_unknown = np.asarray([], dtype=int)

        x_parts = [x_pos, x_neg]
        y_parts = [
            np.ones(len(x_pos), dtype=int),
            np.zeros(len(x_neg), dtype=int),
        ]
        if sampled_unknown.size > 0:
            x_parts.append(x_unknown.iloc[sampled_unknown])
            y_parts.append(np.zeros(sampled_unknown.size, dtype=int))

        x_teacher = pd.concat(x_parts, ignore_index=True)
        y_teacher = np.concatenate(y_parts)
        w_teacher = np.where(y_teacher == 1, class_weight_pos, 1.0)

        teacher = _build_model(
            config,
            backend_override="xgboost",
            model_params_override=teacher_params,
        )
        teacher.fit(x_teacher, y_teacher, sample_weight=w_teacher)

        if unknown_count == 0:
            continue

        in_bag = np.zeros(unknown_count, dtype=bool)
        in_bag[sampled_unknown] = True
        oob_idx = np.flatnonzero(~in_bag)
        if oob_idx.size == 0:
            continue

        oob_scores = _predict_raw(teacher, x_unknown.iloc[oob_idx])
        score_sums[oob_idx] += oob_scores
        vote_counts[oob_idx] += 1

    if unknown_count > 0:
        oob_mean, valid_votes = _oob_mean_scores(score_sums, vote_counts, min_oob_votes=min_oob_votes)
        pseudo_pos_mask = valid_votes & (oob_mean >= pos_threshold)
        pseudo_neg_mask = valid_votes & (oob_mean <= neg_threshold)
        ignored_mask = ~(pseudo_pos_mask | pseudo_neg_mask)
    else:
        pseudo_pos_mask = np.asarray([], dtype=bool)
        pseudo_neg_mask = np.asarray([], dtype=bool)
        ignored_mask = np.asarray([], dtype=bool)
        vote_counts = np.asarray([], dtype=np.int32)

    x_parts = [x_pos, x_neg]
    y_parts = [
        np.ones(len(x_pos), dtype=int),
        np.zeros(len(x_neg), dtype=int),
    ]

    if unknown_count > 0 and int(pseudo_pos_mask.sum()) > 0:
        x_parts.append(x_unknown.iloc[pseudo_pos_mask])
        y_parts.append(np.ones(int(pseudo_pos_mask.sum()), dtype=int))
    if unknown_count > 0 and int(pseudo_neg_mask.sum()) > 0:
        x_parts.append(x_unknown.iloc[pseudo_neg_mask])
        y_parts.append(np.zeros(int(pseudo_neg_mask.sum()), dtype=int))

    x_student = pd.concat(x_parts, ignore_index=True)
    y_student = np.concatenate(y_parts)
    w_student = np.where(y_student == 1, class_weight_pos, 1.0)

    student = _build_model(config, backend_override="xgboost")
    student.fit(x_student, y_student, sample_weight=w_student)

    stats = {
        "method": "teacher_student_oob",
        "bag_count": int(num_bags),
        "train_known_rows": int(known_mask.sum()),
        "train_unknown_rows": int(unknown_count),
        "teacher_sample_size": int(teacher_sample_size),
        "min_oob_votes": int(min_oob_votes),
        "oob_eligible_rows": int((vote_counts >= min_oob_votes).sum()) if unknown_count > 0 else 0,
        "oob_mean_votes": float(vote_counts.mean()) if unknown_count > 0 else 0.0,
        "pseudo_positive_rows": int(pseudo_pos_mask.sum()) if unknown_count > 0 else 0,
        "pseudo_negative_rows": int(pseudo_neg_mask.sum()) if unknown_count > 0 else 0,
        "ignored_rows": int(ignored_mask.sum()) if unknown_count > 0 else 0,
        "pos_threshold": float(pos_threshold),
        "neg_threshold": float(neg_threshold),
    }
    return student, stats


def _oob_mean_scores(
    score_sums: np.ndarray,
    vote_counts: np.ndarray,
    *,
    min_oob_votes: int,
) -> tuple[np.ndarray, np.ndarray]:
    min_votes = max(1, int(min_oob_votes))
    valid_votes = np.asarray(vote_counts, dtype=int) >= min_votes
    means = np.full(len(vote_counts), np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        means[valid_votes] = np.asarray(score_sums, dtype=float)[valid_votes] / np.asarray(
            vote_counts, dtype=float
        )[valid_votes]
    return means, valid_votes


def _calibrate(
    *,
    config: Dict[str, Any],
    model: Any,
    calibration_df: pd.DataFrame,
    features: list[str],
    slice_cols: list[str],
    label_col: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], np.ndarray, np.ndarray]:
    y_cal_all = _map_labels(calibration_df, label_col)
    known_cal_mask = y_cal_all >= 0
    cal_known_df = calibration_df.loc[known_cal_mask].copy()
    y_cal = y_cal_all[known_cal_mask]
    if len(cal_known_df) == 0:
        raise ValueError("No known labels in calibration holdout.")

    raw_cal = _predict_raw(model, _feature_matrix(cal_known_df, features))

    global_method = str(config.get("global_calibration", "platt"))
    if np.unique(y_cal).size < 2:
        LOGGER.warning("Calibration holdout has a single class; using identity calibration.")
        global_cal = {"type": "identity", "model": None}
    else:
        global_cal = _fit_calibrator(raw_cal, y_cal, method=global_method)

    slice_min_samples = int(config.get("slice_calibration_min_samples", 50))
    slice_cals: dict[str, Dict[str, Any]] = {}

    cal_tmp = cal_known_df.copy()
    cal_tmp["_raw_score"] = raw_cal
    cal_tmp["_y"] = y_cal

    for key, grp in cal_tmp.groupby(slice_cols, dropna=False):
        if len(grp) < slice_min_samples or grp["_y"].nunique() < 2:
            continue
        method = "isotonic" if len(grp) >= int(config.get("isotonic_min_samples", 150)) else "platt"
        key_label = "|".join(
            f"{col}={val}"
            for col, val in zip(slice_cols, key if isinstance(key, tuple) else (key,), strict=False)
        )
        slice_cals[key_label] = _fit_calibrator(
            grp["_raw_score"].to_numpy(dtype=float),
            grp["_y"].to_numpy(dtype=int),
            method=method,
        )

    return global_cal, slice_cals, y_cal, raw_cal


def _apply_slice_calibration(
    *,
    eval_df: pd.DataFrame,
    raw_scores: np.ndarray,
    slice_cols: list[str],
    global_calibrator: dict[str, Any],
    slice_calibrators: dict[str, dict[str, Any]],
) -> np.ndarray:
    out = np.zeros(len(eval_df), dtype=float)
    for idx, (_, row_series) in enumerate(eval_df.iterrows()):
        key = _slice_key(row_series, slice_cols)
        cal = slice_calibrators.get(key, global_calibrator)
        out[idx] = float(_apply_calibrator(cal, np.asarray([raw_scores[idx]]))[0])
    return out


def train_denoiser_v2(config: Dict[str, Any]) -> str:
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model_backend = str(config.get("model_backend", "xgboost")).strip().lower()
    coverage_scope = str(config.get("coverage_scope", "covered")).strip().lower()
    coverage_mask_source = str(config.get("coverage_mask_source", "db_mask")).strip()

    features = list(config["features"])
    label_col = str(config.get("label_column", "event_label"))
    slice_cols = list(config.get("slice_columns", ["sensor", "biome_slice"]))

    required_columns = list(dict.fromkeys(features + [label_col] + slice_cols + ["start_time"]))
    if coverage_scope == "covered":
        required_columns.append("truth_covered_mask")
    train_df, eval_df = _load_snapshot(config["snapshot_path"], columns=required_columns)

    for col in features:
        if col not in train_df.columns:
            train_df[col] = np.nan
        if col not in eval_df.columns:
            eval_df[col] = np.nan

    train_scope = _select_scope(train_df, coverage_scope)
    eval_scope = _select_scope(eval_df, coverage_scope)

    y_train = _map_labels(train_scope, label_col)
    known_train = y_train >= 0
    if not known_train.any():
        raise ValueError("No known labels (POSITIVE/NEGATIVE) available in train scope.")

    if model_backend == "xgboost_pu_bagging":
        model, pu_stats = _fit_teacher_student_pu(
            config,
            train_df=train_scope,
            y_train=y_train,
            features=features,
            rng=rng,
        )
    else:
        model, pu_stats = _fit_legacy_two_stage(
            config,
            train_df=train_scope,
            y_train=y_train,
            features=features,
        )

    eval_known_mask = _map_labels(eval_scope, label_col) >= 0
    eval_known_df = eval_scope.loc[eval_known_mask].copy()
    if eval_known_df.empty:
        raise ValueError("No known labels in eval scope; cannot calibrate or compute promotion gates.")

    calibration_fraction = float(config.get("calibration_holdout_fraction", 0.5))
    calibration_df, eval_holdout_df = _split_calibration_eval_holdout(
        eval_known_df,
        calibration_fraction=calibration_fraction,
    )
    if eval_holdout_df.empty:
        eval_holdout_df = calibration_df.copy()

    global_calibrator, slice_calibrators, y_cal, raw_cal = _calibrate(
        config=config,
        model=model,
        calibration_df=calibration_df,
        features=features,
        slice_cols=slice_cols,
        label_col=label_col,
    )

    y_eval = _map_labels(eval_holdout_df, label_col)
    known_eval_mask = y_eval >= 0
    eval_known_holdout = eval_holdout_df.loc[known_eval_mask].copy()
    y_eval_known = y_eval[known_eval_mask]
    if len(eval_known_holdout) == 0:
        raise ValueError("No known labels in eval holdout after split.")

    raw_eval = _predict_raw(model, _feature_matrix(eval_known_holdout, features))
    calibrated_scores = _apply_slice_calibration(
        eval_df=eval_known_holdout,
        raw_scores=raw_eval,
        slice_cols=slice_cols,
        global_calibrator=global_calibrator,
        slice_calibrators=slice_calibrators,
    )

    threshold = float(config.get("decision_threshold", 0.5))
    metrics = _metrics(y_eval_known, calibrated_scores, threshold=threshold)

    # Operational latency estimate: extrapolate per 10k events from eval prediction speed.
    latency_eval_df = eval_scope if not eval_scope.empty else eval_known_holdout
    start = time.perf_counter()
    _ = _predict_raw(model, _feature_matrix(latency_eval_df, features))
    elapsed = max(1e-6, time.perf_counter() - start)
    latency_per_10k = float(elapsed * (10000.0 / max(1, len(latency_eval_df))))

    sensor_bias_pct = None
    if "sensor" in eval_known_holdout.columns and len(eval_known_holdout["sensor"].dropna().unique()) >= 2:
        sensor_means = eval_known_holdout.assign(score=calibrated_scores).groupby("sensor")["score"].mean()
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
        "model_backend": model_backend,
        "gate_scope": coverage_scope,
        "coverage_mask_source": coverage_mask_source,
    }

    joblib.dump(bundle, os.path.join(run_dir, "model_bundle.pkl"))
    joblib.dump(model, os.path.join(run_dir, "model.pkl"))

    with open(os.path.join(run_dir, "feature_list.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    training_summary = {
        "run_id": run_name,
        "model_backend": model_backend,
        "coverage_scope": coverage_scope,
        "coverage_mask_source": coverage_mask_source,
        "train_rows": int(len(train_df)),
        "train_scope_rows": int(len(train_scope)),
        "eval_rows": int(len(eval_df)),
        "eval_scope_rows": int(len(eval_scope)),
        "train_known_rows": int(known_train.sum()),
        "train_unknown_rows": int((y_train == -1).sum()),
        "calibration_known_rows": int(len(y_cal)),
        "eval_known_rows": int(len(y_eval_known)),
        "metrics": metrics,
        "latency_per_10k_seconds": latency_per_10k,
        "sensor_bias_pct": sensor_bias_pct,
        "pu_stats": pu_stats,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    gate_report = {
        "run_id": run_name,
        "pass": gate_pass,
        "gate_scope": coverage_scope,
        "coverage_mask_source": coverage_mask_source,
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
