"""Train event-level denoiser v2 with covered-first and leakage-safe PU options."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from ml.calibration import (
    apply_binary_probability_calibrator,
    fit_binary_probability_calibrator,
    optimize_threshold_for_target_recall,
)
from ml.denoiser.coverage_authority import get_coverage_freshness, require_coverage_freshness
from ml.denoiser.runtime_contract import DenoiserRuntimeContract, write_contract
from ml.parquet_io import read_parquet_with_fallback

try:
    from xgboost import DMatrix, XGBClassifier

    _HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional fallback
    DMatrix = None
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


def _load_snapshot(
    path: str,
    *,
    columns: list[str],
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load snapshot and split into train / calibration / eval using temporal ordering.

    Default fractions: training=first 60%, calibration=middle 20%, eval=final 20%.
    For directory snapshots (``train.parquet`` + ``eval.parquet``), both files are
    concatenated before re-splitting so the temporal boundaries are re-established.
    """
    read_columns = list(dict.fromkeys(columns))

    def _read_with_fallback(parquet_path: str) -> pd.DataFrame:
        try:
            return read_parquet_with_fallback(parquet_path, columns=read_columns)
        except Exception:
            full = read_parquet_with_fallback(parquet_path)
            keep = [c for c in read_columns if c in full.columns]
            return full[keep]

    full_columns = list(dict.fromkeys(read_columns + ["start_time"]))

    if os.path.isdir(path):
        train_path = os.path.join(path, "train.parquet")
        eval_path = os.path.join(path, "eval.parquet")
        full = pd.concat(
            [_read_with_fallback(train_path), _read_with_fallback(eval_path)],
            ignore_index=True,
            sort=False,
        )
    else:
        try:
            full = read_parquet_with_fallback(path, columns=full_columns)
        except Exception:
            full = read_parquet_with_fallback(path)
            keep = [c for c in full_columns if c in full.columns]
            full = full[keep]

    return _temporal_3way_split(full, train_fraction=train_fraction, calibration_fraction=calibration_fraction)


def _map_labels(df: pd.DataFrame, label_col: str = "event_label") -> np.ndarray:
    mapping = {"POSITIVE": 1, "NEGATIVE": 0}
    labels = df[label_col].map(mapping).fillna(-1).astype(int).to_numpy()
    return labels


def _feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].astype(np.float32)


def _label_weights(df: pd.DataFrame) -> np.ndarray:
    """Return per-row label_weight as a float array, defaulting to 1.0 if absent."""
    if "label_weight" in df.columns:
        return df["label_weight"].fillna(1.0).to_numpy(dtype=np.float32)
    return np.ones(len(df), dtype=np.float32)


def _stratified_downsample(
    df: pd.DataFrame,
    *,
    max_rows: int,
    strat_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()

    work = df.copy()
    use_cols = [c for c in strat_cols if c in work.columns]
    if not use_cols:
        chosen_idx = rng.choice(work.index.to_numpy(dtype=int), size=max_rows, replace=False)
        return work.loc[chosen_idx].copy()

    keys = work[use_cols].fillna("unknown").astype(str).agg("|".join, axis=1)
    key_counts = keys.value_counts(dropna=False)
    alloc_float = (key_counts / float(key_counts.sum())) * int(max_rows)
    alloc = np.floor(alloc_float).astype(int)
    remainder = int(max_rows - int(alloc.sum()))
    if remainder > 0:
        frac = (alloc_float - alloc).sort_values(ascending=False)
        for group in frac.index[:remainder]:
            alloc[group] = int(alloc[group]) + 1

    sampled_idx: list[int] = []
    for group, count in alloc.items():
        draw = int(count)
        if draw <= 0:
            continue
        group_idx = keys.index[keys == group].to_numpy(dtype=int)
        take = min(draw, int(group_idx.size))
        if take <= 0:
            continue
        sampled_idx.extend(rng.choice(group_idx, size=take, replace=False).tolist())

    sampled_unique = np.asarray(sorted(set(sampled_idx)), dtype=int)
    if sampled_unique.size > max_rows:
        sampled_unique = rng.choice(sampled_unique, size=max_rows, replace=False)
    elif sampled_unique.size < max_rows:
        pool = np.setdiff1d(work.index.to_numpy(dtype=int), sampled_unique, assume_unique=False)
        top_up = min(int(max_rows - sampled_unique.size), int(pool.size))
        if top_up > 0:
            sampled_unique = np.concatenate(
                [sampled_unique, rng.choice(pool, size=top_up, replace=False)]
            )
    return work.loc[sampled_unique].copy()


def _temporal_3way_split(
    df: pd.DataFrame,
    *,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into (train, calibration, eval) in chronological order.

    Splits at ``start_time`` quantiles when the column is present and quantile
    boundaries produce non-empty partitions; falls back to positional slicing.
    The caller is responsible for checking that no returned partition is empty.
    """
    if "start_time" in df.columns:
        ordered = df.sort_values("start_time").reset_index(drop=True)
        split_train_dt = ordered["start_time"].quantile(train_fraction)
        split_cal_dt = ordered["start_time"].quantile(calibration_fraction)
        train = ordered[ordered["start_time"] < split_train_dt].copy()
        cal = ordered[
            (ordered["start_time"] >= split_train_dt) & (ordered["start_time"] < split_cal_dt)
        ].copy()
        eval_df = ordered[ordered["start_time"] >= split_cal_dt].copy()
        if not train.empty and not cal.empty and not eval_df.empty:
            return train, cal, eval_df
        # Fall through to positional when quantile boundaries collapse (e.g. many duplicate timestamps)
    else:
        ordered = df

    n = len(ordered)
    cut_train = int(n * train_fraction)
    cut_cal = int(n * calibration_fraction)
    return (
        ordered.iloc[:cut_train].copy(),
        ordered.iloc[cut_train:cut_cal].copy(),
        ordered.iloc[cut_cal:].copy(),
    )


def _temporal_2way_split(
    df: pd.DataFrame,
    *,
    eval_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into (eval, validation) in chronological order.

    Splits at ``start_time`` quantile when the column is present; falls back to positional slicing.
    Preserves temporal ordering within each partition.

    Args:
        df: DataFrame to split.
        eval_fraction: Fraction of data for eval set; remainder goes to validation.

    Returns:
        (eval_df, validation_df) tuple.
    """
    if "start_time" in df.columns:
        ordered = df.sort_values("start_time").reset_index(drop=True)
        split_dt = ordered["start_time"].quantile(eval_fraction)
        eval_df = ordered[ordered["start_time"] < split_dt].copy()
        val_df = ordered[ordered["start_time"] >= split_dt].copy()
        if not eval_df.empty and not val_df.empty:
            return eval_df, val_df
        # Fall through to positional when quantile boundaries collapse
    else:
        ordered = df

    n = len(ordered)
    cut_eval = int(n * eval_fraction)
    return (
        ordered.iloc[:cut_eval].copy(),
        ordered.iloc[cut_eval:].copy(),
    )


def _apply_micro_batch(
    *,
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: Dict[str, Any],
    label_col: str,
    default_strat_cols: list[str],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    micro_cfg = dict(config.get("micro_batch", {}))
    if not bool(micro_cfg.get("enabled", False)):
        return train_df, calibration_df, eval_df, {"enabled": False}

    for name, df in [("train_df", train_df), ("calibration_df", calibration_df), ("eval_df", eval_df)]:
        if "start_time" not in df.columns:
            raise ValueError(f"micro_batch requires start_time in {name}.")

    start_raw = micro_cfg.get("start")
    end_raw = micro_cfg.get("end")
    if not start_raw or not end_raw:
        raise ValueError("micro_batch requires both 'start' and 'end' dates.")
    start = pd.Timestamp(str(start_raw), tz="UTC")
    end = pd.Timestamp(str(end_raw), tz="UTC")
    if end <= start:
        raise ValueError(f"Invalid micro_batch window: start={start} end={end}")

    combined = pd.concat(
        [train_df, calibration_df, eval_df], ignore_index=True, sort=False
    )
    combined["start_time"] = pd.to_datetime(combined["start_time"], utc=True, errors="coerce")
    micro = combined[(combined["start_time"] >= start) & (combined["start_time"] < end)].copy()
    if micro.empty:
        raise ValueError(
            f"micro_batch slice produced zero rows for window [{start.isoformat()}, {end.isoformat()})."
        )

    strat_cols = list(micro_cfg.get("stratify_columns", default_strat_cols))
    if label_col not in strat_cols:
        strat_cols = [label_col] + strat_cols
    max_rows = int(micro_cfg.get("max_rows", 30000))
    micro = _stratified_downsample(micro, max_rows=max_rows, strat_cols=strat_cols, rng=rng)
    micro = micro.sort_values("start_time").reset_index(drop=True)

    if label_col not in micro.columns:
        raise ValueError(f"micro_batch requires label column '{label_col}' in snapshot.")
    pos_count = int((micro[label_col] == "POSITIVE").sum())
    min_pos = int(micro_cfg.get("min_positive_rows", 100))
    if pos_count < min_pos:
        raise ValueError(
            "micro_batch has insufficient positives: "
            f"positive_rows={pos_count}, min_positive_rows={min_pos}."
        )

    micro_train, micro_cal, micro_eval = _temporal_3way_split(micro)
    if micro_train.empty or micro_cal.empty or micro_eval.empty:
        raise ValueError(
            f"micro_batch 60/20/20 temporal split produced an empty partition "
            f"(rows={len(micro)}). "
            "Widen the micro_batch window or reduce min_positive_rows."
        )

    stats = {
        "enabled": True,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows_total": int(len(micro)),
        "rows_train": int(len(micro_train)),
        "rows_calibration": int(len(micro_cal)),
        "rows_eval": int(len(micro_eval)),
        "positive_rows_total": int((micro[label_col] == "POSITIVE").sum()),
        "negative_rows_total": int((micro[label_col] == "NEGATIVE").sum()),
        "unknown_rows_total": int((micro[label_col] == "UNKNOWN").sum()),
        "max_rows": int(max_rows),
        "stratify_columns": strat_cols,
    }
    return micro_train, micro_cal, micro_eval, stats


def _compute_shap_top_features(
    *,
    model: Any,
    x: pd.DataFrame,
    top_k: int,
    sample_rows: int,
    rng: np.random.Generator,
) -> list[dict[str, int | float | str]]:
    if int(top_k) <= 0 or x.empty:
        return []
    if DMatrix is None or not hasattr(model, "get_booster"):
        return []

    x_use = x.copy()
    if int(sample_rows) > 0 and len(x_use) > int(sample_rows):
        take_idx = rng.choice(np.arange(len(x_use)), size=int(sample_rows), replace=False)
        x_use = x_use.iloc[np.asarray(take_idx, dtype=int)].copy()
    x_use = x_use.astype(np.float32)

    booster = model.get_booster()
    contrib = booster.predict(
        DMatrix(x_use.to_numpy(dtype=np.float32), feature_names=list(x_use.columns)),
        pred_contribs=True,
    )
    if contrib.ndim != 2 or contrib.shape[0] == 0:
        return []

    # Last column is the bias term for XGBoost pred_contribs.
    feature_contrib = contrib[:, : x_use.shape[1]]
    mean_abs = np.mean(np.abs(feature_contrib), axis=0)
    order = np.argsort(-mean_abs)[: int(top_k)]
    out: list[dict[str, int | float | str]] = []
    for rank, idx in enumerate(order, start=1):
        out.append(
            {
                "rank": int(rank),
                "feature": str(x_use.columns[int(idx)]),
                "mean_abs_shap": float(mean_abs[int(idx)]),
            }
        )
    return out


def _build_model(
    config: Dict[str, Any],
    *,
    backend_override: str | None = None,
    model_params_override: dict[str, Any] | None = None,
) -> Any:
    backend = str(backend_override or config.get("model_backend", "xgboost_pu_bagging")).strip().lower()
    model_params = dict(config.get("model_params", {}))
    if model_params_override:
        model_params.update(model_params_override)

    if backend not in {"xgboost", "xgboost_pu_bagging"}:
        raise ValueError(
            f"Unsupported model_backend={backend!r}. "
            "PU denoiser v2 requires XGBoost backend ('xgboost_pu_bagging')."
        )

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


def _predict_raw(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    pred = np.asarray(model.predict(x), dtype=float)
    return np.clip(pred, 0.0, 1.0)


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


def _ensure_slice_columns(df: pd.DataFrame, slice_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in slice_cols:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown").astype(str)
    return out


def _stratified_majority_sample(
    *,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    ratio_majority_to_positive: float,
    slice_cols: list[str],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    positive_idx = np.flatnonzero(y_train == 1)
    negative_idx = np.flatnonzero(y_train == 0)
    unknown_idx = np.flatnonzero(y_train == -1)
    positive_rows = int(positive_idx.size)
    negative_rows = int(negative_idx.size)
    unknown_rows = int(unknown_idx.size)
    ratio = max(0.0, float(ratio_majority_to_positive))
    # Only downsample known negatives; preserve all unknowns for PU bagging
    # OOB voting. Unknowns are concatenated back after negative downsampling.
    majority_idx = negative_idx
    majority_rows = negative_rows
    if positive_rows == 0 or majority_rows == 0:
        return train_df.copy(), y_train.copy(), {
            "enabled": True,
            "ratio_majority_to_positive": ratio,
            "positive_rows": positive_rows,
            "majority_rows_before_sampling": majority_rows,
            "majority_rows_after_sampling": majority_rows,
            "unknown_rows_preserved": unknown_rows,
            "target_majority_rows": majority_rows,
            "sampling_applied": False,
            "reason": "insufficient_class_rows",
        }

    target_majority = min(majority_rows, int(np.ceil(ratio * positive_rows)))
    if target_majority >= majority_rows:
        return train_df.copy(), y_train.copy(), {
            "enabled": True,
            "ratio_majority_to_positive": ratio,
            "positive_rows": positive_rows,
            "majority_rows_before_sampling": majority_rows,
            "majority_rows_after_sampling": majority_rows,
            "unknown_rows_preserved": unknown_rows,
            "target_majority_rows": target_majority,
            "sampling_applied": False,
            "reason": "target_exceeds_majority",
        }

    majority_positions = np.asarray(majority_idx, dtype=int)
    majority_df = train_df.iloc[majority_positions].copy().reset_index(drop=True)
    use_slice_cols = [c for c in slice_cols if c in majority_df.columns]
    if use_slice_cols:
        keys = majority_df[use_slice_cols].fillna("unknown").astype(str).agg("|".join, axis=1)
    else:
        keys = pd.Series(["__all__"] * len(majority_df), index=majority_df.index)
    key_counts = keys.value_counts(dropna=False)
    alloc_float = (key_counts / max(1, int(key_counts.sum()))) * target_majority
    alloc = np.floor(alloc_float).astype(int)
    remainder = int(target_majority - int(alloc.sum()))
    if remainder > 0:
        frac = (alloc_float - alloc).sort_values(ascending=False)
        top_groups = list(frac.index[:remainder])
        for group in top_groups:
            alloc[group] = int(alloc[group]) + 1

    sampled_majority_idx: list[int] = []
    for group, count in alloc.items():
        if int(count) <= 0:
            continue
        group_local_idx = keys.index[keys == group].to_numpy(dtype=int)
        group_idx = majority_positions[group_local_idx]
        draw = min(int(count), int(group_idx.size))
        if draw <= 0:
            continue
        chosen = rng.choice(group_idx, size=draw, replace=False)
        sampled_majority_idx.extend(chosen.tolist())

    sampled_majority = np.asarray(sampled_majority_idx, dtype=int)
    if sampled_majority.size > target_majority:
        sampled_majority = rng.choice(sampled_majority, size=target_majority, replace=False)
    elif sampled_majority.size < target_majority:
        pool = np.setdiff1d(majority_positions, sampled_majority, assume_unique=False)
        top_up = min(int(target_majority - sampled_majority.size), int(pool.size))
        if top_up > 0:
            sampled_majority = np.concatenate(
                [sampled_majority, rng.choice(pool, size=top_up, replace=False)]
            )

    # Reassemble: positives + downsampled negatives + ALL unknowns
    sampled_idx = np.concatenate([positive_idx, sampled_majority, unknown_idx])
    sampled_df = train_df.iloc[sampled_idx].copy().reset_index(drop=True)
    sampled_y = y_train[sampled_idx]

    return sampled_df, sampled_y, {
        "enabled": True,
        "ratio_majority_to_positive": ratio,
        "positive_rows": positive_rows,
        "majority_rows_before_sampling": negative_rows,
        "majority_rows_after_sampling": int(sampled_majority.size),
        "unknown_rows_preserved": unknown_rows,
        "target_majority_rows": int(target_majority),
        "sampling_applied": True,
        "stratify_columns_used": use_slice_cols,
    }


def _build_adasyn_samples(
    *,
    features_df: pd.DataFrame,
    minority_mask: np.ndarray,
    synthetic_rows: int,
    k_neighbors: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if synthetic_rows <= 0:
        return pd.DataFrame(columns=features_df.columns)
    x_all = features_df.to_numpy(dtype=np.float32, copy=True)
    if np.isnan(x_all).any():
        col_medians = np.nanmedian(x_all, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        nan_rows, nan_cols = np.where(np.isnan(x_all))
        x_all[nan_rows, nan_cols] = col_medians[nan_cols]
    min_idx = np.flatnonzero(minority_mask)
    maj_idx = np.flatnonzero(~minority_mask)
    if min_idx.size < 2 or maj_idx.size == 0:
        return pd.DataFrame(columns=features_df.columns)

    k = max(1, min(int(k_neighbors), int(len(x_all) - 1)))
    all_nn = NearestNeighbors(n_neighbors=k + 1)
    all_nn.fit(x_all)
    all_neighbors = all_nn.kneighbors(x_all[min_idx], return_distance=False)

    r = np.zeros(min_idx.size, dtype=float)
    for i, neigh in enumerate(all_neighbors):
        neigh = neigh[neigh != min_idx[i]][:k]
        if neigh.size == 0:
            continue
        r[i] = float(np.isin(neigh, maj_idx).sum()) / float(neigh.size)

    if np.allclose(r.sum(), 0.0):
        probs = np.full(min_idx.size, 1.0 / float(min_idx.size))
    else:
        probs = r / r.sum()

    synth_alloc_float = probs * int(synthetic_rows)
    synth_alloc = np.floor(synth_alloc_float).astype(int)
    rem = int(synthetic_rows - int(synth_alloc.sum()))
    if rem > 0:
        frac = np.argsort(-(synth_alloc_float - synth_alloc))
        synth_alloc[frac[:rem]] += 1

    k_min = max(1, min(int(k_neighbors), int(min_idx.size - 1)))
    min_nn = NearestNeighbors(n_neighbors=k_min + 1)
    min_nn.fit(x_all[min_idx])
    min_neighbors = min_nn.kneighbors(x_all[min_idx], return_distance=False)

    generated: list[np.ndarray] = []
    for local_i, n_synth in enumerate(synth_alloc):
        if int(n_synth) <= 0:
            continue
        base = x_all[min_idx[local_i]]
        neigh_local = min_neighbors[local_i]
        neigh_local = neigh_local[neigh_local != local_i][:k_min]
        if neigh_local.size == 0:
            continue
        for _ in range(int(n_synth)):
            partner_local = int(rng.choice(neigh_local))
            partner = x_all[min_idx[partner_local]]
            lam = float(rng.random())
            generated.append(base + lam * (partner - base))

    if not generated:
        return pd.DataFrame(columns=features_df.columns)
    return pd.DataFrame(np.vstack(generated), columns=features_df.columns)


def _apply_adasyn_high_intensity(
    *,
    config: Dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    adasyn_cfg = dict(config.get("adasyn", {}))
    enabled = bool(adasyn_cfg.get("enabled", True))
    if not enabled:
        return x_train, y_train, {"enabled": False, "generated_rows": 0, "reason": "disabled"}

    intensity_feature = str(adasyn_cfg.get("intensity_feature", "frp_max"))
    high_q = float(adasyn_cfg.get("high_intensity_quantile", 0.9))
    multiplier = max(1.0, float(adasyn_cfg.get("multiplier", 2.0)))
    min_high_rows = max(2, int(adasyn_cfg.get("min_high_intensity_rows", 10)))
    k_neighbors = max(1, int(adasyn_cfg.get("k_neighbors", 5)))
    max_synth_rows = max(0, int(adasyn_cfg.get("max_synthetic_rows", 20000)))

    if intensity_feature not in x_train.columns:
        return x_train, y_train, {
            "enabled": True,
            "generated_rows": 0,
            "reason": "missing_intensity_feature",
            "intensity_feature": intensity_feature,
        }

    pos_mask = y_train == 1
    if int(pos_mask.sum()) < min_high_rows:
        return x_train, y_train, {
            "enabled": True,
            "generated_rows": 0,
            "reason": "insufficient_positive_rows",
            "positive_rows": int(pos_mask.sum()),
        }

    intensity = pd.to_numeric(x_train.loc[pos_mask, intensity_feature], errors="coerce")
    valid = intensity.notna()
    if int(valid.sum()) < min_high_rows:
        return x_train, y_train, {
            "enabled": True,
            "generated_rows": 0,
            "reason": "insufficient_valid_intensity_rows",
            "intensity_feature": intensity_feature,
            "valid_intensity_rows": int(valid.sum()),
        }

    threshold = float(np.nanquantile(intensity[valid], high_q))
    pos_positions = np.flatnonzero(pos_mask)
    high_pos_positions = pos_positions[np.where(intensity.fillna(-np.inf).to_numpy(dtype=float) >= threshold)[0]]
    high_mask = np.zeros(len(y_train), dtype=bool)
    high_mask[high_pos_positions] = True

    high_rows = int(high_mask.sum())
    if high_rows < min_high_rows:
        return x_train, y_train, {
            "enabled": True,
            "generated_rows": 0,
            "reason": "insufficient_high_intensity_rows",
            "high_intensity_rows": high_rows,
            "threshold": threshold,
            "intensity_feature": intensity_feature,
        }

    target_high_rows = int(np.ceil(high_rows * multiplier))
    requested_synth = max(0, target_high_rows - high_rows)
    requested_synth = min(requested_synth, max_synth_rows)
    if requested_synth <= 0:
        return x_train, y_train, {
            "enabled": True,
            "generated_rows": 0,
            "reason": "target_already_met",
            "high_intensity_rows": high_rows,
            "target_high_intensity_rows": target_high_rows,
        }

    synth = _build_adasyn_samples(
        features_df=x_train,
        minority_mask=high_mask,
        synthetic_rows=requested_synth,
        k_neighbors=k_neighbors,
        rng=rng,
    )
    if synth.empty:
        return x_train, y_train, {
            "enabled": True,
            "generated_rows": 0,
            "reason": "adasyn_generation_empty",
            "high_intensity_rows": high_rows,
            "target_high_intensity_rows": target_high_rows,
            "requested_synth_rows": requested_synth,
        }

    x_aug = pd.concat([x_train, synth], ignore_index=True)
    y_aug = np.concatenate([y_train, np.ones(len(synth), dtype=int)])
    return x_aug, y_aug, {
        "enabled": True,
        "generated_rows": int(len(synth)),
        "intensity_feature": intensity_feature,
        "high_intensity_quantile": high_q,
        "high_intensity_threshold": threshold,
        "high_intensity_rows_before": high_rows,
        "target_high_intensity_rows": target_high_rows,
        "multiplier": multiplier,
        "k_neighbors": k_neighbors,
    }


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
    per_row_weight = _label_weights(train_df.loc[known_train])
    sample_weight = np.where(y_train_known == 1, class_weight_pos, 1.0) * per_row_weight
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
    per_row_weight_stage2 = _label_weights(train_df.loc[stage2_known])
    sample_weight_stage2 = np.where(y_stage2_known == 1, class_weight_pos, 1.0) * per_row_weight_stage2
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
    slice_cols: list[str],
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
    pseudo_negative_max_ratio = float(pu_cfg.get("pseudo_negative_max_ratio", 3.0))
    max_pseudo_negative_rows = int(pu_cfg.get("max_pseudo_negative_rows", 0))
    min_pseudo_positive_rows = int(pu_cfg.get("min_pseudo_positive_rows", 500))
    oob_margin_min = max(0.0, float(pu_cfg.get("oob_margin_min", 0.05)))
    spy_fraction = min(0.49, max(0.0, float(pu_cfg.get("spy_fraction", 0.15))))
    spy_min_rows = max(0, int(pu_cfg.get("spy_min_rows", 30)))
    spy_pos_quantile = min(1.0, max(0.0, float(pu_cfg.get("spy_pos_quantile", 0.65))))
    spy_neg_quantile = min(1.0, max(0.0, float(pu_cfg.get("spy_neg_quantile", 0.10))))

    class_weight_pos = float(config.get("pos_class_weight", 1.0))

    unknown_count = len(x_unknown)
    score_sums = np.zeros(unknown_count, dtype=float)
    vote_counts = np.zeros(unknown_count, dtype=np.int32)
    spy_scores_by_bag: list[np.ndarray] = []

    # Build weighted sampling probabilities for unknowns: boost high-FRP events
    # so they appear more often in teacher bags without using FRP as a label.
    high_frp_pu_boost = float(pu_cfg.get("high_frp_sampling_boost", 2.5))
    high_frp_pu_threshold_mw = float(pu_cfg.get("high_frp_sampling_threshold_mw", 100.0))
    unknown_sample_weights: np.ndarray | None = None
    if unknown_count > 0 and "frp_max" in x_unknown.columns and high_frp_pu_boost > 1.0:
        frp_vals = pd.to_numeric(x_unknown["frp_max"], errors="coerce").fillna(0.0).to_numpy()
        raw_weights = np.where(frp_vals >= high_frp_pu_threshold_mw, high_frp_pu_boost, 1.0)
        unknown_sample_weights = raw_weights / raw_weights.sum()

    teacher_params = dict(config.get("model_params", {}))
    teacher_params["max_depth"] = int(pu_cfg.get("teacher_max_depth", 4))
    teacher_params["colsample_bytree"] = float(pu_cfg.get("teacher_colsample_bytree", 0.8))

    teacher_sample_size = 0
    if unknown_count > 0:
        teacher_sample_size = min(unknown_count, max(1, int(unlabeled_multiplier * len(x_pos))))

    pos_count = len(x_pos)
    for _ in range(num_bags):
        sampled_unknown: np.ndarray
        if unknown_count > 0:
            sampled_unknown = rng.choice(
                unknown_count,
                size=teacher_sample_size,
                replace=False,
                p=unknown_sample_weights,
            )
        else:
            sampled_unknown = np.asarray([], dtype=int)

        spy_idx = np.asarray([], dtype=int)
        if pos_count >= 3 and spy_fraction > 0.0:
            desired_spy = max(1, int(np.floor(pos_count * spy_fraction)))
            spy_count = min(desired_spy, pos_count - 1)
            if spy_count > 0:
                spy_idx = rng.choice(pos_count, size=spy_count, replace=False)
        if spy_idx.size > 0:
            keep_mask = np.ones(pos_count, dtype=bool)
            keep_mask[spy_idx] = False
            x_pos_teacher = x_pos.iloc[keep_mask]
            x_spy = x_pos.iloc[spy_idx]
        else:
            x_pos_teacher = x_pos
            x_spy = x_pos.iloc[0:0]

        x_parts = [x_pos_teacher, x_neg]
        y_parts = [
            np.ones(len(x_pos_teacher), dtype=int),
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
        if len(x_spy) > 0:
            spy_scores_by_bag.append(_predict_raw(teacher, x_spy))

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

    unknown_meta_cols = [c for c in list(dict.fromkeys(slice_cols + ["start_time"])) if c in train_df.columns]
    unknown_meta = train_df.loc[unknown_mask, unknown_meta_cols].reset_index(drop=True)

    if unknown_count > 0:
        oob_mean, valid_votes = _oob_mean_scores(score_sums, vote_counts, min_oob_votes=min_oob_votes)
        spy_scores = (
            np.concatenate(spy_scores_by_bag, axis=0)
            if spy_scores_by_bag
            else np.asarray([], dtype=float)
        )
        if int(spy_scores.size) >= spy_min_rows:
            derived_pos = float(np.quantile(spy_scores, spy_pos_quantile))
            derived_neg = float(np.quantile(spy_scores, spy_neg_quantile))
            pos_threshold = max(pos_threshold, derived_pos)
            neg_threshold = min(neg_threshold, derived_neg)

        # Diagnostic logging for PU bagging
        _n_valid = int(valid_votes.sum())
        _finite_oob = oob_mean[valid_votes & np.isfinite(oob_mean)]
        LOGGER.info(
            "PU OOB diagnostics: unknown_count=%d, valid_votes=%d, "
            "vote_counts_min=%d, vote_counts_max=%d, vote_counts_mean=%.1f, "
            "oob_mean_min=%.4f, oob_mean_max=%.4f, oob_mean_median=%.4f, "
            "pos_threshold=%.4f, neg_threshold=%.4f, oob_margin_min=%.4f",
            unknown_count, _n_valid,
            int(vote_counts.min()), int(vote_counts.max()), float(vote_counts.mean()),
            float(_finite_oob.min()) if _finite_oob.size > 0 else float("nan"),
            float(_finite_oob.max()) if _finite_oob.size > 0 else float("nan"),
            float(np.median(_finite_oob)) if _finite_oob.size > 0 else float("nan"),
            pos_threshold, neg_threshold, oob_margin_min,
        )

        pseudo_pos_mask, pseudo_neg_mask, ignored_mask, pos_cut, neg_cut = _build_pseudo_label_masks(
            oob_mean=oob_mean,
            valid_votes=valid_votes,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
            oob_margin_min=oob_margin_min,
        )
        pseudo_pos_count = int(pseudo_pos_mask.sum())
        if pseudo_pos_count < min_pseudo_positive_rows:
            valid_scores = oob_mean[valid_votes & np.isfinite(oob_mean)]
            target_rows = min(int(min_pseudo_positive_rows), int(valid_scores.size))
            LOGGER.info(
                "PU relaxation: pseudo_pos_count=%d, valid_scores_size=%d, "
                "target_rows=%d, pos_cut=%.4f",
                pseudo_pos_count, int(valid_scores.size), target_rows, pos_cut,
            )
            if target_rows > 0:
                rank_cut = int(max(0, valid_scores.size - target_rows))
                relaxed_pos_cut = float(np.partition(valid_scores, rank_cut)[rank_cut])
                pseudo_pos_mask = valid_votes & (oob_mean >= relaxed_pos_cut)
                pos_cut = min(float(pos_cut), relaxed_pos_cut)
                pseudo_pos_count = int(pseudo_pos_mask.sum())
                LOGGER.info(
                    "PU relaxation result: relaxed_pos_cut=%.4f, pseudo_pos_count=%d",
                    relaxed_pos_cut, pseudo_pos_count,
                )
            if pseudo_pos_count < min_pseudo_positive_rows:
                raise ValueError(
                    "xgboost_pu_bagging produced too few pseudo positives: "
                    f"pseudo_positive_rows={pseudo_pos_count}, "
                    f"min_pseudo_positive_rows={min_pseudo_positive_rows}. "
                    "Adjust PU thresholds or improve label/feature coverage."
                )
        pseudo_neg_mask, neg_cap = _apply_pseudo_negative_caps(
            pseudo_neg_mask=pseudo_neg_mask,
            pseudo_positive_rows=pseudo_pos_count,
            pseudo_negative_max_ratio=pseudo_negative_max_ratio,
            max_pseudo_negative_rows=max_pseudo_negative_rows,
            rng=rng,
        )
        ignored_mask = ~(pseudo_pos_mask | pseudo_neg_mask)
        pseudo_diagnostics = _build_pseudo_diagnostics(
            unknown_meta=unknown_meta,
            pseudo_pos_mask=pseudo_pos_mask,
            pseudo_neg_mask=pseudo_neg_mask,
            valid_votes=valid_votes,
            vote_counts=vote_counts,
            slice_cols=slice_cols,
        )
    else:
        pseudo_pos_mask = np.asarray([], dtype=bool)
        pseudo_neg_mask = np.asarray([], dtype=bool)
        ignored_mask = np.asarray([], dtype=bool)
        valid_votes = np.asarray([], dtype=bool)
        vote_counts = np.asarray([], dtype=np.int32)
        pos_cut = pos_threshold
        neg_cut = neg_threshold
        neg_cap = {
            "pseudo_negative_rows_before_cap": 0,
            "pseudo_negative_rows_after_cap": 0,
            "pseudo_negative_rows_cap_target": 0,
            "pseudo_negative_rows_dropped": 0,
        }
        pseudo_diagnostics = {"by_month": [], "by_slice": []}

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

    x_student_raw = pd.concat(x_parts, ignore_index=True)
    y_student_raw = np.concatenate(y_parts)
    x_student, y_student, adasyn_stats = _apply_adasyn_high_intensity(
        config=config,
        x_train=x_student_raw,
        y_train=y_student_raw,
        rng=rng,
    )
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
        "student_rows_before_adasyn": int(len(y_student_raw)),
        "student_rows_after_adasyn": int(len(y_student)),
        "pos_threshold": float(pos_threshold),
        "neg_threshold": float(neg_threshold),
        "pos_threshold_effective": float(pos_cut),
        "neg_threshold_effective": float(neg_cut),
        "spy_fraction": float(spy_fraction),
        "spy_min_rows": int(spy_min_rows),
        "spy_score_rows": int(sum(len(s) for s in spy_scores_by_bag)),
        "spy_pos_quantile": float(spy_pos_quantile),
        "spy_neg_quantile": float(spy_neg_quantile),
        "oob_margin_min": float(oob_margin_min),
        "pseudo_negative_max_ratio": float(pseudo_negative_max_ratio),
        "max_pseudo_negative_rows": int(max_pseudo_negative_rows),
        "min_pseudo_positive_rows": int(min_pseudo_positive_rows),
        **neg_cap,
        "adasyn": adasyn_stats,
        "pseudo_label_diagnostics": pseudo_diagnostics,
    }
    return student, stats


def _build_pseudo_label_masks(
    *,
    oob_mean: np.ndarray,
    valid_votes: np.ndarray,
    pos_threshold: float,
    neg_threshold: float,
    oob_margin_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    margin = max(0.0, float(oob_margin_min))
    pos_cut = min(1.0, float(pos_threshold) + margin)
    neg_cut = max(0.0, float(neg_threshold) - margin)
    pseudo_pos_mask = valid_votes & (oob_mean >= pos_cut)
    pseudo_neg_mask = valid_votes & (oob_mean <= neg_cut)
    ignored_mask = ~(pseudo_pos_mask | pseudo_neg_mask)
    return pseudo_pos_mask, pseudo_neg_mask, ignored_mask, pos_cut, neg_cut


def _apply_pseudo_negative_caps(
    *,
    pseudo_neg_mask: np.ndarray,
    pseudo_positive_rows: int,
    pseudo_negative_max_ratio: float,
    max_pseudo_negative_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    total_neg = int(pseudo_neg_mask.sum())
    if total_neg == 0:
        return pseudo_neg_mask, {
            "pseudo_negative_rows_before_cap": 0,
            "pseudo_negative_rows_after_cap": 0,
            "pseudo_negative_rows_cap_target": 0,
            "pseudo_negative_rows_dropped": 0,
        }

    cap_target = total_neg
    ratio = float(pseudo_negative_max_ratio)
    if ratio >= 0.0:
        cap_target = min(cap_target, int(max(0.0, ratio) * max(0, int(pseudo_positive_rows))))
    if int(max_pseudo_negative_rows) > 0:
        cap_target = min(cap_target, int(max_pseudo_negative_rows))
    cap_target = max(0, int(cap_target))

    if cap_target >= total_neg:
        return pseudo_neg_mask, {
            "pseudo_negative_rows_before_cap": total_neg,
            "pseudo_negative_rows_after_cap": total_neg,
            "pseudo_negative_rows_cap_target": cap_target,
            "pseudo_negative_rows_dropped": 0,
        }

    neg_idx = np.flatnonzero(pseudo_neg_mask)
    keep_idx = rng.choice(neg_idx, size=cap_target, replace=False) if cap_target > 0 else np.asarray([], dtype=int)
    capped = np.zeros_like(pseudo_neg_mask, dtype=bool)
    capped[keep_idx] = True
    return capped, {
        "pseudo_negative_rows_before_cap": total_neg,
        "pseudo_negative_rows_after_cap": int(capped.sum()),
        "pseudo_negative_rows_cap_target": cap_target,
        "pseudo_negative_rows_dropped": int(total_neg - capped.sum()),
    }


def _build_pseudo_diagnostics(
    *,
    unknown_meta: pd.DataFrame,
    pseudo_pos_mask: np.ndarray,
    pseudo_neg_mask: np.ndarray,
    valid_votes: np.ndarray,
    vote_counts: np.ndarray,
    slice_cols: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if unknown_meta.empty:
        return {"by_month": [], "by_slice": []}

    diag = unknown_meta.copy()
    diag["pseudo_state"] = np.where(
        pseudo_pos_mask,
        "pseudo_positive",
        np.where(pseudo_neg_mask, "pseudo_negative", np.where(valid_votes, "ignored_uncertain", "insufficient_oob")),
    )
    diag["oob_votes"] = np.asarray(vote_counts, dtype=int)
    month_rows: list[dict[str, Any]] = []
    if "start_time" in diag.columns:
        month_key = pd.to_datetime(diag["start_time"], errors="coerce", utc=True).dt.strftime("%Y-%m")
        month_df = pd.DataFrame({"month": month_key, "pseudo_state": diag["pseudo_state"]})
        month_counts = (
            month_df.groupby(["month", "pseudo_state"], dropna=False).size().reset_index(name="rows")
        )
        month_rows = [
            {
                "month": str(r["month"]) if pd.notna(r["month"]) else "unknown",
                "pseudo_state": str(r["pseudo_state"]),
                "rows": int(r["rows"]),
            }
            for _, r in month_counts.iterrows()
        ]

    use_slice_cols = [c for c in slice_cols if c in diag.columns]
    slice_rows: list[dict[str, Any]] = []
    if use_slice_cols:
        grouped = diag.groupby(use_slice_cols + ["pseudo_state"], dropna=False).size().reset_index(name="rows")
        for _, row in grouped.iterrows():
            label = "|".join(f"{c}={row[c]}" for c in use_slice_cols)
            slice_rows.append(
                {
                    "slice": label,
                    "pseudo_state": str(row["pseudo_state"]),
                    "rows": int(row["rows"]),
                }
            )

    return {"by_month": month_rows, "by_slice": slice_rows}


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
        global_cal = fit_binary_probability_calibrator(raw_cal, y_cal, method=global_method)

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
        slice_cals[key_label] = fit_binary_probability_calibrator(
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
        out[idx] = float(apply_binary_probability_calibrator(cal, np.asarray([raw_scores[idx]]))[0])
    return out


def _cross_validate_pu_xgboost(
    *,
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    slice_cols: list[str],
    ratio_majority_to_positive: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    cv_cfg = dict(config.get("cross_validation", {}))
    enabled = bool(cv_cfg.get("enabled", True))
    if not enabled:
        return {"enabled": False, "fold_metrics": []}

    known_idx = np.flatnonzero(y_train >= 0)
    if known_idx.size < 4:
        return {"enabled": True, "fold_metrics": [], "skipped_reason": "insufficient_known_rows"}
    y_known = y_train[known_idx]
    if np.unique(y_known).size < 2:
        return {"enabled": True, "fold_metrics": [], "skipped_reason": "single_known_class"}

    folds = max(2, int(cv_cfg.get("folds", 3)))
    max_known_rows = max(0, int(cv_cfg.get("max_known_rows", 150000)))
    if max_known_rows > 0 and known_idx.size > max_known_rows:
        keep_mask = np.zeros_like(known_idx, dtype=bool)
        for cls in (0, 1):
            cls_local_idx = np.flatnonzero(y_known == cls)
            if cls_local_idx.size == 0:
                continue
            cls_keep = max(1, int(round(max_known_rows * (cls_local_idx.size / known_idx.size))))
            chosen_local = rng.choice(cls_local_idx, size=min(cls_keep, cls_local_idx.size), replace=False)
            keep_mask[chosen_local] = True
        known_idx = known_idx[keep_mask]
        y_known = y_train[known_idx]

    class_counts = np.bincount(y_known.astype(int), minlength=2)
    max_folds = int(class_counts.min())
    if max_folds < 2:
        return {"enabled": True, "fold_metrics": [], "skipped_reason": "insufficient_class_rows"}
    folds = min(folds, max_folds)

    unknown_idx = np.flatnonzero(y_train == -1)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(config.get("seed", 42)),
    )
    threshold = float(config.get("decision_threshold", 0.5))
    fold_metrics: list[dict[str, Any]] = []

    for fold, (train_local, val_local) in enumerate(splitter.split(np.zeros(len(known_idx)), y_known), start=1):
        fold_known_train_idx = known_idx[train_local]
        fold_known_val_idx = known_idx[val_local]
        fold_train_idx = np.concatenate([fold_known_train_idx, unknown_idx])
        fold_train_df = train_df.iloc[fold_train_idx].copy().reset_index(drop=True)
        fold_y_train = y_train[fold_train_idx]

        fold_train_df, fold_y_train, _ = _stratified_majority_sample(
            train_df=fold_train_df,
            y_train=fold_y_train,
            ratio_majority_to_positive=ratio_majority_to_positive,
            slice_cols=slice_cols,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )

        fold_model, _ = _fit_teacher_student_pu(
            config=config,
            train_df=fold_train_df,
            y_train=fold_y_train,
            features=features,
            slice_cols=slice_cols,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )

        fold_val_df = train_df.iloc[fold_known_val_idx].copy().reset_index(drop=True)
        fold_cal_df, fold_eval_df = _split_calibration_eval_holdout(
            fold_val_df,
            calibration_fraction=float(config.get("calibration_holdout_fraction", 0.5)),
        )
        if fold_eval_df.empty:
            fold_eval_df = fold_cal_df.copy()
        if fold_cal_df.empty or fold_eval_df.empty:
            continue

        global_cal, slice_cals, _, _ = _calibrate(
            config=config,
            model=fold_model,
            calibration_df=fold_cal_df,
            features=features,
            slice_cols=slice_cols,
            label_col=str(config.get("label_column", "event_label")),
        )
        y_fold_eval = _map_labels(fold_eval_df, str(config.get("label_column", "event_label")))
        raw_fold_eval = _predict_raw(fold_model, _feature_matrix(fold_eval_df, features))
        p_fold_eval = _apply_slice_calibration(
            eval_df=fold_eval_df,
            raw_scores=raw_fold_eval,
            slice_cols=slice_cols,
            global_calibrator=global_cal,
            slice_calibrators=slice_cals,
        )
        metric = _metrics(y_fold_eval, p_fold_eval, threshold=threshold)
        metric["fold"] = int(fold)
        metric["calibration_type"] = "slice_then_global"
        fold_metrics.append(metric)

    if not fold_metrics:
        return {"enabled": True, "fold_metrics": [], "skipped_reason": "no_valid_folds"}

    keys = ["precision", "recall", "f1", "roc_auc"]
    mean_metrics = {
        key: float(np.nanmean([m[key] for m in fold_metrics if m.get(key) is not None]))
        for key in keys
    }
    return {
        "enabled": True,
        "fold_count": int(len(fold_metrics)),
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
    }


def train_denoiser_v2(config: Dict[str, Any]) -> str:
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    rng_train = np.random.default_rng(seed + 1)
    rng_cv = np.random.default_rng(seed + 2)

    model_backend = str(config.get("model_backend", "xgboost_pu_bagging")).strip().lower()
    coverage_scope = str(config.get("coverage_scope", "covered")).strip().lower()
    coverage_mask_source = str(config.get("coverage_mask_source", "db_mask")).strip()
    coverage_authority_profile = str(config.get("coverage_authority_profile", "wfigs_us")).strip()
    coverage_max_age_hours = float(config.get("coverage_max_age_hours", 72.0))
    coverage_fail_on_stale = bool(config.get("coverage_fail_on_stale", True))
    coverage_freshness: dict[str, Any] | None = None

    if coverage_scope == "covered" and coverage_mask_source == "db_mask":
        if coverage_fail_on_stale:
            coverage_freshness = require_coverage_freshness(
                authority_profile=coverage_authority_profile,
                max_age_hours=coverage_max_age_hours,
            )
        else:
            coverage_freshness = get_coverage_freshness(
                authority_profile=coverage_authority_profile,
                max_age_hours=coverage_max_age_hours,
            )

    features = list(config["features"])
    label_col = str(config.get("label_column", "event_label"))
    slice_cols = list(config.get("slice_columns", ["sensor", "biome_slice"]))

    required_columns = list(dict.fromkeys(features + [label_col] + slice_cols + ["start_time"]))
    if coverage_scope == "covered":
        required_columns.append("truth_covered_mask")
    train_df, calibration_df, eval_df = _load_snapshot(
        config["snapshot_path"], columns=required_columns
    )
    train_df = _ensure_slice_columns(train_df, slice_cols)
    calibration_df = _ensure_slice_columns(calibration_df, slice_cols)
    eval_df = _ensure_slice_columns(eval_df, slice_cols)
    train_df, calibration_df, eval_df, micro_batch_stats = _apply_micro_batch(
        train_df=train_df,
        calibration_df=calibration_df,
        eval_df=eval_df,
        config=config,
        label_col=label_col,
        default_strat_cols=slice_cols,
        rng=np.random.default_rng(seed + 101),
    )

    for col in features:
        if col not in train_df.columns:
            train_df[col] = np.nan
        if col not in calibration_df.columns:
            calibration_df[col] = np.nan
        if col not in eval_df.columns:
            eval_df[col] = np.nan
    train_df = _ensure_slice_columns(train_df, slice_cols)
    calibration_df = _ensure_slice_columns(calibration_df, slice_cols)
    eval_df = _ensure_slice_columns(eval_df, slice_cols)

    train_scope = _select_scope(train_df, coverage_scope)
    calibration_scope = _select_scope(calibration_df, coverage_scope)
    eval_scope = _select_scope(eval_df, coverage_scope)

    y_train = _map_labels(train_scope, label_col)
    known_train = y_train >= 0
    if not known_train.any():
        raise ValueError("No known labels (POSITIVE/NEGATIVE) available in train scope.")

    if model_backend not in {"xgboost", "xgboost_pu_bagging"}:
        raise ValueError(
            f"Unsupported model_backend={model_backend!r}. "
            "Use XGBoost PU backend (xgboost_pu_bagging)."
        )

    sampling_cfg = dict(config.get("sampling", {}))
    ratio_majority_to_positive = float(sampling_cfg.get("majority_ratio", 10.0))
    apply_sampling = bool(sampling_cfg.get("enabled", True))
    if apply_sampling:
        fit_train_df, fit_y_train, sampling_stats = _stratified_majority_sample(
            train_df=train_scope,
            y_train=y_train,
            ratio_majority_to_positive=ratio_majority_to_positive,
            slice_cols=slice_cols,
            rng=rng_train,
        )
    else:
        fit_train_df = train_scope.copy()
        fit_y_train = y_train.copy()
        sampling_stats = {"enabled": False, "reason": "disabled"}

    cv_stats = _cross_validate_pu_xgboost(
        config=config,
        train_df=train_scope,
        y_train=y_train,
        features=features,
        slice_cols=slice_cols,
        ratio_majority_to_positive=ratio_majority_to_positive,
        rng=rng_cv,
    )

    model, pu_stats = _fit_teacher_student_pu(
        config,
        train_df=fit_train_df,
        y_train=fit_y_train,
        features=features,
        slice_cols=slice_cols,
        rng=rng,
    )

    cal_known_mask = _map_labels(calibration_scope, label_col) >= 0
    calibration_known_df = calibration_scope.loc[cal_known_mask].copy()
    if calibration_known_df.empty:
        raise ValueError("No known labels in calibration scope; cannot calibrate.")

    eval_known_mask = _map_labels(eval_scope, label_col) >= 0
    eval_known_holdout = eval_scope.loc[eval_known_mask].copy()
    if eval_known_holdout.empty:
        raise ValueError("No known labels in eval scope; cannot compute promotion gates.")

    # Temporal non-overlap assertion: every calibration timestamp must precede every eval timestamp.
    if "start_time" in calibration_known_df.columns and "start_time" in eval_known_holdout.columns:
        cal_max = calibration_known_df["start_time"].max()
        eval_min = eval_known_holdout["start_time"].min()
        if cal_max >= eval_min:
            raise ValueError(
                f"Temporal leakage detected: calibration max timestamp ({cal_max}) "
                f">= eval min timestamp ({eval_min}). "
                "Ensure the snapshot covers a wide enough time range for a 60/20/20 split."
            )

    global_calibrator, slice_calibrators, y_cal, raw_cal = _calibrate(
        config=config,
        model=model,
        calibration_df=calibration_known_df,
        features=features,
        slice_cols=slice_cols,
        label_col=label_col,
    )

    # Split eval_known_holdout into eval_holdout (80%) and calibrator_validation (20%).
    # Preserve temporal ordering if start_time exists.
    eval_holdout, calibrator_validation = _temporal_2way_split(
        eval_known_holdout, eval_fraction=0.8
    )

    y_eval_known = _map_labels(eval_holdout, label_col)
    if len(eval_holdout) == 0:
        raise ValueError("No known labels in eval holdout.")

    raw_eval = _predict_raw(model, _feature_matrix(eval_holdout, features))
    calibrated_scores = _apply_slice_calibration(
        eval_df=eval_holdout,
        raw_scores=raw_eval,
        slice_cols=slice_cols,
        global_calibrator=global_calibrator,
        slice_calibrators=slice_calibrators,
    )

    # Compute calibrator validation metrics if calibrator_validation is non-empty.
    calibrator_validation_metrics = None
    calibrator_overfitting_warning = False
    if len(calibrator_validation) > 0:
        y_cal_val = _map_labels(calibrator_validation, label_col)
        if (y_cal_val >= 0).any():  # Only compute if there are known labels
            raw_cal_val = _predict_raw(model, _feature_matrix(calibrator_validation, features))
            calibrated_cal_val = _apply_slice_calibration(
                eval_df=calibrator_validation,
                raw_scores=raw_cal_val,
                slice_cols=slice_cols,
                global_calibrator=global_calibrator,
                slice_calibrators=slice_calibrators,
            )
            brier_eval = float(brier_score_loss(y_eval_known, np.clip(calibrated_scores, 0.0, 1.0)))
            brier_val = float(brier_score_loss(y_cal_val, np.clip(calibrated_cal_val, 0.0, 1.0)))
            brier_degradation_pct = ((brier_val - brier_eval) / (brier_eval + 1e-10)) * 100.0
            calibrator_validation_metrics = {
                "n_samples": int(len(calibrator_validation)),
                "brier_calibrated": brier_val,
                "brier_degradation_pct": float(brier_degradation_pct),
            }
            # Warn if Brier loss increases >5% on validation set (possible overfitting).
            if brier_degradation_pct > 5.0:
                calibrator_overfitting_warning = True
                LOGGER.warning(
                    f"Calibrator overfitting detected: validation Brier loss {brier_val:.4f} "
                    f"is {brier_degradation_pct:.1f}% worse than eval Brier loss {brier_eval:.4f}. "
                    f"Consider using Platt scaling if isotonic regression was selected."
                )

    fallback_threshold = float(config.get("decision_threshold", 0.5))
    target_event_recall = float(config.get("target_event_recall", 0.92))
    threshold_optimization_enabled = bool(config.get("optimize_threshold_for_recall", True))
    threshold_optimization = {
        "enabled": bool(threshold_optimization_enabled),
        "target_event_recall": float(target_event_recall),
        "fallback_threshold": float(fallback_threshold),
        "optimal_threshold": float(fallback_threshold),
        "achieved_recall": None,
        "achieved_precision": None,
        "target_met": False,
        "selection_strategy": "fallback",
        "candidate_threshold_count": 0,
    }
    if threshold_optimization_enabled:
        threshold_optimization = optimize_threshold_for_target_recall(
            calibrated_scores,
            y_eval_known,
            target_recall=target_event_recall,
            fallback_threshold=fallback_threshold,
        )
    threshold = float(threshold_optimization.get("optimal_threshold", fallback_threshold))
    metrics = _metrics(y_eval_known, calibrated_scores, threshold=threshold)
    raw_metrics = _metrics(y_eval_known, raw_eval, threshold=threshold)
    calibration_active = (global_calibrator.get("type") != "identity") or bool(slice_calibrators)
    calibration_diagnostics = {
        "calibration_active": bool(calibration_active),
        "global_calibrator_type": str(global_calibrator.get("type", "identity")),
        "slice_calibrator_count": int(len(slice_calibrators)),
        "mean_abs_shift": float(np.mean(np.abs(calibrated_scores - raw_eval))),
        "brier_raw": float(brier_score_loss(y_eval_known, np.clip(raw_eval, 0.0, 1.0))),
        "brier_calibrated": float(brier_score_loss(y_eval_known, np.clip(calibrated_scores, 0.0, 1.0))),
        "raw_metrics": raw_metrics,
        "calibrator_validation_metrics": calibrator_validation_metrics,
        "calibrator_overfitting_warning": bool(calibrator_overfitting_warning),
    }
    shap_cfg = dict(config.get("shap", {}))
    shap_enabled = bool(shap_cfg.get("enabled", False))
    shap_top_features: list[dict[str, float | str]] = []
    if shap_enabled:
        shap_top_features = _compute_shap_top_features(
            model=model,
            x=_feature_matrix(eval_holdout, features),
            top_k=int(shap_cfg.get("top_k", 5)),
            sample_rows=int(shap_cfg.get("sample_rows", 5000)),
            rng=np.random.default_rng(seed + 303),
        )

    # Operational latency estimate: extrapolate per 10k events from eval prediction speed.
    latency_eval_df = eval_scope if not eval_scope.empty else eval_holdout
    start = time.perf_counter()
    _ = _predict_raw(model, _feature_matrix(latency_eval_df, features))
    elapsed = max(1e-6, time.perf_counter() - start)
    latency_per_10k = float(elapsed * (10000.0 / max(1, len(latency_eval_df))))

    sensor_bias_pct = None
    if "sensor" in eval_holdout.columns and len(eval_holdout["sensor"].dropna().unique()) >= 2:
        sensor_means = eval_holdout.assign(score=calibrated_scores).groupby("sensor")["score"].mean()
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
            "decision_fallback": float(fallback_threshold),
            "target_event_recall": float(target_event_recall),
            "strong_filter": float(config.get("strong_filter_threshold", 0.5)),
            "downweight": float(config.get("downweight_threshold", 0.7)),
            "uncertainty_band_low": float(config.get("uncertainty_band_low", 0.45)),
            "uncertainty_band_high": float(config.get("uncertainty_band_high", 0.55)),
        },
        "threshold_optimization": threshold_optimization,
        "latency_per_10k_seconds": latency_per_10k,
        "run_id": run_name,
        "model_backend": model_backend,
        "calibration_diagnostics": calibration_diagnostics,
        "shap_top_features": shap_top_features,
        "gate_scope": coverage_scope,
        "coverage_mask_source": coverage_mask_source,
        "coverage_authority_profile": coverage_authority_profile,
        "coverage_run_id": (coverage_freshness or {}).get("run_id"),
        "coverage_data_freshness": coverage_freshness,
    }

    joblib.dump(bundle, os.path.join(run_dir, "model_bundle.pkl"))
    joblib.dump(model, os.path.join(run_dir, "model.pkl"))

    with open(os.path.join(run_dir, "feature_list.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    # Export runtime contract for inference-time feature validation (Issue #281).
    contract = DenoiserRuntimeContract(features=tuple(features))
    write_contract(Path(run_dir) / "runtime_contract.json", contract)

    training_summary = {
        "run_id": run_name,
        "model_backend": model_backend,
        "coverage_scope": coverage_scope,
        "coverage_mask_source": coverage_mask_source,
        "coverage_authority_profile": coverage_authority_profile,
        "coverage_data_freshness": coverage_freshness,
        "train_rows": int(len(train_df)),
        "train_scope_rows": int(len(train_scope)),
        "train_fit_rows": int(len(fit_train_df)),
        "calibration_rows": int(len(calibration_df)),
        "calibration_scope_rows": int(len(calibration_scope)),
        "eval_rows": int(len(eval_df)),
        "eval_scope_rows": int(len(eval_scope)),
        "train_known_rows": int(known_train.sum()),
        "train_unknown_rows": int((y_train == -1).sum()),
        "calibration_known_rows": int(len(y_cal)),
        "eval_known_rows": int(len(y_eval_known)),
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "threshold_optimization": threshold_optimization,
        "calibration_diagnostics": calibration_diagnostics,
        "cross_validation": cv_stats,
        "latency_per_10k_seconds": latency_per_10k,
        "sensor_bias_pct": sensor_bias_pct,
        "sampling_stats": sampling_stats,
        "pu_stats": pu_stats,
        "micro_batch_stats": micro_batch_stats,
        "shap_top_features": shap_top_features,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    gate_report = {
        "run_id": run_name,
        "pass": gate_pass,
        "gate_scope": coverage_scope,
        "coverage_mask_source": coverage_mask_source,
        "coverage_authority_profile": coverage_authority_profile,
        "coverage_data_freshness": coverage_freshness,
        "coverage_run_id": (coverage_freshness or {}).get("run_id"),
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

    cv_mean = cv_stats.get("mean_metrics") if isinstance(cv_stats, dict) else None
    if cv_mean:
        LOGGER.info(
            "Cross-validation metrics: precision=%.4f recall=%.4f f1=%.4f auc=%.4f",
            float(cv_mean.get("precision", np.nan)),
            float(cv_mean.get("recall", np.nan)),
            float(cv_mean.get("f1", np.nan)),
            float(cv_mean.get("roc_auc", np.nan)),
        )
    LOGGER.info(
        "Eval metrics (calibrated): precision=%.4f recall=%.4f f1=%.4f auc=%.4f",
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["f1"]),
        float(metrics["roc_auc"]) if metrics["roc_auc"] is not None else float("nan"),
    )
    LOGGER.info(
        "Calibration: active=%s global=%s slice_models=%d mean_abs_shift=%.6f brier_raw=%.6f brier_calibrated=%.6f",
        str(calibration_diagnostics["calibration_active"]).lower(),
        calibration_diagnostics["global_calibrator_type"],
        int(calibration_diagnostics["slice_calibrator_count"]),
        float(calibration_diagnostics["mean_abs_shift"]),
        float(calibration_diagnostics["brier_raw"]),
        float(calibration_diagnostics["brier_calibrated"]),
    )

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train denoiser v2 (event-level PU + calibration).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--micro-batch-start", type=str, default=None, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--micro-batch-end", type=str, default=None, help="YYYY-MM-DD (UTC, exclusive)")
    parser.add_argument("--micro-batch-max-rows", type=int, default=30000)
    parser.add_argument("--micro-batch-min-positive", type=int, default=100)
    parser.add_argument("--micro-batch-shap-top-k", type=int, default=5)
    parser.add_argument("--micro-batch-shap-sample-rows", type=int, default=5000)
    args = parser.parse_args()

    config = load_config(args.config)
    if bool(args.micro_batch_start) ^ bool(args.micro_batch_end):
        raise SystemExit("Both --micro-batch-start and --micro-batch-end are required together.")
    if args.micro_batch_start and args.micro_batch_end:
        label_col = str(config.get("label_column", "event_label"))
        slice_cols = list(config.get("slice_columns", ["sensor", "biome_slice"]))
        config["micro_batch"] = {
            "enabled": True,
            "start": args.micro_batch_start,
            "end": args.micro_batch_end,
            "max_rows": int(args.micro_batch_max_rows),
            "min_positive_rows": int(args.micro_batch_min_positive),
            "stratify_columns": list(dict.fromkeys([label_col] + slice_cols)),
        }
        config["shap"] = {
            "enabled": True,
            "top_k": int(args.micro_batch_shap_top_k),
            "sample_rows": int(args.micro_batch_shap_sample_rows),
        }
    run_dir = train_denoiser_v2(config)
    metrics_path = os.path.join(run_dir, "metrics.json")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        summary = {}
    cv_mean = ((summary.get("cross_validation") or {}).get("mean_metrics") or {})
    holdout = summary.get("metrics") or {}
    cal = summary.get("calibration_diagnostics") or {}
    if cv_mean:
        print(
            "CV_METRICS "
            f"precision={float(cv_mean.get('precision', float('nan'))):.6f} "
            f"recall={float(cv_mean.get('recall', float('nan'))):.6f} "
            f"f1={float(cv_mean.get('f1', float('nan'))):.6f} "
            f"auc_roc={float(cv_mean.get('roc_auc', float('nan'))):.6f}"
        )
    if holdout:
        auc = holdout.get("roc_auc")
        auc_val = float(auc) if auc is not None else float("nan")
        print(
            "HOLDOUT_METRICS "
            f"precision={float(holdout.get('precision', float('nan'))):.6f} "
            f"recall={float(holdout.get('recall', float('nan'))):.6f} "
            f"f1={float(holdout.get('f1', float('nan'))):.6f} "
            f"auc_roc={auc_val:.6f}"
        )
    threshold_opt = summary.get("threshold_optimization") or {}
    if threshold_opt:
        print(
            "OPTIMAL_THRESHOLD "
            f"value={float(threshold_opt.get('optimal_threshold', float('nan'))):.6f} "
            f"target_event_recall={float(threshold_opt.get('target_recall', threshold_opt.get('target_event_recall', float('nan')))):.6f} "
            f"achieved_recall={float(threshold_opt.get('achieved_recall', float('nan'))):.6f} "
            f"achieved_precision={float(threshold_opt.get('achieved_precision', float('nan'))):.6f} "
            f"target_met={str(bool(threshold_opt.get('target_met', False))).lower()}"
        )
    if cal:
        print(
            "CALIBRATION "
            f"active={str(bool(cal.get('calibration_active', False))).lower()} "
            f"global={cal.get('global_calibrator_type', 'identity')} "
            f"slice_models={int(cal.get('slice_calibrator_count', 0))} "
            f"mean_abs_shift={float(cal.get('mean_abs_shift', 0.0)):.6f} "
            f"brier_raw={float(cal.get('brier_raw', float('nan'))):.6f} "
            f"brier_calibrated={float(cal.get('brier_calibrated', float('nan'))):.6f}"
        )
    shap_top = summary.get("shap_top_features") or []
    for item in shap_top:
        print(
            "SHAP_IMPORTANCE "
            f"rank={int(float(item.get('rank', 0)))} "
            f"feature={item.get('feature', 'unknown')} "
            f"mean_abs_shap={float(item.get('mean_abs_shap', float('nan'))):.6f}"
        )
    print(run_dir)


if __name__ == "__main__":
    main()
