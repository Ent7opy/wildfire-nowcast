import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import hashlib
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("train_denoiser")

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _maybe_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        sha = out.strip()
        return sha if sha else None
    except Exception:
        return None

def load_data(snapshot_path: str, feature_cols: List[str]) -> pd.DataFrame:
    """Load parquet and ensure features exist."""
    LOGGER.info(f"Loading data from {snapshot_path}")
    df = pd.read_parquet(snapshot_path)
    
    # Validation: no NaN explosions; drop/Impute if needed
    for col in feature_cols:
        if col not in df.columns:
            LOGGER.warning(f"Feature column {col} missing from snapshot. Filling with NaN.")
            df[col] = np.nan
            
    nan_counts = df[feature_cols].isna().sum()
    if nan_counts.any():
        LOGGER.info(f"NaN counts in features:\n{nan_counts[nan_counts > 0]}")
    
    return df

def split_data(
    df: pd.DataFrame, 
    split_time: str = None, 
    split_percentile: float = 0.8,
    gap_hours: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based holdout split."""
    df = df.copy()
    df["acq_time"] = pd.to_datetime(df["acq_time"])
    
    if split_time:
        split_dt = pd.to_datetime(split_time)
    else:
        # Default to percentile-based split
        split_dt = df["acq_time"].quantile(split_percentile)
        
    LOGGER.info(f"Splitting data at {split_dt} (gap: {gap_hours}h)")
    
    train_df = df[df["acq_time"] < split_dt].copy()
    
    eval_start = split_dt + pd.Timedelta(hours=gap_hours)
    eval_df = df[df["acq_time"] >= eval_start].copy()
    
    return train_df, eval_df

def _safe_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    """Return ROC-AUC/PR-AUC, or nulls when undefined (e.g., single-class eval)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    unique = np.unique(y_true)
    if unique.size < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "warnings": ["auc_undefined_single_class_eval"],
        }

    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)
    roc_f = float(roc) if np.isfinite(roc) else None
    pr_f = float(pr) if np.isfinite(pr) else None
    warnings: List[str] = []
    if roc_f is None:
        warnings.append("roc_auc_non_finite")
    if pr_f is None:
        warnings.append("pr_auc_non_finite")
    return {"roc_auc": roc_f, "pr_auc": pr_f, "warnings": warnings}

def compute_metrics(y_true, y_score, thresholds: List[float]):
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    # Drop non-finite scores (and corresponding labels) so metrics don't crash on NaNs/Infs.
    finite_mask = np.isfinite(y_score_arr)
    dropped = int((~finite_mask).sum())
    if dropped:
        y_true_arr = y_true_arr[finite_mask]
        y_score_arr = y_score_arr[finite_mask]

    auc = _safe_auc_metrics(y_true_arr, y_score_arr)
    metrics: Dict[str, Any] = {
        "roc_auc": auc["roc_auc"],
        "pr_auc": auc["pr_auc"],
        "thresholds": {},
    }
    if dropped:
        metrics["dropped_non_finite_scores"] = dropped
    if auc.get("warnings"):
        metrics["warnings"] = auc["warnings"]
    
    for t in thresholds:
        y_pred = (y_score_arr >= t).astype(int)
        # Handle cases where confusion_matrix might not return all 4 values due to single class
        cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        metrics["thresholds"][str(t)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
        
    return metrics

def _ensure_label_numeric(df: pd.DataFrame) -> pd.Series:
    if "label_numeric" not in df.columns:
        raise ValueError("Expected column 'label_numeric' in snapshot data.")
    y = df["label_numeric"]
    # Coerce bool/float/object into int 0/1
    y = pd.to_numeric(y, errors="raise").astype(int)
    bad = set(y.unique()) - {0, 1}
    if bad:
        raise ValueError(f"label_numeric must be 0/1; found values {sorted(bad)}")
    return y

def _split_stratified_random(df: pd.DataFrame, eval_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic split that (when possible) forces at least one sample from each class into eval,
    so AUC metrics are defined.

    If there are <2 samples of a class, we cannot guarantee both classes appear in both splits.
    """
    y = _ensure_label_numeric(df).to_numpy()
    idx = df.index.to_numpy()
    n = idx.size
    n_eval = int(np.ceil(eval_size * n))
    n_eval = max(1, min(n - 1, n_eval))  # keep both sides non-empty

    rng = np.random.RandomState(seed)

    pos_idx = idx[y == 1]
    neg_idx = idx[y == 0]

    # If we can, force at least one from each class into eval, and keep at least one from each in train.
    force_both = (pos_idx.size >= 2) and (neg_idx.size >= 2) and (n_eval >= 2)
    if force_both:
        eval_seed_idx = np.concatenate([
            rng.choice(pos_idx, size=1, replace=False),
            rng.choice(neg_idx, size=1, replace=False),
        ])
        remaining = np.setdiff1d(idx, eval_seed_idx, assume_unique=False)
        remaining_eval_slots = n_eval - eval_seed_idx.size
        if remaining_eval_slots > 0:
            eval_rest = rng.choice(remaining, size=remaining_eval_slots, replace=False)
            eval_idx = np.concatenate([eval_seed_idx, eval_rest])
        else:
            eval_idx = eval_seed_idx
        train_idx = np.setdiff1d(idx, eval_idx, assume_unique=False)
        return df.loc[train_idx].copy(), df.loc[eval_idx].copy()

    # Fallback: best-effort stratified split. This may still yield single-class eval if a class is extremely rare.
    train_idx, eval_idx = train_test_split(
        idx,
        test_size=n_eval,
        random_state=seed,
        shuffle=True,
        stratify=y if np.unique(y).size >= 2 else None,
    )
    return df.loc[train_idx].copy(), df.loc[eval_idx].copy()

def _split_with_fallback(config: Dict[str, Any], df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split according to config. If the chosen split yields an empty set or single-class eval,
    fall back to a stratified random split to enable meaningful ROC/PR AUC.
    """
    split_meta: Dict[str, Any] = {}
    strategy = config.get("split_strategy", "time")
    seed = int(config.get("seed", 42))

    if strategy == "stratified_random":
        eval_size = float(config.get("eval_size", 0.2))
        train_df, eval_df = _split_stratified_random(df, eval_size=eval_size, seed=seed)
        split_meta["strategy"] = "stratified_random"
        split_meta["eval_size"] = eval_size
        return train_df, eval_df, split_meta

    # Default: time split
    train_df, eval_df = split_data(
        df,
        split_time=config.get("split_time"),
        split_percentile=config.get("split_percentile", 0.8),
        gap_hours=config.get("gap_hours", 0),
    )
    split_meta["strategy"] = "time"
    split_meta["split_time"] = config.get("split_time")
    split_meta["split_percentile"] = config.get("split_percentile", 0.8)
    split_meta["gap_hours"] = config.get("gap_hours", 0)

    if train_df.empty or eval_df.empty:
        split_meta["fallback_reason"] = "empty_train_or_eval"
    else:
        y_eval = _ensure_label_numeric(eval_df)
        if y_eval.nunique() < 2:
            split_meta["fallback_reason"] = "single_class_eval"

    if "fallback_reason" in split_meta:
        fb = config.get("fallback_split_strategy", "stratified_random")
        if fb != "stratified_random":
            LOGGER.warning("Split produced %s; fallback strategy %s not supported. Using stratified_random.", split_meta["fallback_reason"], fb)
        eval_size = float(config.get("eval_size", 0.2))
        LOGGER.warning("Falling back to stratified_random split (eval_size=%s) due to %s", eval_size, split_meta["fallback_reason"])
        train_df, eval_df = _split_stratified_random(df, eval_size=eval_size, seed=seed)
        split_meta["fallback_strategy"] = "stratified_random"
        split_meta["eval_size"] = eval_size

    return train_df, eval_df, split_meta

def _compute_baseline_metrics(df_eval: pd.DataFrame, y_true: np.ndarray, thresholds: List[float], baselines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Baselines are simple single-feature scorers. Example:
      baselines:
        - name: confidence_norm
          column: confidence_norm
        - name: frp
          column: frp
    """
    out: Dict[str, Any] = {}
    for b in baselines or []:
        name = b.get("name") or b.get("column")
        col = b.get("column")
        if not name or not col:
            continue
        if col not in df_eval.columns:
            out[name] = {"error": f"missing_column:{col}"}
            continue
        score = pd.to_numeric(df_eval[col], errors="coerce").to_numpy()
        # If higher score should mean more likely positive, keep as-is; caller can provide invert: true
        if bool(b.get("invert", False)):
            score = -score
        # Normalize to [0,1] for threshold metrics unless explicitly disabled.
        if bool(b.get("normalize", True)):
            finite = np.isfinite(score)
            if finite.any():
                s_min = float(np.min(score[finite]))
                s_max = float(np.max(score[finite]))
                if s_max > s_min:
                    score = (score - s_min) / (s_max - s_min)
                else:
                    score = np.zeros_like(score, dtype=float)
        out[name] = compute_metrics(y_true, score, thresholds)
        out[name]["column"] = col
        out[name]["invert"] = bool(b.get("invert", False))
        out[name]["normalize"] = bool(b.get("normalize", True))
    return out

def _balanced_sample_weight(y: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute per-sample weights to approximately balance classes.

    We use sample_weight (supported by HistGradientBoostingClassifier.fit) instead of
    passing class_weight as an init param (not consistently supported across sklearn versions).
    """
    y = np.asarray(y).astype(int)
    n = int(y.size)
    if n == 0:
        return None
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    return np.where(y == 1, w_pos, w_neg).astype(float)

def train_baseline(config: Dict[str, Any]):
    # 1. Setup run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = _maybe_git_sha() or "unknown"
    run_name = f"{timestamp}_{git_sha}"
    model_dir = os.path.join(config.get("model_output_root", "models/denoiser_v1"), run_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 2. Load data
    feature_cols = config["features"]
    snapshot_path = config["snapshot_path"]
    
    # Handle if snapshot_path is a directory (from export_snapshot)
    if os.path.isdir(snapshot_path):
        train_path = os.path.join(snapshot_path, "train.parquet")
        eval_path = os.path.join(snapshot_path, "eval.parquet")
        LOGGER.info(f"Snapshot path is a directory. Loading {train_path} and {eval_path}")
        train_df = pd.read_parquet(train_path)
        eval_df = pd.read_parquet(eval_path)
        
        # If one of them is empty, we might have a bad temporal split in the snapshot
        # Fallback: combine and re-split if allowed
        if train_df.empty or eval_df.empty:
            LOGGER.warning("One of the pre-split files is empty. Combining and re-splitting...")
            df = pd.concat([train_df, eval_df], axis=0)
            train_df, eval_df, split_meta = _split_with_fallback(config, df)
        else:
            # Even if non-empty, ensure eval has both classes for AUC metrics; else re-split combined.
            try:
                y_eval_check = _ensure_label_numeric(eval_df)
                if y_eval_check.nunique() < 2:
                    LOGGER.warning("Pre-split eval.parquet is single-class; combining and re-splitting for meaningful AUC metrics...")
                    df = pd.concat([train_df, eval_df], axis=0)
                    train_df, eval_df, split_meta = _split_with_fallback(config, df)
                else:
                    split_meta = {"strategy": "snapshot_presplit"}
            except Exception as e:
                LOGGER.warning("Could not validate pre-split labels (%s); combining and re-splitting...", e)
                df = pd.concat([train_df, eval_df], axis=0)
                train_df, eval_df, split_meta = _split_with_fallback(config, df)
    else:
        df = load_data(snapshot_path, feature_cols)
        train_df, eval_df, split_meta = _split_with_fallback(config, df)
    
    if train_df.empty:
        raise ValueError("Training set is empty. Check your data or split_time/split_percentile.")
    if eval_df.empty:
        LOGGER.warning("Evaluation set is empty. Metrics will not be computed.")
    
    y_train_full = _ensure_label_numeric(train_df)
    y_eval_full = _ensure_label_numeric(eval_df) if not eval_df.empty else pd.Series(dtype=int)
    LOGGER.info(f"Train samples: {len(train_df)} (Pos: {(y_train_full == 1).sum()}, Neg: {(y_train_full == 0).sum()})")
    LOGGER.info(f"Eval samples: {len(eval_df)} (Pos: {(y_eval_full == 1).sum()}, Neg: {(y_eval_full == 0).sum()})")
    
    X_train = train_df[feature_cols]
    y_train = y_train_full
    X_eval = eval_df[feature_cols]
    y_eval = y_eval_full
    
    # 3. Train model
    LOGGER.info("Training HistGradientBoostingClassifier...")
    model_params = config.get("model_params", {})
    # Default seeds and class_weight for imbalance
    if "random_state" not in model_params:
        model_params["random_state"] = config.get("seed", 42)
        
    clf = HistGradientBoostingClassifier(**model_params)
    sample_weight = None
    if config.get("handle_imbalance", True):
        sample_weight = _balanced_sample_weight(y_train.to_numpy())
        if sample_weight is not None:
            LOGGER.info("Using balanced sample_weight for training (pos/neg reweighting).")
        else:
            LOGGER.info("Not using sample_weight (missing a class or empty training set).")
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    
    # 4. Predict and evaluate
    if eval_df.empty:
        LOGGER.warning("Evaluation set is empty. Skipping metrics computation.")
        metrics = {"error": "Empty evaluation set"}
        y_prob = np.array([])
    else:
        y_prob = clf.predict_proba(X_eval)[:, 1]
        thresholds = config.get("eval_thresholds", [0.2, 0.5, 0.8])
        metrics = compute_metrics(y_eval.to_numpy(), y_prob, thresholds)
        metrics["baselines"] = _compute_baseline_metrics(
            df_eval=eval_df,
            y_true=y_eval.to_numpy(),
            thresholds=thresholds,
            baselines=config.get("baseline_models", []),
        )
        
        if metrics.get("roc_auc") is None:
            LOGGER.warning("ROC-AUC undefined (likely single-class eval).")
        else:
            LOGGER.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        if metrics.get("pr_auc") is None:
            LOGGER.warning("PR-AUC undefined (likely single-class eval).")
        else:
            LOGGER.info(f"PR-AUC: {metrics['pr_auc']:.4f}")
        for t in thresholds:
            m = metrics["thresholds"][str(t)]
            LOGGER.info(f"Threshold {t}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")
        
    # 5. Save artifacts
    LOGGER.info(f"Saving artifacts to {model_dir}")
    joblib.dump(clf, os.path.join(model_dir, "model.pkl"))
    
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
        
    with open(os.path.join(model_dir, "feature_list.json"), "w") as f:
        json.dump(feature_cols, f, indent=2, allow_nan=False)

    # Save resolved config for reproducibility
    with open(os.path.join(model_dir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
        
    # Fingerprint snapshot inputs
    snapshot_fingerprint: Dict[str, Any] = {"snapshot_path": snapshot_path}
    try:
        if os.path.isdir(snapshot_path):
            tp = os.path.join(snapshot_path, "train.parquet")
            ep = os.path.join(snapshot_path, "eval.parquet")
            if os.path.exists(tp):
                snapshot_fingerprint["train_parquet_sha256"] = _sha256_file(tp)
            if os.path.exists(ep):
                snapshot_fingerprint["eval_parquet_sha256"] = _sha256_file(ep)
            mp = os.path.join(snapshot_path, "metadata.json")
            if os.path.exists(mp):
                snapshot_fingerprint["snapshot_metadata_sha256"] = _sha256_file(mp)
        else:
            snapshot_fingerprint["parquet_sha256"] = _sha256_file(snapshot_path)
    except Exception as e:
        snapshot_fingerprint["warning"] = f"fingerprint_failed:{e}"

    metadata = {
        "run_id": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "feature_list": feature_cols,
        "split_info": {
            "train_size": len(train_df),
            "eval_size": len(eval_df),
            "train_pos_rate": float(y_train.mean()) if len(y_train) else None,
            "eval_pos_rate": float(y_eval.mean()) if len(y_eval) else None,
            "details": split_meta,
        },
        "training": {
            "handle_imbalance": bool(config.get("handle_imbalance", True)),
            "used_sample_weight": bool(sample_weight is not None),
        },
        "snapshot": snapshot_fingerprint,
        "package_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "sklearn": sklearn.__version__,
        }
    }
    metadata["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
        
    LOGGER.info("Training pipeline finished successfully.")

def main():
    parser = argparse.ArgumentParser(description="Train denoiser baseline model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default=None,
        help="Override snapshot_path from config (useful for pipeline automation).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.snapshot_path:
        config["snapshot_path"] = args.snapshot_path
    train_baseline(config)

if __name__ == "__main__":
    main()
