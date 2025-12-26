"""Train learned spread model (v1) baseline."""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from ml.spread.hindcast_dataset import build_hindcast_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("train_spread_v1")

def _maybe_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        sha = out.strip()
        return sha if sha else None
    except Exception:
        return None

def _fmt_float(v: Any) -> str:
    try:
        if v is None:
            return "null"
        fv = float(v)
        if not np.isfinite(fv):
            return "null"
        return f"{fv:.4f}"
    except Exception:
        return "null"


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _safe_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    """Return ROC-AUC/PR-AUC, or nulls when undefined."""
    unique = np.unique(y_true)
    if unique.size < 2:
        return {"roc_auc": None, "pr_auc": None}

    roc = float(roc_auc_score(y_true, y_score))
    pr = float(average_precision_score(y_true, y_score))
    return {"roc_auc": roc, "pr_auc": pr}


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, thresholds: List[float]):
    auc = _safe_auc_metrics(y_true, y_score)
    metrics = {**auc, "thresholds": {}}

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        # Simple IoU (Jaccard) for binary masks
        intersection = tp
        union = tp + fp + fn
        iou = float(intersection / union) if union > 0 else 0.0

        metrics["thresholds"][str(t)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    return metrics


def train_spread_v1(config: Dict[str, Any]):
    # 1. Setup run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = _maybe_git_sha() or "unknown"
    run_name = f"{timestamp}_{git_sha}"
    model_dir = os.path.join(config.get("model_output_root", "models/spread_v1"), run_name)
    os.makedirs(model_dir, exist_ok=True)

    # 2. Build dataset
    LOGGER.info("Building hindcast dataset...")
    start_time = datetime.fromisoformat(config["start_time"]).replace(tzinfo=timezone.utc)
    end_time = datetime.fromisoformat(config["end_time"]).replace(tzinfo=timezone.utc)
    
    df = build_hindcast_dataset(
        region_name=config["region_name"],
        bbox=tuple(config["bbox"]),
        start_time=start_time,
        end_time=end_time,
        horizons_hours=config["horizons_hours"],
        min_detections=config.get("min_detections", 5),
        interval_hours=config.get("interval_hours", 24),
        negative_ratio=config.get("negative_ratio", 5.0),
        seed=config.get("seed", 42),
    )

    if df.empty:
        LOGGER.error("Hindcast dataset is empty. Check your data or parameters.")
        return

    LOGGER.info(f"Loaded {len(df)} samples total.")

    # 3. Time-based split
    df = df.copy()
    df["ref_time"] = pd.to_datetime(df["ref_time"], utc=True)
    try:
        split_date = df["ref_time"].quantile(config.get("split_percentile", 0.8))
    except Exception:
        # Fallback: split using ordered unique ref_times to avoid dtype quirks.
        uniq = np.sort(df["ref_time"].dropna().unique())
        if uniq.size == 0:
            LOGGER.error("No valid ref_time values available for splitting.")
            return
        idx = int(np.floor(float(config.get("split_percentile", 0.8)) * (uniq.size - 1)))
        split_date = pd.Timestamp(uniq[max(0, min(idx, uniq.size - 1))], tz="UTC")
    train_df = df[df["ref_time"] < split_date].copy()
    eval_df = df[df["ref_time"] >= split_date].copy()
    
    LOGGER.info(f"Split at {split_date}: {len(train_df)} train, {len(eval_df)} eval.")

    feature_cols = config["features"]
    
    # 4. Train per horizon
    horizons = config["horizons_hours"]
    models = {}
    horizon_metrics = {}

    for h in horizons:
        LOGGER.info(f"Training model for horizon T+{h}h...")
        h_train = train_df[train_df["horizon_h"] == h]
        h_eval = eval_df[eval_df["horizon_h"] == h]

        if h_train.empty:
            LOGGER.warning(f"No training data for horizon {h}; skipping.")
            continue

        X_train = h_train[feature_cols]
        y_train = h_train["label"]
        
        # Default seeds and params
        model_params = config.get("model_params", {})
        if "random_state" not in model_params:
            model_params["random_state"] = config.get("seed", 42)
            
        clf = HistGradientBoostingClassifier(**model_params)
        clf.fit(X_train, y_train)
        
        models[h] = clf
        
        # Evaluate
        if not h_eval.empty:
            X_eval = h_eval[feature_cols]
            y_eval = h_eval["label"]
            y_prob = clf.predict_proba(X_eval)[:, 1]
            
            metrics = compute_metrics(
                y_eval.to_numpy(), y_prob, config.get("eval_thresholds", [0.5])
            )
            horizon_metrics[h] = metrics
            
            # Avoid formatting crashes when AUC metrics are undefined (single-class eval).
            default_t = str((config.get("eval_thresholds") or [0.5])[0])
            LOGGER.info(
                "T+%sh metrics: roc_auc=%s pr_auc=%s iou(%s)=%s",
                h,
                _fmt_float(metrics.get("roc_auc")),
                _fmt_float(metrics.get("pr_auc")),
                default_t,
                _fmt_float(metrics.get("thresholds", {}).get(default_t, {}).get("iou")),
            )

    # 5. Save artifacts
    LOGGER.info(f"Saving artifacts to {model_dir}")
    # Save a dict of models mapping horizon -> clf
    joblib.dump(models, os.path.join(model_dir, "model.pkl"))
    
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(horizon_metrics, f, indent=2)
        
    with open(os.path.join(model_dir, "feature_list.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(os.path.join(model_dir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    metadata = {
        "run_id": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "config": config,
        "horizons": horizons,
        "split_date": split_date.isoformat(),
        "counts": {
            "total": len(df),
            "train": len(train_df),
            "eval": len(eval_df),
        }
    }
    metadata["package_versions"] = {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }
    metadata["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info("Training pipeline finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train learned spread v1 model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_spread_v1(config)


if __name__ == "__main__":
    main()

