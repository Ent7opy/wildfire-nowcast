"""
Evaluate a trained denoiser_v1 model on a labeled snapshot dataset and choose default thresholds.

Usage:
  python -m ml.eval_denoiser --model_run models/denoiser_v1/<run_id>/ --snapshot <snapshot.parquet|snapshot_dir> [--out reports/denoiser_v1/<run_id>/]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


@dataclass(frozen=True)
class SplitDetails:
    """Represents the *actual* split behavior used during training (from metadata.split_info.details)."""

    strategy: str
    split_time: Optional[str]
    split_percentile: float
    gap_hours: int
    fallback_reason: Optional[str] = None
    fallback_strategy: Optional[str] = None
    eval_size: Optional[float] = None


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_default(o: Any) -> Any:
    """
    JSON serializer for objects produced by pandas/numpy.

    `pandas.Series.to_dict()` / `DataFrame.to_dict()` can contain NumPy scalars
    (e.g., `np.int64`, `np.float64`) which the stdlib `json` module cannot
    serialize by default.
    """
    # NumPy scalars / arrays
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()

    # pandas missing values / timestamps
    # Note: `pd.isna(...)` returns arrays for array-likes; only call it for
    # scalars via the try/except guard below.
    if o is pd.NA:
        return None
    if isinstance(o, pd.Timestamp):
        return o.isoformat()

    # datetime-like
    if isinstance(o, datetime):
        return o.isoformat()

    try:
        if pd.isna(o):
            return None
    except Exception:
        pass

    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _load_feature_list(model_run_dir: str) -> List[str]:
    fp = os.path.join(model_run_dir, "feature_list.json")
    if os.path.exists(fp):
        return list(_read_json(fp))
    meta = _read_json(os.path.join(model_run_dir, "metadata.json"))
    feats = meta.get("feature_list") or meta.get("config", {}).get("features")
    if not feats:
        raise ValueError("Could not find feature list in feature_list.json or metadata.json.")
    return list(feats)


def _load_metadata(model_run_dir: str) -> Dict[str, Any]:
    return _read_json(os.path.join(model_run_dir, "metadata.json"))


def _parse_split_details(metadata: Dict[str, Any]) -> SplitDetails:
    details = (metadata.get("split_info") or {}).get("details") or {}
    cfg = metadata.get("config") or {}
    eval_size = details.get("eval_size", cfg.get("eval_size"))
    return SplitDetails(
        strategy=str(details.get("strategy") or cfg.get("split_strategy") or "time"),
        split_time=details.get("split_time", cfg.get("split_time")),
        split_percentile=float(details.get("split_percentile", cfg.get("split_percentile", 0.8))),
        gap_hours=int(details.get("gap_hours", cfg.get("gap_hours", 0))),
        fallback_reason=details.get("fallback_reason"),
        fallback_strategy=details.get("fallback_strategy"),
        eval_size=(float(eval_size) if eval_size is not None else None),
    )


def _ensure_label_numeric(df: pd.DataFrame) -> pd.Series:
    if "label_numeric" not in df.columns:
        raise ValueError("Expected column 'label_numeric' in snapshot data.")
    y = pd.to_numeric(df["label_numeric"], errors="raise").astype(int)
    bad = set(y.unique()) - {0, 1}
    if bad:
        raise ValueError(f"label_numeric must be 0/1; found values {sorted(bad)}")
    return y


def _split_stratified_random(df: pd.DataFrame, eval_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic split that (when possible) forces at least one sample from each class into eval,
    so AUC metrics are defined.

    This mirrors `ml/train_denoiser.py::_split_stratified_random`.
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
        eval_seed_idx = np.concatenate(
            [
                rng.choice(pos_idx, size=1, replace=False),
                rng.choice(neg_idx, size=1, replace=False),
            ]
        )
        remaining = np.setdiff1d(idx, eval_seed_idx, assume_unique=False)
        remaining_eval_slots = n_eval - eval_seed_idx.size
        if remaining_eval_slots > 0:
            eval_rest = rng.choice(remaining, size=remaining_eval_slots, replace=False)
            eval_idx = np.concatenate([eval_seed_idx, eval_rest])
        else:
            eval_idx = eval_seed_idx
        train_idx = np.setdiff1d(idx, eval_idx, assume_unique=False)
        return df.loc[train_idx].copy(), df.loc[eval_idx].copy()

    # Best-effort stratified split.
    from sklearn.model_selection import train_test_split

    train_idx, eval_idx = train_test_split(
        idx,
        test_size=n_eval,
        random_state=seed,
        shuffle=True,
        stratify=y if np.unique(y).size >= 2 else None,
    )
    return df.loc[train_idx].copy(), df.loc[eval_idx].copy()


def _split_time(
    df: pd.DataFrame, split_time: Optional[str], split_percentile: float, gap_hours: int
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    df = df.copy()
    df["acq_time"] = pd.to_datetime(df["acq_time"])
    if split_time:
        split_dt = pd.to_datetime(split_time)
        split_kind = "time_explicit"
    else:
        split_dt = df["acq_time"].quantile(split_percentile)
        split_kind = "time_percentile"

    train_df = df[df["acq_time"] < split_dt].copy()
    eval_start = split_dt + pd.Timedelta(hours=gap_hours)
    eval_df = df[df["acq_time"] >= eval_start].copy()
    return train_df, eval_df, split_dt.isoformat(), split_kind


def _load_snapshot(snapshot_path: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Returns:
      - full_df: full dataset (if snapshot_path is a dir, concatenated train+eval; else the parquet)
      - presplit_train: if dir and train.parquet exists
      - presplit_eval: if dir and eval.parquet exists
    """
    if os.path.isdir(snapshot_path):
        train_path = os.path.join(snapshot_path, "train.parquet")
        eval_path = os.path.join(snapshot_path, "eval.parquet")
        presplit_train = pd.read_parquet(train_path) if os.path.exists(train_path) else None
        presplit_eval = pd.read_parquet(eval_path) if os.path.exists(eval_path) else None
        parts = [p for p in [presplit_train, presplit_eval] if p is not None]
        if not parts:
            raise ValueError(f"No parquet files found in snapshot dir: {snapshot_path}")
        full_df = pd.concat(parts, axis=0, ignore_index=True)
        return full_df, presplit_train, presplit_eval

    full_df = pd.read_parquet(snapshot_path)
    return full_df, None, None


def _align_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_cols]


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        return np.asarray(p, dtype=float)
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float)
        # logistic to [0,1]
        return 1.0 / (1.0 + np.exp(-s))
    raise ValueError("Model has neither predict_proba nor decision_function.")


def _f_beta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision + recall)
    return float((1 + b2) * precision * recall / denom) if denom > 0 else 0.0


def _threshold_sweep(y_true: np.ndarray, p: np.ndarray, step: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    thresholds = np.round(np.arange(0.0, 1.0 + 1e-12, step), 10)
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f05 = _f_beta(prec, rec, beta=0.5)
        rows.append(
            {
                "threshold": float(t),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "f0_5": f05,
                "predicted_positive_rate": float((p >= t).mean()),
            }
        )
    return pd.DataFrame(rows)


def _pick_strong_filter(sweep: pd.DataFrame, target_precision: float) -> Tuple[float, Dict[str, Any], str]:
    valid = sweep.copy()
    # Prefer thresholds with high precision; among those, maximize recall.
    cand = valid[(valid["precision"] >= target_precision) & ((valid["tp"] + valid["fp"]) > 0)]
    if len(cand) > 0:
        # If multiple thresholds tie, pick the higher threshold (more conservative).
        best = cand.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
        return float(best["threshold"]), best.to_dict(), f"met_target_precision>={target_precision:.2f}_maximize_recall"
    # Fallback: maximize F0.5 (precision-weighted).
    best = valid.sort_values(["f0_5", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"]), best.to_dict(), f"fallback_maximize_f0_5_target_precision_unmet({target_precision:.2f})"


def _pick_downweight(
    sweep: pd.DataFrame,
    target_recall: float,
    min_downweight_rate: float,
) -> Tuple[float, Dict[str, Any], str]:
    # Avoid degenerate defaults (e.g., threshold=0.00) that would downweight nothing.
    # `predicted_positive_rate` is the fraction with p >= threshold (i.e., "full weight" group).
    # So to ensure at least `min_downweight_rate` gets downweighted, require:
    #   predicted_positive_rate <= 1 - min_downweight_rate
    min_downweight_rate = float(min_downweight_rate)
    if not (0.0 <= min_downweight_rate < 1.0):
        raise ValueError("--min_downweight_rate must be in [0, 1).")
    max_full_weight_rate = 1.0 - min_downweight_rate

    # Note: `strong_filter_threshold` (drop mode) and `downweight_threshold` (weight mode)
    # are independent operating points; we don't constrain downweight_threshold relative to strong.
    valid = sweep.copy()
    cand = valid[
        (valid["recall"] >= target_recall)
        & ((valid["tp"] + valid["fp"]) > 0)
        & (valid["predicted_positive_rate"] <= (max_full_weight_rate + 1e-12))
    ]
    if len(cand) > 0:
        # Prefer recall/coverage: among thresholds meeting target recall, maximize F1.
        # If tied, prefer lower thresholds (more permissive).
        best = cand.sort_values(
            ["f1", "recall", "precision", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
        return (
            float(best["threshold"]),
            best.to_dict(),
            f"met_target_recall>={target_recall:.2f}_maximize_f1_min_downweight_rate>={min_downweight_rate:.2f}",
        )

    # If nothing meets the "non-degenerate downweight" constraint, fall back to recall-driven F1 selection.
    cand2 = valid[(valid["recall"] >= target_recall) & ((valid["tp"] + valid["fp"]) > 0)]
    if len(cand2) > 0:
        best = cand2.sort_values(
            ["f1", "recall", "precision", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
        return float(best["threshold"]), best.to_dict(), f"met_target_recall>={target_recall:.2f}_maximize_f1"
    best = valid.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
    return float(best["threshold"]), best.to_dict(), f"fallback_maximize_f1_target_recall_unmet({target_recall:.2f})"


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: Optional[float], out_path: str) -> None:
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required to write plots; please install it in the ml environment.")
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})" if roc_auc is not None else "ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_pr(rec: np.ndarray, prec: np.ndarray, pr_auc: Optional[float], prevalence: float, out_path: str) -> None:
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required to write plots; please install it in the ml environment.")
    plt.figure(figsize=(6, 6))
    plt.plot(rec, prec, label=f"PR (AP={pr_auc:.4f})" if pr_auc is not None else "PR")
    plt.hlines(prevalence, 0, 1, linestyles="--", colors="gray", linewidth=1, label=f"prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_calibration(y_true: np.ndarray, p: np.ndarray, brier: float, out_path: str, n_bins: int = 10) -> None:
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required to write plots; please install it in the ml environment.")
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration (Brier={brier:.4f})")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_thresholds_md(
    out_dir: str,
    run_id: str,
    split_details: SplitDetails,
    metrics_summary: Dict[str, Any],
    strong: Dict[str, Any],
    downweight: Dict[str, Any],
    slice_notes: List[str],
    hard_example_notes: List[str],
) -> None:
    p_strong = float(strong["threshold"])
    p_dw = float(downweight["threshold"])
    lines: List[str] = []
    lines.append(f"# denoiser_v1 thresholds ({run_id})")
    lines.append("")
    lines.append("## Dataset / split context")
    lines.append(f"- eval_n: `{metrics_summary.get('n_eval')}`")
    lines.append(f"- eval_pos: `{metrics_summary.get('n_pos')}`")
    lines.append(f"- eval_neg: `{metrics_summary.get('n_neg')}`")
    lines.append(f"- prevalence (val): `{metrics_summary.get('prevalence')}`")
    lines.append("")
    warnings = metrics_summary.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("## Defaults")
    lines.append(f"- **strong_filter_threshold**: `{p_strong:.2f}`")
    lines.append(f"- **downweight_threshold**: `{p_dw:.2f}`")
    lines.append("")
    lines.append("## Rationale (auto-picked from validation sweep)")
    lines.append(f"- Strong filter selection: `{metrics_summary['threshold_selection']['strong_filter']['rule']}`")
    lines.append(f"- Downweight selection: `{metrics_summary['threshold_selection']['downweight']['rule']}`")
    lines.append("")
    lines.append("### Metrics at defaults (validation)")
    lines.append("")
    def _fmt_row(d: Dict[str, Any]) -> str:
        return (
            f"threshold={float(d['threshold']):.2f}, "
            f"P={float(d['precision']):.3f}, R={float(d['recall']):.3f}, F1={float(d['f1']):.3f}, "
            f"TP={int(d['tp'])}, FP={int(d['fp'])}, TN={int(d['tn'])}, FN={int(d['fn'])}, "
            f"kept={float(d['predicted_positive_rate']):.3f}"
        )
    lines.append(f"- **strong_filter_threshold**: {_fmt_row(strong)}")
    lines.append(f"- **downweight_threshold**: {_fmt_row(downweight)}")
    lines.append("")
    lines.append("## Curves / summary")
    lines.append(f"- ROC-AUC: `{metrics_summary.get('roc_auc')}`")
    lines.append(f"- PR-AUC (Average Precision): `{metrics_summary.get('pr_auc')}`")
    lines.append(f"- Brier score: `{metrics_summary.get('brier')}`")
    lines.append(f"- Prevalence (val): `{metrics_summary.get('prevalence')}`")
    lines.append("")
    lines.append("## Split policy (reproduced from training metadata)")
    lines.append(f"- strategy: `{split_details.strategy}`")
    lines.append(f"- split_percentile: `{split_details.split_percentile}`")
    lines.append(f"- split_time: `{split_details.split_time}`")
    lines.append(f"- gap_hours: `{split_details.gap_hours}`")
    if split_details.fallback_strategy:
        lines.append(f"- fallback_strategy: `{split_details.fallback_strategy}` (reason: `{split_details.fallback_reason}`)")
        if split_details.eval_size is not None:
            lines.append(f"- eval_size (fallback): `{split_details.eval_size}`")
    lines.append("")
    lines.append("## Downstream interpretation contract")
    lines.append("")
    lines.append("### Drop mode (precision-first)")
    lines.append("- If `p < strong_filter_threshold` then **drop** the detection from downstream.")
    lines.append("")
    lines.append("### Weight mode (recall/coverage-first)")
    lines.append("- Keep all detections, but if `p < downweight_threshold` apply a **weight**.")
    lines.append("")
    lines.append("#### Default weight mapping")
    lines.append("- Proposed default:")
    lines.append("  - `weight = max(w_min, p)` with `w_min = 0.1`")
    lines.append("- Where to apply weight (recommended):")
    lines.append("  - Multiply detection contribution in clustering density / risk-index aggregation / alert score by `weight`.")
    lines.append("")
    lines.append("## Failure modes (quick slices)")
    if slice_notes:
        for n in slice_notes:
            lines.append(f"- {n}")
    else:
        lines.append("- (No slice columns available in snapshot.)")
    if hard_example_notes:
        lines.append("")
        lines.append("### Hard examples (highest-p negatives / lowest-p positives)")
        for n in hard_example_notes:
            lines.append(f"- {n}")
    lines.append("")
    lines.append("### Likely failure categories (hypotheses)")
    lines.append("- **Low-signal marginal detections**: low FRP / low confidence detections can look like noise and may be downweighted or dropped at higher thresholds.")
    lines.append("- **Persistent hotspots / industrial sources**: can appear fire-like and produce high predicted probabilities for negatives; validate against known industrial POIs.")
    lines.append("- **Sensor/source artifacts**: differences between sources (e.g., SNPP vs NOAA20 streams) can shift calibration; monitor slice metrics once multi-sensor data is present.")
    lines.append("")

    with open(os.path.join(out_dir, "thresholds.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _metrics_at_threshold(sweep: pd.DataFrame, t: float) -> Dict[str, Any]:
    row = sweep.iloc[(sweep["threshold"] - t).abs().argsort()[:1]].iloc[0]
    return row.to_dict()


def _slice_metrics_at_threshold(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, t: float, group_col: str) -> Optional[pd.DataFrame]:
    if group_col not in df.columns:
        return None
    g = df[group_col].fillna("NULL")
    rows = []
    for key, idx in g.groupby(g).groups.items():
        yy = y[np.array(list(idx))]
        pp = p[np.array(list(idx))]
        if yy.size == 0:
            continue
        y_pred = (pp >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yy, y_pred, labels=[0, 1]).ravel()
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        rows.append(
            {
                group_col: key,
                "n": int(yy.size),
                "pos_rate": float(yy.mean()),
                "precision": prec,
                "recall": rec,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
    out = pd.DataFrame(rows).sort_values(["n"], ascending=False)
    return out


def evaluate(
    model_run_dir: str,
    snapshot_path: str,
    out_dir: Optional[str],
    threshold_step: float,
    target_precision: float,
    target_recall: float,
    min_downweight_rate: float,
    top_k_errors: int,
) -> str:
    model_run_dir = os.path.normpath(model_run_dir)
    run_id = os.path.basename(model_run_dir.rstrip(os.sep))

    if out_dir is None:
        out_dir = os.path.join("reports", "denoiser_v1", run_id)
    _ensure_dir(out_dir)

    metadata = _load_metadata(model_run_dir)
    cfg = metadata.get("config") or {}
    split_details = _parse_split_details(metadata)

    feature_cols = _load_feature_list(model_run_dir)
    model = joblib.load(os.path.join(model_run_dir, "model.pkl"))

    full_df, presplit_train, presplit_eval = _load_snapshot(snapshot_path)
    if "acq_time" not in full_df.columns:
        raise ValueError("Snapshot is missing required column 'acq_time'.")

    # Determine eval split to match training behavior.
    seed = int(cfg.get("seed", 42))
    split_meta_out: Dict[str, Any] = {"run_id": run_id, "evaluated_at_utc": _utc_now_compact()}

    if split_details.strategy == "snapshot_presplit":
        if presplit_eval is None:
            raise ValueError("Training used snapshot_presplit but provided snapshot path is not a snapshot dir with eval.parquet.")
        eval_df = presplit_eval.copy()
        split_meta_out["split_strategy_used"] = "snapshot_presplit"
    elif split_details.strategy == "stratified_random":
        if split_details.eval_size is None:
            raise ValueError("Missing eval_size for stratified_random split.")
        _, eval_df = _split_stratified_random(full_df, eval_size=float(split_details.eval_size), seed=seed)
        split_meta_out["split_strategy_used"] = "stratified_random"
        split_meta_out["eval_size"] = float(split_details.eval_size)
    elif split_details.strategy == "time":
        _, eval_time_df, split_dt_iso, split_kind = _split_time(
            full_df,
            split_time=split_details.split_time,
            split_percentile=split_details.split_percentile,
            gap_hours=split_details.gap_hours,
        )
        split_meta_out["split_strategy_used"] = split_kind
        split_meta_out["split_dt"] = split_dt_iso
        split_meta_out["gap_hours"] = int(split_details.gap_hours)

        # If training recorded a fallback strategy, reproduce it verbatim.
        if split_details.fallback_strategy:
            if split_details.fallback_strategy != "stratified_random":
                raise ValueError(f"Unsupported fallback_strategy: {split_details.fallback_strategy}")
            if split_details.eval_size is None:
                raise ValueError("Missing eval_size for fallback stratified_random split.")
            _, eval_df = _split_stratified_random(full_df, eval_size=float(split_details.eval_size), seed=seed)
            split_meta_out["fallback_strategy_used"] = "stratified_random"
            split_meta_out["fallback_reason"] = split_details.fallback_reason
            split_meta_out["eval_size"] = float(split_details.eval_size)
        else:
            eval_df = eval_time_df
    else:
        raise ValueError(f"Unsupported split strategy: {split_details.strategy}")

    # Ensure eval_df uses a dense 0..N-1 index so position-based arrays (y/p) and group slices align.
    eval_df = eval_df.reset_index(drop=True)

    # Ensure expected helper columns exist for slicing (best-effort, non-invasive).
    if "daynight" not in eval_df.columns and "raw_properties" in eval_df.columns:
        eval_df = eval_df.copy()
        eval_df["daynight"] = eval_df["raw_properties"].apply(lambda x: x.get("daynight") if isinstance(x, dict) else None)
    if "is_day" not in eval_df.columns and "daynight" in eval_df.columns:
        eval_df = eval_df.copy()
        eval_df["is_day"] = (eval_df["daynight"] == "D").astype(int)

    y = _ensure_label_numeric(eval_df).to_numpy()
    X = _align_features(eval_df, feature_cols)
    p = _predict_proba(model, X)

    # Drop any non-finite scores (and aligned labels) for metrics stability.
    finite = np.isfinite(p)
    dropped_scores = int((~finite).sum())
    if dropped_scores:
        p = p[finite]
        y = y[finite]
        eval_df = eval_df.loc[eval_df.index[finite]].copy()
        eval_df = eval_df.reset_index(drop=True)

    eval_df = eval_df.copy()
    eval_df["p_real_fire"] = p

    prevalence = float(y.mean()) if y.size else 0.0

    # Curves / AUCs (guard single-class).
    unique = np.unique(y)
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    if unique.size < 2:
        roc_auc = None
        pr_auc = None
        fpr = tpr = np.array([])
        pr_prec = pr_rec = np.array([])
    else:
        roc_auc = float(roc_auc_score(y, p))
        pr_auc = float(average_precision_score(y, p))
        fpr, tpr, _ = roc_curve(y, p)
        pr_prec, pr_rec, _ = precision_recall_curve(y, p)

    brier = float(brier_score_loss(y, p)) if y.size else None

    # Heuristic warnings about statistical usefulness of the eval set.
    warnings: List[str] = []
    n_eval = int(len(eval_df))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_eval < 200:
        warnings.append(f"small_eval_set(n={n_eval})_thresholds_may_be_unstable")
    if n_neg < 20:
        warnings.append(f"too_few_negatives(n_neg={n_neg})_precision_and_auc_unreliable")
    if n_pos < 20:
        warnings.append(f"too_few_positives(n_pos={n_pos})_recall_and_auc_unreliable")
    if unique.size < 2:
        warnings.append("single_class_eval_auc_undefined")

    # Threshold sweep.
    sweep = _threshold_sweep(y, p, step=threshold_step)
    sweep_path = os.path.join(out_dir, "threshold_sweep.csv")
    sweep.to_csv(sweep_path, index=False)

    # Choose defaults.
    strong_t, strong_row, strong_rule = _pick_strong_filter(sweep, target_precision=target_precision)
    down_t, down_row, down_rule = _pick_downweight(
        sweep,
        target_recall=target_recall,
        min_downweight_rate=min_downweight_rate,
    )

    # Key thresholds for confusion export.
    key_thresholds = sorted(set([0.2, 0.5, 0.8, round(strong_t, 2), round(down_t, 2)]))
    confusion_at: Dict[str, Any] = {}
    for t in key_thresholds:
        confusion_at[f"{t:.2f}"] = _metrics_at_threshold(sweep, float(t))
    with open(os.path.join(out_dir, "confusion_at_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(confusion_at, f, indent=2, allow_nan=False, default=_json_default)

    # Errors: top-K FP / FN.
    fp_df = eval_df[(eval_df["label_numeric"] == 0) & np.isfinite(eval_df["p_real_fire"])].sort_values("p_real_fire", ascending=False).head(top_k_errors)
    fn_df = eval_df[(eval_df["label_numeric"] == 1) & np.isfinite(eval_df["p_real_fire"])].sort_values("p_real_fire", ascending=True).head(top_k_errors)
    fp_df.to_parquet(os.path.join(out_dir, "errors_fp.parquet"), index=False)
    fn_df.to_parquet(os.path.join(out_dir, "errors_fn.parquet"), index=False)

    hard_example_notes: List[str] = []
    if not fp_df.empty:
        r = fp_df.iloc[0]
        hard_example_notes.append(
            "Highest-p negative: "
            f"id={r.get('id')}, time={r.get('acq_time')}, "
            f"sensor={r.get('sensor')}, source={r.get('source')}, "
            f"conf={r.get('confidence_norm')}, frp={r.get('frp')}, "
            f"p={float(r.get('p_real_fire')):.3f}"
        )
    if not fn_df.empty:
        r = fn_df.iloc[0]
        hard_example_notes.append(
            "Lowest-p positive: "
            f"id={r.get('id')}, time={r.get('acq_time')}, "
            f"sensor={r.get('sensor')}, source={r.get('source')}, "
            f"conf={r.get('confidence_norm')}, frp={r.get('frp')}, "
            f"p={float(r.get('p_real_fire')):.3f}"
        )

    # Slice metrics (at strong filter threshold for precision-first view).
    slice_notes: List[str] = []
    slice_outputs: List[Tuple[str, Optional[pd.DataFrame]]] = []
    for col in ["sensor", "daynight"]:
        sm = _slice_metrics_at_threshold(eval_df, y, p, t=strong_t, group_col=col)
        slice_outputs.append((col, sm))
        if sm is not None and not sm.empty:
            # Note worst precision group among decent-sized slices.
            sm2 = sm[sm["n"] >= max(5, int(0.05 * len(eval_df)))]
            if not sm2.empty:
                worst = sm2.sort_values(["precision", "n"], ascending=[True, False]).iloc[0]
                slice_notes.append(f"`{col}` lowest precision @ strong filter: {worst[col]} (n={int(worst['n'])}, P={float(worst['precision']):.3f}, R={float(worst['recall']):.3f})")

    # Confidence bins if available.
    if "confidence_norm" in eval_df.columns:
        bins = [-np.inf, 30, 60, 90, np.inf]
        labels = ["<=30", "30-60", "60-90", ">=90"]
        conf_bin = pd.cut(pd.to_numeric(eval_df["confidence_norm"], errors="coerce"), bins=bins, labels=labels)
        tmp = eval_df.copy()
        tmp["confidence_bin"] = conf_bin.astype(str).fillna("NULL")
        sm = _slice_metrics_at_threshold(tmp, y, p, t=strong_t, group_col="confidence_bin")
        slice_outputs.append(("confidence_bin", sm))
        if sm is not None and not sm.empty:
            worst = sm.sort_values(["precision", "n"], ascending=[True, False]).iloc[0]
            slice_notes.append(f"`confidence_bin` lowest precision @ strong filter: {worst['confidence_bin']} (n={int(worst['n'])}, P={float(worst['precision']):.3f})")

    for name, df_slice in slice_outputs:
        if df_slice is not None and not df_slice.empty:
            df_slice.to_csv(os.path.join(out_dir, f"slice_metrics_by_{name}.csv"), index=False)

    # Plots.
    if unique.size >= 2:
        _plot_roc(fpr, tpr, roc_auc, os.path.join(out_dir, "roc_curve.png"))
        _plot_pr(pr_rec, pr_prec, pr_auc, prevalence, os.path.join(out_dir, "pr_curve.png"))
    if brier is not None and unique.size >= 2:
        _plot_calibration(y, p, brier=brier, out_path=os.path.join(out_dir, "calibration.png"))

    # Summary JSON.
    metrics_summary: Dict[str, Any] = {
        "run_id": run_id,
        "model_run_dir": model_run_dir,
        "snapshot_path": snapshot_path,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_eval": n_eval,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": prevalence,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "dropped_non_finite_scores": dropped_scores,
        "warnings": warnings,
        "split_details": split_details.__dict__,
        "split_replay": split_meta_out,
        "threshold_selection": {
            "strong_filter": {"threshold": strong_t, "rule": strong_rule, "metrics": strong_row},
            "downweight": {"threshold": down_t, "rule": down_rule, "metrics": down_row},
        },
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, allow_nan=False, default=_json_default)

    _write_thresholds_md(
        out_dir=out_dir,
        run_id=run_id,
        split_details=split_details,
        metrics_summary=metrics_summary,
        strong=strong_row,
        downweight=down_row,
        slice_notes=slice_notes,
        hard_example_notes=hard_example_notes,
    )

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate denoiser_v1 and choose default thresholds.")
    parser.add_argument("--model_run", type=str, required=True, help="Path to model run directory (contains model.pkl, metadata.json, feature_list.json).")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to labeled snapshot Parquet or snapshot dir containing train.parquet/eval.parquet.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: reports/denoiser_v1/<run_id>/)")
    parser.add_argument("--threshold_step", type=float, default=0.01, help="Step size for threshold sweep (default: 0.01)")
    parser.add_argument("--target_precision", type=float, default=0.90, help="Target precision for strong_filter_threshold (default: 0.90)")
    parser.add_argument("--target_recall", type=float, default=0.90, help="Target recall for downweight_threshold (default: 0.90)")
    parser.add_argument("--min_downweight_rate", type=float, default=0.01, help="Ensure at least this fraction would be downweighted (default: 0.01)")
    parser.add_argument("--top_k_errors", type=int, default=200, help="Top-K errors to save for FP/FN (default: 200)")
    args = parser.parse_args()

    out_dir = evaluate(
        model_run_dir=args.model_run,
        snapshot_path=args.snapshot,
        out_dir=args.out,
        threshold_step=args.threshold_step,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
        min_downweight_rate=args.min_downweight_rate,
        top_k_errors=args.top_k_errors,
    )
    print(out_dir)


if __name__ == "__main__":
    main()


