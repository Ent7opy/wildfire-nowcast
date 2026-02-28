"""Evaluate denoiser v2 event model and emit promotion gate artifacts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from api.db import get_engine
from ml.denoiser.coverage_authority import get_coverage_freshness, require_coverage_freshness


def _load_snapshot(path: str, *, columns: list[str]) -> pd.DataFrame:
    read_columns = list(dict.fromkeys(columns))
    def _read_with_fallback(parquet_path: str) -> pd.DataFrame:
        try:
            return pd.read_parquet(parquet_path, columns=read_columns)
        except Exception:
            full = pd.read_parquet(parquet_path)
            keep = [c for c in read_columns if c in full.columns]
            return full[keep]

    if os.path.isdir(path):
        eval_path = os.path.join(path, "eval.parquet")
        return _read_with_fallback(eval_path)
    return _read_with_fallback(path)


def _predict_raw(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    pred = np.asarray(model.predict(x), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def _apply_calibrator(cal: Dict[str, Any], scores: np.ndarray) -> np.ndarray:
    ctype = cal.get("type")
    model = cal.get("model")
    if ctype == "isotonic" and model is not None:
        return np.asarray(model.predict(scores), dtype=float)
    if ctype == "platt" and model is not None:
        return np.asarray(model.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)
    return np.asarray(scores, dtype=float)


def _slice_key(row: pd.Series, slice_cols: list[str]) -> str:
    return "|".join(f"{col}={row.get(col, 'unknown')}" for col in slice_cols)


def _feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].astype(np.float32)


def _metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else None
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "predicted_positive_rate": float((p >= threshold).mean()),
    }


def _sweep(y_true: np.ndarray, p: np.ndarray, step: float = 0.01) -> pd.DataFrame:
    rows = [_metrics(y_true, p, t) for t in np.round(np.arange(0.0, 1.0 + 1e-12, step), 10)]
    return pd.DataFrame(rows)


def _pick_strong_filter(sweep: pd.DataFrame, target_precision: float) -> Dict[str, Any]:
    cand = sweep[sweep["precision"] >= target_precision]
    if not cand.empty:
        return cand.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0].to_dict()
    return sweep.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0].to_dict()


def _pick_downweight(sweep: pd.DataFrame, target_recall: float) -> Dict[str, Any]:
    cand = sweep[sweep["recall"] >= target_recall]
    if not cand.empty:
        return cand.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).iloc[0].to_dict()
    return sweep.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).iloc[0].to_dict()


def _collect_mask_ids(series: pd.Series) -> list[str]:
    out: set[str] = set()
    for value in series:
        if value is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            for item in value:
                if item is not None:
                    out.add(str(item))
            continue
        if isinstance(value, str):
            out.add(value)
    return sorted(out)


def _load_industrial_policy_provenance(policy_version: str | None = None) -> dict[str, Any] | None:
    policy_stmt = text(
        """
        SELECT
            policy_version,
            strict_no_go,
            gold_buffer_m,
            silver_buffer_min_m,
            silver_buffer_max_m,
            active_from,
            active_to
        FROM industrial_mask_policies
        WHERE (
                :policy_version IS NOT NULL
                AND policy_version = :policy_version
              )
           OR (
                :policy_version IS NULL
                AND (active_to IS NULL OR active_to > NOW())
              )
        ORDER BY active_from DESC, policy_version DESC
        LIMIT 1
        """
    )
    tier_stats_stmt = text(
        """
        SELECT
            COUNT(*) FILTER (WHERE COALESCE(is_active, TRUE) AND authority_tier = 'gold') AS gold_sources,
            COUNT(*) FILTER (WHERE COALESCE(is_active, TRUE) AND authority_tier = 'silver') AS silver_sources,
            COUNT(*) FILTER (WHERE COALESCE(is_active, TRUE) AND authority_tier = 'blocked') AS blocked_sources,
            COUNT(*) FILTER (WHERE COALESCE(is_active, TRUE)) AS active_sources
        FROM industrial_sources
        """
    )
    no_go_stmt = text(
        """
        SELECT COUNT(*) AS active_no_go_zones
        FROM industrial_no_go_zones
        WHERE is_active
          AND (:policy_version IS NULL OR policy_version = :policy_version)
        """
    )
    try:
        with get_engine().begin() as conn:
            policy_row = conn.execute(
                policy_stmt,
                {"policy_version": policy_version},
            ).mappings().first()
            if policy_row is None:
                return None
            resolved_policy_version = str(policy_row["policy_version"])
            tier_row = conn.execute(tier_stats_stmt).mappings().first()
            no_go_row = conn.execute(
                no_go_stmt,
                {"policy_version": resolved_policy_version},
            ).mappings().first()
    except Exception:
        return None

    return {
        "policy_version": str(policy_row["policy_version"]),
        "strict_no_go": bool(policy_row["strict_no_go"]),
        "gold_buffer_m": float(policy_row["gold_buffer_m"]),
        "silver_buffer_min_m": float(policy_row["silver_buffer_min_m"]),
        "silver_buffer_max_m": float(policy_row["silver_buffer_max_m"]),
        "active_from": policy_row["active_from"].isoformat() if policy_row["active_from"] is not None else None,
        "active_to": policy_row["active_to"].isoformat() if policy_row["active_to"] is not None else None,
        "active_sources": int((tier_row or {}).get("active_sources") or 0),
        "gold_sources": int((tier_row or {}).get("gold_sources") or 0),
        "silver_sources": int((tier_row or {}).get("silver_sources") or 0),
        "blocked_sources": int((tier_row or {}).get("blocked_sources") or 0),
        "active_no_go_zones": int((no_go_row or {}).get("active_no_go_zones") or 0),
    }


def _evaluate_scope(
    *,
    scope_name: str,
    eval_known: pd.DataFrame,
    y: np.ndarray,
    calibrated: np.ndarray,
    default_threshold: float,
    target_precision: float,
    target_recall: float,
    threshold_step: float,
    gate_thresholds: dict[str, Any],
    latency_per_10k: float,
) -> dict[str, Any]:
    sweep = _sweep(y, calibrated, step=threshold_step)
    strong = _pick_strong_filter(sweep, target_precision=target_precision)
    downweight = _pick_downweight(sweep, target_recall=target_recall)
    default_metrics = _metrics(y, calibrated, threshold=default_threshold)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    gate_results = {
        "event_recall": {
            "value": float(default_metrics["recall"]),
            "pass": float(default_metrics["recall"]) >= gate_thresholds["event_recall_min"],
        },
        "event_precision": {
            "value": float(default_metrics["precision"]),
            "pass": float(default_metrics["precision"]) >= gate_thresholds["event_precision_min"],
        },
        "global_f1": {
            "value": float(default_metrics["f1"]),
            "pass": float(default_metrics["f1"]) >= gate_thresholds["global_f1_min"],
        },
        "roc_auc": {
            "value": default_metrics["roc_auc"],
            "pass": default_metrics["roc_auc"] is not None
            and float(default_metrics["roc_auc"]) >= gate_thresholds["roc_auc_min"],
        },
        "latency_per_10k_seconds": {
            "value": latency_per_10k,
            "pass": latency_per_10k <= gate_thresholds["latency_per_10k_max_seconds"],
        },
        "min_event_positives": {
            "value": n_pos,
            "pass": n_pos >= gate_thresholds["min_event_positives"],
        },
        "min_event_negatives": {
            "value": n_neg,
            "pass": n_neg >= gate_thresholds["min_event_negatives"],
        },
    }
    gate_pass = all(bool(x["pass"]) for x in gate_results.values())

    slice_metrics: dict[str, list[dict[str, Any]]] = {}
    for col in [c for c in ["sensor", "biome_slice", "is_day_ratio"] if c in eval_known.columns]:
        rows: list[dict[str, Any]] = []
        for key, grp in eval_known.groupby(col, dropna=False):
            idx = grp.index.to_numpy()
            yy = y[idx]
            pp = calibrated[idx]
            rows.append({
                col: str(key),
                "n": int(len(grp)),
                **_metrics(yy, pp, threshold=default_threshold),
            })
        slice_metrics[col] = rows

    return {
        "scope": scope_name,
        "n_eval": int(len(eval_known)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "default_metrics": default_metrics,
        "threshold_recommendations": {
            "strong_filter": strong,
            "downweight": downweight,
        },
        "gate_results": gate_results,
        "gate_pass": gate_pass,
        "slice_metrics": slice_metrics,
        "sweep": sweep,
    }


def evaluate_denoiser_v2(
    model_run_dir: str,
    snapshot_path: str,
    out_dir: str,
    *,
    target_precision: float = 0.9,
    target_recall: float = 0.9,
    threshold_step: float = 0.01,
    write_db: bool = False,
    model_id: str | None = None,
    gate_scope: str = "covered",
    coverage_mask_source: str = "db_mask",
    fail_on_missing_coverage_mask: bool = True,
    coverage_authority_profile: str = "wfigs_us",
    coverage_max_age_hours: float = 72.0,
    fail_on_stale_coverage_mask: bool = True,
    industrial_policy_version: str | None = None,
) -> str:
    gate_scope = str(gate_scope).strip().lower()
    if gate_scope not in {"covered", "global", "both"}:
        raise ValueError("gate_scope must be one of: covered, global, both")

    bundle = joblib.load(os.path.join(model_run_dir, "model_bundle.pkl"))
    model = bundle["model"]
    features = list(bundle["features"])
    slice_cols = list(bundle.get("slice_cols", ["sensor", "biome_slice"]))
    global_calibrator = bundle["global_calibrator"]
    slice_calibrators = dict(bundle.get("slice_calibrators", {}))

    snapshot_columns = list(
        dict.fromkeys(
            features
            + ["event_label", "sensor", "biome_slice", "is_day_ratio", "truth_covered_mask", "coverage_mask_ids"]
            + slice_cols
        )
    )
    eval_df = _load_snapshot(snapshot_path, columns=snapshot_columns).copy()
    for col in features:
        if col not in eval_df.columns:
            eval_df[col] = np.nan

    label_map = {"POSITIVE": 1, "NEGATIVE": 0}
    y_all = eval_df["event_label"].map(label_map).fillna(-1).astype(int).to_numpy()
    known_mask = y_all >= 0
    eval_known = eval_df.loc[known_mask].reset_index(drop=True)
    y = y_all[known_mask]
    if len(eval_known) == 0:
        raise ValueError("No known labels in eval snapshot for denoiser v2 evaluation.")

    raw = _predict_raw(model, _feature_matrix(eval_known, features))
    calibrated = np.zeros(len(eval_known), dtype=float)

    for idx, row in enumerate(eval_known.itertuples(index=False)):
        row_series = pd.Series(row._asdict())
        key = _slice_key(row_series, slice_cols)
        cal = slice_calibrators.get(key, global_calibrator)
        calibrated[idx] = float(_apply_calibrator(cal, np.asarray([raw[idx]]))[0])

    default_threshold = float(bundle.get("thresholds", {}).get("decision", 0.5))
    latency_per_10k = float(bundle.get("latency_per_10k_seconds", 0.0))

    gate_thresholds = {
        "event_recall_min": 0.92,
        "event_precision_min": 0.75,
        "global_f1_min": 0.85,
        "roc_auc_min": 0.95,
        "latency_per_10k_max_seconds": 300.0,
        "min_event_positives": 50,
        "min_event_negatives": 50,
    }

    global_eval = _evaluate_scope(
        scope_name="global",
        eval_known=eval_known,
        y=y,
        calibrated=calibrated,
        default_threshold=default_threshold,
        target_precision=target_precision,
        target_recall=target_recall,
        threshold_step=threshold_step,
        gate_thresholds=gate_thresholds,
        latency_per_10k=latency_per_10k,
    )

    covered_eval: dict[str, Any] | None = None
    coverage_freshness: dict[str, Any] | None = None
    needs_covered = gate_scope in {"covered", "both"}
    has_coverage_col = "truth_covered_mask" in eval_known.columns
    if needs_covered:
        if coverage_mask_source == "db_mask":
            if fail_on_stale_coverage_mask:
                coverage_freshness = require_coverage_freshness(
                    authority_profile=coverage_authority_profile,
                    max_age_hours=float(coverage_max_age_hours),
                )
            else:
                coverage_freshness = get_coverage_freshness(
                    authority_profile=coverage_authority_profile,
                    max_age_hours=float(coverage_max_age_hours),
                )
        if not has_coverage_col:
            if fail_on_missing_coverage_mask:
                raise ValueError("gate_scope requires truth_covered_mask in snapshot but column is missing")
        else:
            covered_mask = eval_known["truth_covered_mask"].fillna(False).astype(bool).to_numpy()
            covered_idx = np.flatnonzero(covered_mask)
            if covered_idx.size == 0:
                if fail_on_missing_coverage_mask:
                    raise ValueError("gate_scope requires covered rows but none found in truth_covered_mask")
            else:
                covered_eval = _evaluate_scope(
                    scope_name="covered",
                    eval_known=eval_known.iloc[covered_idx].reset_index(drop=True),
                    y=y[covered_idx],
                    calibrated=calibrated[covered_idx],
                    default_threshold=default_threshold,
                    target_precision=target_precision,
                    target_recall=target_recall,
                    threshold_step=threshold_step,
                    gate_thresholds=gate_thresholds,
                    latency_per_10k=latency_per_10k,
                )

    if gate_scope == "global":
        primary_name = "global"
        primary_eval = global_eval
    elif gate_scope == "covered":
        primary_name = "covered" if covered_eval is not None else "global"
        primary_eval = covered_eval if covered_eval is not None else global_eval
    else:  # both
        primary_name = "covered" if covered_eval is not None else "global"
        primary_eval = covered_eval if covered_eval is not None else global_eval

    gate_pass = bool(primary_eval["gate_pass"])

    os.makedirs(out_dir, exist_ok=True)

    global_eval["sweep"].to_csv(os.path.join(out_dir, "threshold_sweep_global.csv"), index=False)
    if covered_eval is not None:
        covered_eval["sweep"].to_csv(os.path.join(out_dir, "threshold_sweep_covered.csv"), index=False)
    primary_eval["sweep"].to_csv(os.path.join(out_dir, "threshold_sweep.csv"), index=False)

    coverage_mask_ids = []
    if "coverage_mask_ids" in eval_known.columns:
        coverage_mask_ids = _collect_mask_ids(eval_known["coverage_mask_ids"])
    industrial_policy = _load_industrial_policy_provenance(policy_version=industrial_policy_version)

    summary = {
        "run_id": os.path.basename(model_run_dir.rstrip(os.sep)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "gate_scope": gate_scope,
        "gate_scope_primary": primary_name,
        "coverage_mask_source": coverage_mask_source,
        "coverage_authority_profile": coverage_authority_profile,
        "coverage_data_freshness": coverage_freshness,
        "coverage_run_id": (coverage_freshness or {}).get("run_id"),
        "coverage_mask_ids": coverage_mask_ids,
        "industrial_policy": industrial_policy,
        "n_eval": int(primary_eval["n_eval"]),
        "n_pos": int(primary_eval["n_pos"]),
        "n_neg": int(primary_eval["n_neg"]),
        "default_threshold": default_threshold,
        "default_metrics": primary_eval["default_metrics"],
        "threshold_recommendations": {
            "strong_filter": primary_eval["threshold_recommendations"]["strong_filter"],
            "downweight": primary_eval["threshold_recommendations"]["downweight"],
            "uncertainty_band_low": float(bundle.get("thresholds", {}).get("uncertainty_band_low", 0.45)),
            "uncertainty_band_high": float(bundle.get("thresholds", {}).get("uncertainty_band_high", 0.55)),
        },
        "gate_thresholds": gate_thresholds,
        "gate_results": primary_eval["gate_results"],
        "gate_pass": gate_pass,
        "global": {
            "n_eval": int(global_eval["n_eval"]),
            "n_pos": int(global_eval["n_pos"]),
            "n_neg": int(global_eval["n_neg"]),
            "default_metrics": global_eval["default_metrics"],
            "gate_results": global_eval["gate_results"],
            "gate_pass": bool(global_eval["gate_pass"]),
            "threshold_recommendations": global_eval["threshold_recommendations"],
        },
        "covered": None,
    }
    if covered_eval is not None:
        summary["covered"] = {
            "n_eval": int(covered_eval["n_eval"]),
            "n_pos": int(covered_eval["n_pos"]),
            "n_neg": int(covered_eval["n_neg"]),
            "default_metrics": covered_eval["default_metrics"],
            "gate_results": covered_eval["gate_results"],
            "gate_pass": bool(covered_eval["gate_pass"]),
            "threshold_recommendations": covered_eval["threshold_recommendations"],
        }

    slice_metrics = {
        "primary": primary_eval["slice_metrics"],
        "global": global_eval["slice_metrics"],
        "covered": covered_eval["slice_metrics"] if covered_eval is not None else None,
    }

    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "slice_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(slice_metrics, f, indent=2)

    gate_report = {
        "run_id": summary["run_id"],
        "pass": gate_pass,
        "gate_scope": gate_scope,
        "gate_scope_primary": primary_name,
        "coverage_mask_source": coverage_mask_source,
        "coverage_authority_profile": coverage_authority_profile,
        "coverage_data_freshness": coverage_freshness,
        "coverage_run_id": (coverage_freshness or {}).get("run_id"),
        "coverage_mask_ids": coverage_mask_ids,
        "industrial_policy": industrial_policy,
        "thresholds": gate_thresholds,
        "results": primary_eval["gate_results"],
        "global_results": global_eval["gate_results"],
        "covered_results": covered_eval["gate_results"] if covered_eval is not None else None,
        "evaluated_at": summary["evaluated_at"],
    }
    with open(os.path.join(out_dir, "gate_report.json"), "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=2)

    with open(os.path.join(out_dir, "thresholds.md"), "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"# denoiser-v2 thresholds ({summary['run_id']})",
                    "",
                    f"- gate_scope: {gate_scope}",
                    f"- gate_scope_primary: {primary_name}",
                    f"- strong_filter_threshold: {float(summary['threshold_recommendations']['strong_filter']['threshold']):.2f}",
                    f"- downweight_threshold: {float(summary['threshold_recommendations']['downweight']['threshold']):.2f}",
                    f"- uncertainty_band_low: {summary['threshold_recommendations']['uncertainty_band_low']:.2f}",
                    f"- uncertainty_band_high: {summary['threshold_recommendations']['uncertainty_band_high']:.2f}",
                    "",
                    f"- gate_pass: {gate_pass}",
                ]
            )
        )

    if write_db:
        run_id = f"eval_{summary['run_id']}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        stmt = text(
            """
            INSERT INTO denoiser_eval_runs (
                run_id,
                model_id,
                family,
                status,
                metrics_json,
                gate_report_json,
                slice_metrics_json,
                artifact_uri,
                evaluated_at,
                created_at
            )
            VALUES (
                :run_id,
                :model_id,
                'denoiser',
                :status,
                :metrics_json,
                :gate_report_json,
                :slice_metrics_json,
                :artifact_uri,
                NOW(),
                NOW()
            )
            """
        )
        with get_engine().begin() as conn:
            conn.execute(
                stmt,
                {
                    "run_id": run_id,
                    "model_id": model_id,
                    "status": "passed" if gate_pass else "failed",
                    "metrics_json": json.dumps(summary),
                    "gate_report_json": json.dumps(gate_report),
                    "slice_metrics_json": json.dumps(slice_metrics),
                    "artifact_uri": model_run_dir,
                },
            )

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate denoiser v2 and emit gate report.")
    parser.add_argument("--model_run", type=str, required=True)
    parser.add_argument("--snapshot", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--target_precision", type=float, default=0.9)
    parser.add_argument("--target_recall", type=float, default=0.9)
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--gate-scope", type=str, default="covered", choices=["covered", "global", "both"])
    parser.add_argument("--coverage-mask-source", type=str, default="db_mask")
    parser.add_argument("--coverage-authority-profile", type=str, default="wfigs_us")
    parser.add_argument("--coverage-max-age-hours", type=float, default=72.0)
    parser.add_argument("--industrial-policy-version", type=str, default=None)
    parser.add_argument(
        "--fail-on-missing-coverage-mask",
        dest="fail_on_missing_coverage_mask",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--allow-missing-coverage-mask",
        dest="fail_on_missing_coverage_mask",
        action="store_false",
    )
    parser.add_argument(
        "--fail-on-stale-coverage-mask",
        dest="fail_on_stale_coverage_mask",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--allow-stale-coverage-mask",
        dest="fail_on_stale_coverage_mask",
        action="store_false",
    )
    args = parser.parse_args()

    out_dir = evaluate_denoiser_v2(
        model_run_dir=args.model_run,
        snapshot_path=args.snapshot,
        out_dir=args.out,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
        threshold_step=args.threshold_step,
        write_db=args.write_db,
        model_id=args.model_id,
        gate_scope=args.gate_scope,
        coverage_mask_source=args.coverage_mask_source,
        fail_on_missing_coverage_mask=bool(args.fail_on_missing_coverage_mask),
        coverage_authority_profile=args.coverage_authority_profile,
        coverage_max_age_hours=float(args.coverage_max_age_hours),
        fail_on_stale_coverage_mask=bool(args.fail_on_stale_coverage_mask),
        industrial_policy_version=args.industrial_policy_version,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
