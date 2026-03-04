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
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import BallTree

from api.db import get_engine
from ml.denoiser.coverage_authority import get_coverage_freshness, require_coverage_freshness
from ml.parquet_io import read_parquet_with_fallback


class ModelPromotionError(RuntimeError):
    """Raised when strict denoiser promotion gates fail."""


_EARTH_RADIUS_M = 6_371_000.0
_EVENT_MATCH_BUFFER_M = 2315.0
_INDUSTRIAL_GOLD_BUFFER_M = 375.0
_INDUSTRIAL_SILVER_BUFFER_M = 750.0
_FAIL_CLOSED_FRP_MW = 500.0


def _load_snapshot(path: str, *, columns: list[str]) -> pd.DataFrame:
    read_columns = list(dict.fromkeys(columns))

    def _read_with_fallback(parquet_path: str) -> pd.DataFrame:
        try:
            return read_parquet_with_fallback(parquet_path, columns=read_columns)
        except Exception:
            full = read_parquet_with_fallback(parquet_path)
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


def _extract_eval_coordinates(eval_known: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    lat_col = next((c for c in ("lat_centroid", "lat") if c in eval_known.columns), None)
    lon_col = next((c for c in ("lon_centroid", "lon") if c in eval_known.columns), None)
    if lat_col is None or lon_col is None:
        return np.empty((0, 2), dtype=float), np.asarray([], dtype=int), "unavailable"

    lat = pd.to_numeric(eval_known[lat_col], errors="coerce")
    lon = pd.to_numeric(eval_known[lon_col], errors="coerce")
    valid = lat.notna() & lon.notna()
    if not bool(valid.any()):
        return np.empty((0, 2), dtype=float), np.asarray([], dtype=int), f"{lat_col},{lon_col}"

    coords = np.column_stack(
        [
            lat.loc[valid].to_numpy(dtype=float),
            lon.loc[valid].to_numpy(dtype=float),
        ]
    )
    valid_idx = np.flatnonzero(valid.to_numpy(dtype=bool))
    return coords, valid_idx.astype(int), f"{lat_col},{lon_col}"


def _cluster_event_points(coords_deg: np.ndarray, *, eps_m: float) -> np.ndarray:
    if coords_deg.size == 0:
        return np.asarray([], dtype=int)
    coords_rad = np.radians(coords_deg)
    eps_rad = float(eps_m) / _EARTH_RADIUS_M
    labels = DBSCAN(
        eps=eps_rad,
        min_samples=1,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coords_rad)
    return np.asarray(labels, dtype=int)


def _event_level_metrics(
    *,
    eval_known: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    match_buffer_m: float = _EVENT_MATCH_BUFFER_M,
) -> Dict[str, Any]:
    coords_deg, valid_df_idx, coord_basis = _extract_eval_coordinates(eval_known)
    if coords_deg.size == 0:
        return {
            "threshold": float(threshold),
            "match_buffer_m": float(match_buffer_m),
            "coordinate_basis": coord_basis,
            "event_recall": None,
            "event_precision": None,
            "event_f1": None,
            "ground_truth_events": 0,
            "predicted_events": 0,
            "detected_ground_truth_events": 0,
            "matched_predicted_events": 0,
            "tp_events": 0,
            "fp_events": 0,
            "fn_events": 0,
            "usable_point_rows": 0,
            "total_rows": int(len(eval_known)),
            "unusable_rows_missing_coords": int(len(eval_known)),
        }

    df_to_coord = np.full(len(eval_known), -1, dtype=int)
    df_to_coord[valid_df_idx] = np.arange(len(valid_df_idx), dtype=int)

    gt_df_idx = np.flatnonzero(np.asarray(y_true, dtype=int) == 1)
    pred_df_idx = np.flatnonzero(np.asarray(probabilities, dtype=float) >= float(threshold))
    gt_coord_idx = df_to_coord[gt_df_idx]
    pred_coord_idx = df_to_coord[pred_df_idx]
    gt_coord_idx = gt_coord_idx[gt_coord_idx >= 0]
    pred_coord_idx = pred_coord_idx[pred_coord_idx >= 0]

    gt_coords = coords_deg[gt_coord_idx] if gt_coord_idx.size > 0 else np.empty((0, 2), dtype=float)
    pred_coords = coords_deg[pred_coord_idx] if pred_coord_idx.size > 0 else np.empty((0, 2), dtype=float)
    gt_labels = _cluster_event_points(gt_coords, eps_m=float(match_buffer_m))
    pred_labels = _cluster_event_points(pred_coords, eps_m=float(match_buffer_m))

    gt_events = int(np.unique(gt_labels).size) if gt_labels.size > 0 else 0
    pred_events = int(np.unique(pred_labels).size) if pred_labels.size > 0 else 0

    gt_point_matched = np.zeros(len(gt_coords), dtype=bool)
    pred_point_matched = np.zeros(len(pred_coords), dtype=bool)
    if len(gt_coords) > 0 and len(pred_coords) > 0:
        eps_rad = float(match_buffer_m) / _EARTH_RADIUS_M
        pred_tree = BallTree(np.radians(pred_coords), metric="haversine")
        gt_tree = BallTree(np.radians(gt_coords), metric="haversine")
        gt_point_matched = pred_tree.query_radius(np.radians(gt_coords), r=eps_rad, count_only=True) > 0
        pred_point_matched = gt_tree.query_radius(np.radians(pred_coords), r=eps_rad, count_only=True) > 0

    detected_gt_events = 0
    if gt_labels.size > 0:
        for lab in np.unique(gt_labels):
            if bool(gt_point_matched[gt_labels == lab].any()):
                detected_gt_events += 1

    matched_pred_events = 0
    if pred_labels.size > 0:
        for lab in np.unique(pred_labels):
            if bool(pred_point_matched[pred_labels == lab].any()):
                matched_pred_events += 1

    event_recall = (
        float(detected_gt_events / gt_events)
        if gt_events > 0
        else None
    )
    if pred_events > 0:
        event_precision = float(matched_pred_events / pred_events)
    else:
        event_precision = 0.0 if gt_events > 0 else None

    if event_recall is None or event_precision is None or (event_recall + event_precision) == 0.0:
        event_f1 = None if event_recall is None or event_precision is None else 0.0
    else:
        event_f1 = float(2.0 * event_precision * event_recall / (event_precision + event_recall))

    tp_events = int(detected_gt_events)
    fn_events = int(max(0, gt_events - detected_gt_events))
    fp_events = int(max(0, pred_events - matched_pred_events))

    return {
        "threshold": float(threshold),
        "match_buffer_m": float(match_buffer_m),
        "coordinate_basis": coord_basis,
        "event_recall": event_recall,
        "event_precision": event_precision,
        "event_f1": event_f1,
        "ground_truth_events": int(gt_events),
        "predicted_events": int(pred_events),
        "detected_ground_truth_events": int(detected_gt_events),
        "matched_predicted_events": int(matched_pred_events),
        "tp_events": tp_events,
        "fp_events": fp_events,
        "fn_events": fn_events,
        "usable_point_rows": int(coords_deg.shape[0]),
        "total_rows": int(len(eval_known)),
        "unusable_rows_missing_coords": int(len(eval_known) - coords_deg.shape[0]),
    }


def _slice_metrics_or_empty(y: np.ndarray, p: np.ndarray, mask: np.ndarray, *, threshold: float) -> Dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    n = int(selected.sum())
    if n == 0:
        return {
            "n": 0,
            "threshold": float(threshold),
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "predicted_positive_rate": None,
        }
    return {"n": n, **_metrics(y[selected], p[selected], threshold=threshold)}


def _required_slice_metrics(
    *,
    eval_known: pd.DataFrame,
    y: np.ndarray,
    calibrated: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    biome_text = (
        eval_known.get("biome_slice", pd.Series(["unknown"] * len(eval_known)))
        .astype(str)
        .str.lower()
    )
    lat_col = None
    for c in ("lat_centroid", "lat"):
        if c in eval_known.columns:
            lat_col = c
            break
    if lat_col is not None:
        lat_abs = pd.to_numeric(eval_known[lat_col], errors="coerce").abs()
        boreal_mask = lat_abs >= 50.0
        tropical_mask = lat_abs <= 23.5
        biome_basis = f"latitude:{lat_col}"
    else:
        boreal_mask = biome_text.str.contains("boreal", na=False)
        tropical_mask = biome_text.str.contains("tropical", na=False)
        biome_basis = "biome_slice_label"

    out["biome_boreal_vs_tropical"] = {
        "basis": biome_basis,
        "boreal": _slice_metrics_or_empty(y, calibrated, boreal_mask.to_numpy(), threshold=threshold),
        "tropical": _slice_metrics_or_empty(y, calibrated, tropical_mask.to_numpy(), threshold=threshold),
    }

    if "is_day_ratio" in eval_known.columns:
        day_ratio = pd.to_numeric(eval_known["is_day_ratio"], errors="coerce")
        day_mask = (day_ratio >= 0.5).fillna(False)
        night_mask = (day_ratio < 0.5).fillna(False)
        tod_basis = "is_day_ratio>=0.5"
    elif "hour_of_day" in eval_known.columns:
        hour = pd.to_numeric(eval_known["hour_of_day"], errors="coerce")
        day_mask = ((hour >= 6.0) & (hour < 18.0)).fillna(False)
        night_mask = ~day_mask
        tod_basis = "hour_of_day[6,18)"
    else:
        day_mask = pd.Series([False] * len(eval_known))
        night_mask = pd.Series([False] * len(eval_known))
        tod_basis = "unavailable"

    out["time_of_day_day_vs_night"] = {
        "basis": tod_basis,
        "day": _slice_metrics_or_empty(y, calibrated, day_mask.to_numpy(), threshold=threshold),
        "night": _slice_metrics_or_empty(y, calibrated, night_mask.to_numpy(), threshold=threshold),
    }

    scan_col = None
    for c in ("scan_angle_max", "scan_angle_mean", "scan_angle"):
        if c in eval_known.columns:
            scan_col = c
            break
    if scan_col is not None:
        scan = pd.to_numeric(eval_known[scan_col], errors="coerce")
        scan_mask = (scan > 45.0).fillna(False)
        scan_basis = f"{scan_col}>45"
    else:
        scan_mask = pd.Series([False] * len(eval_known))
        scan_basis = "unavailable"

    out["scan_angle_gt_45"] = {
        "basis": scan_basis,
        "gt_45": _slice_metrics_or_empty(y, calibrated, scan_mask.to_numpy(), threshold=threshold),
    }

    return out


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


def _industrial_suppression_mask(
    eval_known: pd.DataFrame,
    *,
    policy_version: str | None,
    strict_no_go: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    lat_col = next((c for c in ("lat_centroid", "lat") if c in eval_known.columns), None)
    lon_col = next((c for c in ("lon_centroid", "lon") if c in eval_known.columns), None)
    time_col = next((c for c in ("start_time", "acq_time") if c in eval_known.columns), None)
    n = int(len(eval_known))
    if lat_col is None or lon_col is None:
        return np.zeros(n, dtype=bool), {
            "applied": False,
            "reason": "missing_coordinates",
            "masked_rows": 0,
            "total_rows": n,
        }

    lat = pd.to_numeric(eval_known[lat_col], errors="coerce")
    lon = pd.to_numeric(eval_known[lon_col], errors="coerce")
    valid = lat.notna() & lon.notna()
    if not bool(valid.any()):
        return np.zeros(n, dtype=bool), {
            "applied": False,
            "reason": "no_valid_coordinates",
            "masked_rows": 0,
            "total_rows": n,
        }

    if time_col is not None:
        acq = pd.to_datetime(eval_known[time_col], utc=True, errors="coerce")
    else:
        acq = pd.Series([pd.NaT] * n, index=eval_known.index)

    payload: list[dict[str, Any]] = []
    for i in np.flatnonzero(valid.to_numpy(dtype=bool)):
        ts = acq.iloc[i]
        payload.append(
            {
                "row_idx": int(i),
                "lon": float(lon.iloc[i]),
                "lat": float(lat.iloc[i]),
                "acq_time": ts.to_pydatetime() if pd.notna(ts) else None,
            }
        )

    mask = np.zeros(n, dtype=bool)
    meters_to_deg = 1.0 / 111000.0
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                CREATE TEMP TABLE tmp_eval_industrial_points (
                    row_idx integer PRIMARY KEY,
                    acq_time timestamptz,
                    geom geometry(Point, 4326)
                ) ON COMMIT DROP
                """
            )
        )
        insert_stmt = text(
            """
            INSERT INTO tmp_eval_industrial_points (row_idx, acq_time, geom)
            VALUES (
                :row_idx,
                :acq_time,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
            )
            """
        )
        conn.execute(insert_stmt, payload)
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT p.row_idx
                FROM tmp_eval_industrial_points p
                JOIN industrial_sources i
                  ON COALESCE(i.is_active, TRUE)
                 AND i.authority_tier IN ('gold', 'silver')
                 AND (p.acq_time IS NULL OR i.valid_from IS NULL OR i.valid_from <= p.acq_time)
                 AND (p.acq_time IS NULL OR i.valid_to IS NULL OR i.valid_to >= p.acq_time)
                 AND i.geom && ST_Expand(
                        p.geom,
                        CASE
                            WHEN i.authority_tier = 'gold' THEN :gold_buffer_deg
                            ELSE :silver_buffer_deg
                        END
                    )
                 AND ST_DWithin(
                        p.geom::geography,
                        i.geom::geography,
                        CASE
                            WHEN i.authority_tier = 'gold' THEN :gold_buffer_m
                            ELSE :silver_buffer_m
                        END
                    )
                WHERE NOT (
                    :strict_no_go
                    AND :policy_version IS NOT NULL
                    AND EXISTS (
                        SELECT 1
                        FROM industrial_no_go_zones z
                        WHERE z.is_active
                          AND z.policy_version = :policy_version
                          AND z.geom && p.geom
                          AND ST_Intersects(z.geom, p.geom)
                    )
                )
                """
            ),
            {
                "gold_buffer_m": float(_INDUSTRIAL_GOLD_BUFFER_M),
                "silver_buffer_m": float(_INDUSTRIAL_SILVER_BUFFER_M),
                "gold_buffer_deg": float(_INDUSTRIAL_GOLD_BUFFER_M) * meters_to_deg,
                "silver_buffer_deg": float(_INDUSTRIAL_SILVER_BUFFER_M) * meters_to_deg,
                "strict_no_go": bool(strict_no_go),
                "policy_version": policy_version,
            },
        ).mappings().all()

    for row in rows:
        idx = int(row["row_idx"])
        if 0 <= idx < n:
            mask[idx] = True

    return mask, {
        "applied": True,
        "policy_version": policy_version,
        "strict_no_go": bool(strict_no_go),
        "gold_buffer_m": float(_INDUSTRIAL_GOLD_BUFFER_M),
        "silver_buffer_m": float(_INDUSTRIAL_SILVER_BUFFER_M),
        "masked_rows": int(mask.sum()),
        "total_rows": n,
        "coordinate_columns": f"{lat_col},{lon_col}",
        "time_column": time_col,
    }


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
    event_level = _event_level_metrics(
        eval_known=eval_known,
        y_true=y,
        probabilities=calibrated,
        threshold=default_threshold,
        match_buffer_m=_EVENT_MATCH_BUFFER_M,
    )
    event_recall_value = event_level.get("event_recall")
    event_precision_value = event_level.get("event_precision")
    if event_recall_value is None:
        event_recall_value = float(default_metrics["recall"])
    if event_precision_value is None:
        event_precision_value = float(default_metrics["precision"])

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    gate_results = {
        "event_recall": {
            "value": float(event_recall_value),
            "pass": float(event_recall_value) >= gate_thresholds["event_recall_min"],
        },
        "event_precision": {
            "value": float(event_precision_value),
            "pass": float(event_precision_value) >= gate_thresholds["event_precision_min"],
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
    slice_metrics["required_slices"] = _required_slice_metrics(
        eval_known=eval_known,
        y=y,
        calibrated=calibrated,
        threshold=default_threshold,
    )

    return {
        "scope": scope_name,
        "n_eval": int(len(eval_known)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "default_metrics": default_metrics,
        "event_level_metrics": event_level,
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
            + [
                "event_label",
                "sensor",
                "biome_slice",
                "is_day_ratio",
                "hour_of_day",
                "scan_angle",
                "scan_angle_mean",
                "scan_angle_max",
                "lat",
                "lat_centroid",
                "lon",
                "lon_centroid",
                "frp_max",
                "frp_mean",
                "start_time",
                "truth_covered_mask",
                "coverage_mask_ids",
            ]
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

    industrial_policy = _load_industrial_policy_provenance(policy_version=industrial_policy_version)
    policy_version_for_mask = (
        str(industrial_policy.get("policy_version"))
        if industrial_policy is not None and industrial_policy.get("policy_version") is not None
        else industrial_policy_version
    )
    industrial_mask, industrial_mask_summary = _industrial_suppression_mask(
        eval_known,
        policy_version=policy_version_for_mask,
        strict_no_go=bool((industrial_policy or {}).get("strict_no_go", False)),
    )
    frp_col = next((c for c in ("frp_max", "frp_mean", "frp") if c in eval_known.columns), None)
    if frp_col is not None:
        frp_vals = pd.to_numeric(eval_known[frp_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        frp_vals = np.zeros(len(eval_known), dtype=float)
    fail_closed_override_mask = frp_vals > float(_FAIL_CLOSED_FRP_MW)
    industrial_suppression_mask = industrial_mask & ~fail_closed_override_mask
    calibrated = np.asarray(calibrated, dtype=float).copy()
    calibrated[industrial_suppression_mask] = 0.0

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
        "promotion_event_recall_min": 0.92,
        "promotion_global_f1_min": 0.85,
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

    operational_gate_pass = bool(primary_eval["gate_pass"])
    default_metrics = dict(primary_eval["default_metrics"])
    promotion_event_recall = primary_eval.get("event_level_metrics", {}).get("event_recall")
    if promotion_event_recall is None:
        promotion_event_recall = float(default_metrics["recall"])
    promotion_gate_results = {
        "event_recall": {
            "value": float(promotion_event_recall),
            "threshold": float(gate_thresholds["promotion_event_recall_min"]),
            "pass": float(promotion_event_recall) > float(gate_thresholds["promotion_event_recall_min"]),
        },
        "global_f1": {
            "value": float(default_metrics["f1"]),
            "threshold": float(gate_thresholds["promotion_global_f1_min"]),
            "pass": float(default_metrics["f1"]) > float(gate_thresholds["promotion_global_f1_min"]),
        },
    }
    gate_pass = all(bool(x["pass"]) for x in promotion_gate_results.values())

    os.makedirs(out_dir, exist_ok=True)

    global_eval["sweep"].to_csv(os.path.join(out_dir, "threshold_sweep_global.csv"), index=False)
    if covered_eval is not None:
        covered_eval["sweep"].to_csv(os.path.join(out_dir, "threshold_sweep_covered.csv"), index=False)
    primary_eval["sweep"].to_csv(os.path.join(out_dir, "threshold_sweep.csv"), index=False)

    coverage_mask_ids = []
    if "coverage_mask_ids" in eval_known.columns:
        coverage_mask_ids = _collect_mask_ids(eval_known["coverage_mask_ids"])
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
        "industrial_suppression": {
            **industrial_mask_summary,
            "fail_closed_override_frp_mw": float(_FAIL_CLOSED_FRP_MW),
            "fail_closed_override_rows": int(fail_closed_override_mask.sum()),
            "suppressed_rows": int(industrial_suppression_mask.sum()),
        },
        "n_eval": int(primary_eval["n_eval"]),
        "n_pos": int(primary_eval["n_pos"]),
        "n_neg": int(primary_eval["n_neg"]),
        "default_threshold": default_threshold,
        "default_metrics": primary_eval["default_metrics"],
        "event_level_metrics": primary_eval["event_level_metrics"],
        "threshold_recommendations": {
            "strong_filter": primary_eval["threshold_recommendations"]["strong_filter"],
            "downweight": primary_eval["threshold_recommendations"]["downweight"],
            "uncertainty_band_low": float(bundle.get("thresholds", {}).get("uncertainty_band_low", 0.45)),
            "uncertainty_band_high": float(bundle.get("thresholds", {}).get("uncertainty_band_high", 0.55)),
        },
        "gate_thresholds": gate_thresholds,
        "gate_results": primary_eval["gate_results"],
        "promotion_gate_results": promotion_gate_results,
        "gate_pass": gate_pass,
        "operational_gate_pass": operational_gate_pass,
        "global": {
            "n_eval": int(global_eval["n_eval"]),
            "n_pos": int(global_eval["n_pos"]),
            "n_neg": int(global_eval["n_neg"]),
            "default_metrics": global_eval["default_metrics"],
            "event_level_metrics": global_eval["event_level_metrics"],
            "gate_results": global_eval["gate_results"],
            "promotion_gate_results": {
                "event_recall": {
                    "value": float(
                        (
                            global_eval.get("event_level_metrics", {}).get("event_recall")
                            if global_eval.get("event_level_metrics", {}).get("event_recall") is not None
                            else global_eval["default_metrics"]["recall"]
                        )
                    ),
                    "threshold": float(gate_thresholds["promotion_event_recall_min"]),
                    "pass": float(
                        (
                            global_eval.get("event_level_metrics", {}).get("event_recall")
                            if global_eval.get("event_level_metrics", {}).get("event_recall") is not None
                            else global_eval["default_metrics"]["recall"]
                        )
                    )
                    > float(gate_thresholds["promotion_event_recall_min"]),
                },
                "global_f1": {
                    "value": float(global_eval["default_metrics"]["f1"]),
                    "threshold": float(gate_thresholds["promotion_global_f1_min"]),
                    "pass": float(global_eval["default_metrics"]["f1"])
                    > float(gate_thresholds["promotion_global_f1_min"]),
                },
            },
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
            "event_level_metrics": covered_eval["event_level_metrics"],
            "gate_results": covered_eval["gate_results"],
            "promotion_gate_results": {
                "event_recall": {
                    "value": float(
                        (
                            covered_eval.get("event_level_metrics", {}).get("event_recall")
                            if covered_eval.get("event_level_metrics", {}).get("event_recall") is not None
                            else covered_eval["default_metrics"]["recall"]
                        )
                    ),
                    "threshold": float(gate_thresholds["promotion_event_recall_min"]),
                    "pass": float(
                        (
                            covered_eval.get("event_level_metrics", {}).get("event_recall")
                            if covered_eval.get("event_level_metrics", {}).get("event_recall") is not None
                            else covered_eval["default_metrics"]["recall"]
                        )
                    )
                    > float(gate_thresholds["promotion_event_recall_min"]),
                },
                "global_f1": {
                    "value": float(covered_eval["default_metrics"]["f1"]),
                    "threshold": float(gate_thresholds["promotion_global_f1_min"]),
                    "pass": float(covered_eval["default_metrics"]["f1"])
                    > float(gate_thresholds["promotion_global_f1_min"]),
                },
            },
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
        "industrial_suppression": summary.get("industrial_suppression"),
        "thresholds": gate_thresholds,
        "results": primary_eval["gate_results"],
        "event_level_metrics": primary_eval.get("event_level_metrics"),
        "promotion_results": promotion_gate_results,
        "global_results": global_eval["gate_results"],
        "global_event_level_metrics": global_eval.get("event_level_metrics"),
        "covered_results": covered_eval["gate_results"] if covered_eval is not None else None,
        "covered_event_level_metrics": covered_eval.get("event_level_metrics") if covered_eval is not None else None,
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

    if not gate_pass:
        raise ModelPromotionError(
            "Promotion gate failed: "
            f"event_recall={promotion_gate_results['event_recall']['value']:.6f} "
            f"(required > {promotion_gate_results['event_recall']['threshold']:.2f}), "
            f"global_f1={promotion_gate_results['global_f1']['value']:.6f} "
            f"(required > {promotion_gate_results['global_f1']['threshold']:.2f})."
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
