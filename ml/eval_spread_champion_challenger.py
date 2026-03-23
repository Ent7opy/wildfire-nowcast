"""Evaluate champion vs challenger spread models on identical reference cases.

Produces per-horizon comparison metrics and a promotion recommendation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score

from api.db import get_engine
from api.fires.service import get_fire_cells_heatmap, get_region_grid_spec
from ml.spread.factory import get_spread_model, normalize_model_selection
from ml.spread.hindcast_dataset import sample_fire_reference_times
from ml.spread.runtime_contract import (
    CANONICAL_CHANNELS_BY_MODEL,
    ContractViolationError,
    load_contract,
    validate_channel_alignment,
)
from ml.spread_features import assert_grid_alignment, build_spread_inputs

LOGGER = logging.getLogger(__name__)

# Model families that accept a calibrator and require freshness validation at gate time.
_CAL_FRESHNESS_MODELS: frozenset[str] = frozenset({"LearnedSpreadModelV2", "LearnedSpreadModelV3"})


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) for binary outcomes."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]
    if y_true.size == 0:
        return float("nan")

    y_true = (y_true > 0.5).astype(float)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)

    ece = 0.0
    n = float(y_true.size)
    for b in range(int(n_bins)):
        mask = idx == b
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        w = float(np.sum(mask)) / n
        ece += w * abs(acc - conf)
    return float(ece)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true_b = (np.asarray(y_true) > 0.5).astype(int)
    y_pred = (np.asarray(y_prob) >= float(threshold)).astype(int)

    tp = int(np.sum((y_true_b == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true_b == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true_b == 1) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def _safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    y_true = (np.asarray(y_true) > 0.5).astype(int)
    y_prob = np.asarray(y_prob)
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def _brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    y_true = (np.asarray(y_true) > 0.5).astype(np.float32)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float32), 0.0, 1.0)
    brier = float(np.mean((y_prob - y_true) ** 2))
    climatology = float(np.mean(y_true))
    brier_ref = float(np.mean((climatology - y_true) ** 2))
    if brier_ref <= 1e-12:
        return brier, 0.0
    return brier, float(1.0 - (brier / brier_ref))


def _sal_from_cases(y_true_cases: list[np.ndarray], y_pred_cases: list[np.ndarray]) -> dict[str, float]:
    if not y_true_cases:
        return {"S": float("nan"), "A": float("nan"), "L": float("nan"), "composite": float("nan")}

    # Try pysteps if available. Fallback to a deterministic approximation.
    try:
        from pysteps.verification.salscores import sal as sal_score  # type: ignore

        triples = []
        for obs, pred in zip(y_true_cases, y_pred_cases):
            s, a, loc = sal_score(np.asarray(pred, dtype=float), np.asarray(obs, dtype=float))
            triples.append((float(abs(s)), float(abs(a)), float(abs(loc))))
        arr = np.asarray(triples, dtype=float)
        return {
            "S": float(np.nanmean(arr[:, 0])),
            "A": float(np.nanmean(arr[:, 1])),
            "L": float(np.nanmean(arr[:, 2])),
            "composite": float(np.nanmean(np.sum(arr, axis=1) / 3.0)),
        }
    except Exception:
        pass

    def _centroid(grid: np.ndarray) -> tuple[float, float]:
        g = np.asarray(grid, dtype=float)
        w = np.clip(g, 0.0, None)
        if np.sum(w) <= 1e-12:
            h, w_ = g.shape
            return (h / 2.0, w_ / 2.0)
        ys, xs = np.indices(g.shape)
        return (float(np.sum(ys * w) / np.sum(w)), float(np.sum(xs * w) / np.sum(w)))

    parts = []
    for obs, pred in zip(y_true_cases, y_pred_cases):
        obs_f = np.asarray(obs, dtype=float)
        pred_f = np.asarray(pred, dtype=float)
        s = abs(np.nanstd(pred_f) - np.nanstd(obs_f)) / (np.nanstd(obs_f) + 1e-6)
        a = abs(np.nansum(pred_f) - np.nansum(obs_f)) / (np.nansum(obs_f) + 1.0)
        cy_o, cx_o = _centroid(obs_f)
        cy_p, cx_p = _centroid(pred_f)
        norm = math.sqrt(float(obs_f.shape[0] ** 2 + obs_f.shape[1] ** 2)) + 1e-6
        loc = float(math.sqrt((cy_o - cy_p) ** 2 + (cx_o - cx_p) ** 2) / norm)
        parts.append((float(s), float(a), float(loc)))

    arr = np.asarray(parts, dtype=float)
    return {
        "S": float(np.nanmean(arr[:, 0])),
        "A": float(np.nanmean(arr[:, 1])),
        "L": float(np.nanmean(arr[:, 2])),
        "composite": float(np.nanmean(np.sum(arr, axis=1) / 3.0)),
    }


def _diebold_mariano_pvalue(
    y_true_cases: list[np.ndarray],
    champion_cases: list[np.ndarray],
    challenger_cases: list[np.ndarray],
) -> dict[str, float | None]:
    if not y_true_cases:
        return {"dm_stat": None, "dm_p_value": None}

    diffs = []
    for obs, champ, chall in zip(y_true_cases, champion_cases, challenger_cases):
        obs_b = (np.asarray(obs, dtype=np.float32) > 0.5).astype(np.float32)
        champ_p = np.clip(np.asarray(champ, dtype=np.float32), 0.0, 1.0)
        chall_p = np.clip(np.asarray(chall, dtype=np.float32), 0.0, 1.0)
        champ_loss = float(np.mean((champ_p - obs_b) ** 2))
        chall_loss = float(np.mean((chall_p - obs_b) ** 2))
        # Positive means challenger lower loss (better).
        diffs.append(champ_loss - chall_loss)

    d = np.asarray(diffs, dtype=float)
    if d.size < 2 or np.nanstd(d, ddof=1) < 1e-12:
        return {"dm_stat": None, "dm_p_value": None}

    dm_stat = float(np.nanmean(d) / (np.nanstd(d, ddof=1) / np.sqrt(d.size)))
    dm_p = float(2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=max(1, d.size - 1))))
    return {"dm_stat": dm_stat, "dm_p_value": dm_p}


def summarize_comparison_for_horizon(
    *,
    horizon_hours: int,
    y_true: np.ndarray,
    y_prob_champion: np.ndarray,
    y_prob_challenger: np.ndarray,
    y_true_cases: list[np.ndarray] | None = None,
    champion_cases: list[np.ndarray] | None = None,
    challenger_cases: list[np.ndarray] | None = None,
    ece_bins: int = 10,
) -> dict[str, Any]:
    y_true = (np.asarray(y_true) > 0.5).astype(np.float32, copy=False)
    p_champion = np.clip(np.asarray(y_prob_champion, dtype=np.float32), 0.0, 1.0)
    p_challenger = np.clip(np.asarray(y_prob_challenger, dtype=np.float32), 0.0, 1.0)

    champion_brier, champion_bss = _brier_skill_score(y_true, p_champion)
    challenger_brier, challenger_bss = _brier_skill_score(y_true, p_challenger)

    champion_ece = expected_calibration_error(y_true, p_champion, n_bins=ece_bins)
    challenger_ece = expected_calibration_error(y_true, p_challenger, n_bins=ece_bins)

    champion_pr_auc = _safe_pr_auc(y_true, p_champion)
    challenger_pr_auc = _safe_pr_auc(y_true, p_challenger)

    champion_iou_03 = _binary_metrics(y_true, p_champion, 0.3)["iou"]
    challenger_iou_03 = _binary_metrics(y_true, p_challenger, 0.3)["iou"]
    champion_iou_05 = _binary_metrics(y_true, p_champion, 0.5)["iou"]
    challenger_iou_05 = _binary_metrics(y_true, p_challenger, 0.5)["iou"]

    true_cases = y_true_cases or []
    champ_cases = champion_cases or []
    chall_cases = challenger_cases or []
    sal_champion = _sal_from_cases(true_cases, champ_cases) if true_cases else {"S": float("nan"), "A": float("nan"), "L": float("nan"), "composite": float("nan")}
    sal_challenger = _sal_from_cases(true_cases, chall_cases) if true_cases else {"S": float("nan"), "A": float("nan"), "L": float("nan"), "composite": float("nan")}
    dm = _diebold_mariano_pvalue(true_cases, champ_cases, chall_cases) if true_cases else {"dm_stat": None, "dm_p_value": None}

    return {
        "horizon_hours": int(horizon_hours),
        "n": int(y_true.size),
        "champion_brier": champion_brier,
        "challenger_brier": challenger_brier,
        "brier_improvement": champion_brier - challenger_brier,
        "champion_bss": float(champion_bss),
        "challenger_bss": float(challenger_bss),
        "bss_improvement": float(challenger_bss - champion_bss),
        "champion_ece": float(champion_ece),
        "challenger_ece": float(challenger_ece),
        "ece_improvement": float(champion_ece - challenger_ece),
        "champion_pr_auc": champion_pr_auc,
        "challenger_pr_auc": challenger_pr_auc,
        "pr_auc_improvement": (
            None
            if champion_pr_auc is None or challenger_pr_auc is None
            else float(challenger_pr_auc - champion_pr_auc)
        ),
        "champion_iou_03": champion_iou_03,
        "challenger_iou_03": challenger_iou_03,
        "iou_03_improvement": float(challenger_iou_03 - champion_iou_03),
        "champion_iou_05": champion_iou_05,
        "challenger_iou_05": challenger_iou_05,
        "iou_05_improvement": float(challenger_iou_05 - champion_iou_05),
        "champion_sal_S": float(sal_champion.get("S", float("nan"))),
        "champion_sal_A": float(sal_champion.get("A", float("nan"))),
        "champion_sal_L": float(sal_champion.get("L", float("nan"))),
        "champion_sal_composite": float(sal_champion.get("composite", float("nan"))),
        "challenger_sal_S": float(sal_challenger.get("S", float("nan"))),
        "challenger_sal_A": float(sal_challenger.get("A", float("nan"))),
        "challenger_sal_L": float(sal_challenger.get("L", float("nan"))),
        "challenger_sal_composite": float(sal_challenger.get("composite", float("nan"))),
        "sal_composite_improvement": float(
            sal_champion.get("composite", float("nan")) - sal_challenger.get("composite", float("nan"))
        ),
        "dm_stat": dm.get("dm_stat"),
        "dm_p_value": dm.get("dm_p_value"),
    }


def compute_recommendation(
    summary_rows: list[dict[str, Any]],
    *,
    bss_improvement_min: float = 0.03,
    bss_horizon_floor: float = -0.005,
    sal_regression_max: float = 0.05,
    dm_pvalue_max: float = 0.05,
    max_pr_auc_drop: float = 0.01,
    max_iou_drop: float = 0.02,
) -> dict[str, Any]:
    """Moderate gate for champion/challenger recommendation."""
    if not summary_rows:
        return {
            "recommend_challenger": False,
            "pass": False,
            "reasons": ["No summary rows available."],
        }

    reasons: list[str] = []
    weights = np.asarray([max(1, int(r.get("n", 1))) for r in summary_rows], dtype=float)
    bss_vals = np.asarray([float(r.get("bss_improvement", 0.0)) for r in summary_rows], dtype=float)
    weighted_bss = float(np.average(bss_vals, weights=weights))

    primary_ok = True
    secondary_ok = True

    if weighted_bss < bss_improvement_min:
        primary_ok = False
        reasons.append(
            f"Weighted BSS improvement {weighted_bss:.4f} is below threshold {bss_improvement_min:.4f}."
        )

    sal_improved_count = 0
    for row in summary_rows:
        h = int(row.get("horizon_hours", -1))
        bss_h = float(row.get("bss_improvement", 0.0))
        if bss_h < bss_horizon_floor:
            primary_ok = False
            reasons.append(
                f"T+{h}h: BSS regression {bss_h:.4f} breaches floor {bss_horizon_floor:.4f}."
            )

        sal_imp = float(row.get("sal_composite_improvement", float("nan")))
        if np.isfinite(sal_imp):
            if sal_imp > 0:
                sal_improved_count += 1
            if sal_imp < -abs(sal_regression_max):
                primary_ok = False
                reasons.append(
                    f"T+{h}h: SAL composite regression {sal_imp:.4f} exceeds {sal_regression_max:.4f}."
                )

        dm_p = row.get("dm_p_value")
        if dm_p is not None and np.isfinite(float(dm_p)) and float(dm_p) > dm_pvalue_max:
            primary_ok = False
            reasons.append(
                f"T+{h}h: Diebold-Mariano p-value {float(dm_p):.4f} exceeds {dm_pvalue_max:.4f}."
            )

        pr_auc_improvement = row.get("pr_auc_improvement")
        if pr_auc_improvement is not None and float(pr_auc_improvement) < -abs(max_pr_auc_drop):
            secondary_ok = False
            reasons.append(
                f"T+{h}h: PR-AUC regression exceeds threshold ({float(pr_auc_improvement):.4f})."
            )

        for key in ("iou_03_improvement", "iou_05_improvement"):
            if float(row.get(key, 0.0)) < -abs(max_iou_drop):
                secondary_ok = False
                reasons.append(
                    f"T+{h}h: {key} regression exceeds threshold ({float(row[key]):.4f})."
                )

    min_sal_improved = max(1, int(math.ceil((2.0 / 3.0) * len(summary_rows))))
    if sal_improved_count < min_sal_improved:
        primary_ok = False
        reasons.append(
            f"SAL improvement only on {sal_improved_count}/{len(summary_rows)} horizons (requires >= {min_sal_improved})."
        )

    recommend = bool(primary_ok and secondary_ok)
    if recommend:
        reasons.append("Challenger passed BSS/SAL/DM primary gates and PR-AUC/IoU guardrails.")

    return {
        "recommend_challenger": recommend,
        "pass": recommend,
        "primary_ok": bool(primary_ok),
        "secondary_ok": bool(secondary_ok),
        "weighted_bss_improvement": weighted_bss,
        "bss_improvement_min": float(bss_improvement_min),
        "bss_horizon_floor": float(bss_horizon_floor),
        "sal_regression_max": float(sal_regression_max),
        "dm_pvalue_max": float(dm_pvalue_max),
        "max_pr_auc_drop": float(max_pr_auc_drop),
        "max_iou_drop": float(max_iou_drop),
        "reasons": reasons,
    }


def _calibrator_artifact_present(model_params: dict[str, Any] | None) -> bool:
    params = dict(model_params or {})
    raw = params.get("calibrator_run_dir")
    if raw is None:
        return False
    path = Path(str(raw))
    if path.is_file():
        return path.name == "calibrator.pkl"
    if path.is_dir():
        return (path / "calibrator.pkl").exists()
    return False


def _read_metadata_created_at(run_dir: Path) -> datetime | None:
    """Return the ``created_at`` timestamp from *run_dir*/metadata.json, or None.

    Both model training runs (``train_spread_v2.py``) and calibration runs
    (``ml/calibration.py``) write a ``metadata.json`` that includes a
    ``created_at`` ISO-8601 string.  Returns a timezone-aware UTC datetime, or
    ``None`` if the file is absent, malformed, or the field is missing.
    """
    try:
        payload = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        raw = payload.get("created_at")
        if not raw:
            return None
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None


def _check_calibrator_freshness(
    model_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare calibrator training date against model training date.

    Returns a dict with keys:
      - ``stale``: True when the calibrator predates the model
      - ``model_created_at``: ISO string or None
      - ``calibrator_created_at``: ISO string or None
      - ``reason``: human-readable explanation (non-empty only when stale or
        when dates could not be read)
      - ``unreadable``: True when one or both dates could not be determined
    """
    params = dict(model_params or {})
    model_run_dir_raw = params.get("model_run_dir")
    cal_run_dir_raw = params.get("calibrator_run_dir")

    result: dict[str, Any] = {
        "stale": False,
        "unreadable": False,
        "model_created_at": None,
        "calibrator_created_at": None,
        "reason": "",
    }

    if not model_run_dir_raw or not cal_run_dir_raw:
        return result  # absence handled by STOP-CAL-001 / STOP-CONTRACT-001

    model_dir = Path(str(model_run_dir_raw))
    cal_dir = Path(str(cal_run_dir_raw))
    if cal_dir.is_file():
        cal_dir = cal_dir.parent

    model_ts = _read_metadata_created_at(model_dir)
    cal_ts = _read_metadata_created_at(cal_dir)

    result["model_created_at"] = model_ts.isoformat() if model_ts else None
    result["calibrator_created_at"] = cal_ts.isoformat() if cal_ts else None

    if model_ts is None or cal_ts is None:
        missing = []
        if model_ts is None:
            missing.append(f"model ({model_dir / 'metadata.json'})")
        if cal_ts is None:
            missing.append(f"calibrator ({cal_dir / 'metadata.json'})")
        result["unreadable"] = True
        result["reason"] = (
            "Cannot verify calibrator freshness: metadata.json missing or unreadable for "
            + " and ".join(missing)
            + "."
        )
        return result

    if cal_ts < model_ts:
        result["stale"] = True
        result["reason"] = (
            f"Calibrator was trained at {cal_ts.isoformat()} but the model was trained at "
            f"{model_ts.isoformat()}. The calibrator predates the model — it was fitted on "
            "a different (older) model's outputs and its probability mappings are invalid "
            "for this model."
        )

    return result


def _build_stage_governance(
    *,
    config: dict[str, Any],
    decision: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    geo_alignment_error: str | None = None,
) -> dict[str, Any]:
    gate_cfg = dict(config.get("gate", {}) or {})
    maturity_stage = str(gate_cfg.get("maturity_stage", "mvp_operational"))
    valid_stages = {"mvp_operational", "science_grade"}

    hard_stops: list[dict[str, Any]] = []
    stage_warnings: list[dict[str, Any]] = []
    science_debt_register: list[dict[str, Any]] = []

    if maturity_stage not in valid_stages:
        hard_stops.append(
            {
                "id": "STOP-STAGE-001",
                "message": f"Unsupported maturity_stage={maturity_stage!r}.",
                "mitigation": "Use one of: mvp_operational, science_grade.",
                "target_stage": "mvp_operational",
            }
        )

    # STOP-GEO-001: the eval region grid must pass the canonical analysis-grid
    # contract (CRS=EPSG:4326, cell_size=0.01°).  A mismatched CRS or resolution
    # means train-time features were computed at a different spatial frame than
    # the eval inputs, making metric comparisons scientifically invalid.
    if geo_alignment_error is not None:
        hard_stops.append(
            {
                "id": "STOP-GEO-001",
                "message": geo_alignment_error,
                "mitigation": (
                    "Re-project the eval region grid to EPSG:4326 at 0.01° cell size "
                    "(DEFAULT_CRS / DEFAULT_CELL_SIZE_DEG) before running champion-challenger eval."
                ),
                "target_stage": maturity_stage,
            }
        )

    # STOP-SOURCE-001: every eval config must declare authoritative sources for all
    # four spread input categories.  Per docs/spread_data_sources.md §5, absence of
    # any required key is a hard stop — undeclared provenance cannot be promoted.
    _REQUIRED_SOURCE_KEYS = ("fires", "weather", "terrain", "fuels")
    declared_sources = dict(config.get("data_sources") or {})
    missing_source_keys = [k for k in _REQUIRED_SOURCE_KEYS if not declared_sources.get(k)]
    if missing_source_keys:
        hard_stops.append(
            {
                "id": "STOP-SOURCE-001",
                "message": (
                    "data_sources declaration is missing required input source(s): "
                    + ", ".join(missing_source_keys)
                    + ". See docs/spread_data_sources.md §5 for required keys and example values."
                ),
                "mitigation": (
                    "Add a data_sources block to the eval config with keys: "
                    + ", ".join(_REQUIRED_SOURCE_KEYS)
                    + "."
                ),
                "target_stage": maturity_stage,
            }
        )

    challenger_cfg = dict(config.get("challenger", {}) or {})
    challenger_name = challenger_cfg.get("model_name")
    challenger_params = challenger_cfg.get("model_params")
    if challenger_name == "LearnedSpreadModelV3" and not _calibrator_artifact_present(challenger_params):
        hard_stops.append(
            {
                "id": "STOP-CAL-001",
                "message": "missing calibrator artifact for LearnedSpreadModelV3 promotion.",
                "mitigation": "Provide challenger.model_params.calibrator_run_dir containing calibrator.pkl.",
                "target_stage": maturity_stage,
            }
        )

    # STOP-CAL-002 / WARN-CAL-FRESH-001: calibrator must have been trained *after* the
    # model it calibrates.  A calibrator that predates the model was fitted on a different
    # model's raw outputs — its isotonic/Platt mappings do not apply to the current model.
    # At science_grade this is a hard stop; at mvp_operational it is a stage warning.
    if challenger_name in _CAL_FRESHNESS_MODELS and _calibrator_artifact_present(challenger_params):
        freshness = _check_calibrator_freshness(challenger_params)
        if freshness["stale"] or freshness["unreadable"]:
            if maturity_stage == "science_grade":
                hard_stops.append(
                    {
                        "id": "STOP-CAL-002",
                        "message": freshness["reason"],
                        "mitigation": (
                            "Re-run calibration training against the current model's hindcast outputs "
                            "to produce a fresh calibrator artifact, then update calibrator_run_dir."
                        ),
                        "target_stage": maturity_stage,
                        "calibrator_created_at": freshness["calibrator_created_at"],
                        "model_created_at": freshness["model_created_at"],
                    }
                )
            else:
                stage_warnings.append(
                    {
                        "id": "WARN-CAL-FRESH-001",
                        "tracking_id": "spread-science-debt-calibrator-freshness",
                        "warning": freshness["reason"],
                        "mitigation": (
                            "Re-run calibration training against the current model's hindcast outputs "
                            "to produce a fresh calibrator artifact, then update calibrator_run_dir. "
                            "This will become a hard stop (STOP-CAL-002) at science_grade."
                        ),
                        "target_stage": "science_grade",
                        "calibrator_created_at": freshness["calibrator_created_at"],
                        "model_created_at": freshness["model_created_at"],
                    }
                )

    # STOP-CONTRACT-001: challenger feature schema must exactly match the canonical
    # channel list for spatial models. A mismatch means train/infer channels diverged
    # and the metric comparison is scientifically invalid.
    if challenger_name in CANONICAL_CHANNELS_BY_MODEL:
        canonical = CANONICAL_CHANNELS_BY_MODEL[challenger_name]
        challenger_model_run_dir = (challenger_params or {}).get("model_run_dir")
        contract_stop: dict[str, Any] | None = None
        if not challenger_model_run_dir:
            contract_stop = {
                "id": "STOP-CONTRACT-001",
                "message": (
                    f"challenger {challenger_name!r} has no model_run_dir in model_params; "
                    "cannot verify feature contract."
                ),
                "mitigation": "Set challenger.model_params.model_run_dir to the trained model artifact directory.",
                "target_stage": maturity_stage,
            }
        else:
            run_dir = Path(challenger_model_run_dir)
            try:
                try:
                    infer_channels = load_contract(run_dir / "runtime_contract.json").channels
                except FileNotFoundError:
                    payload = json.loads((run_dir / "feature_schema.json").read_text(encoding="utf-8"))
                    raw = payload.get("channels")
                    if not isinstance(raw, list) or not raw:
                        raise KeyError("channels key missing or empty in feature_schema.json")
                    infer_channels = tuple(str(c) for c in raw)
                validate_channel_alignment(infer_channels, canonical)
            except (ContractViolationError, FileNotFoundError, KeyError, ValueError) as exc:
                contract_stop = {
                    "id": "STOP-CONTRACT-001",
                    "message": str(exc),
                    "mitigation": (
                        "Re-export the challenger model so its feature_schema.json / "
                        "runtime_contract.json matches CANONICAL_V2_CHANNELS, or update "
                        "CANONICAL_V2_CHANNELS and retrain."
                    ),
                    "target_stage": maturity_stage,
                }
        if contract_stop is not None:
            hard_stops.append(contract_stop)

    weighted_bss = float(decision.get("weighted_bss_improvement", 0.0) or 0.0)
    if weighted_bss <= 0.0:
        stage_warnings.append(
            {
                "id": "WARN-MVP-BSS-001",
                "tracking_id": "spread-science-debt-bss-positive-skill",
                "warning": f"Weighted BSS improvement must be > 0 for MVP; observed {weighted_bss:.4f}.",
                "mitigation": "Re-train challenger or refine inputs until aggregated BSS improvement is positive.",
                "target_stage": "mvp_operational",
            }
        )

    if not bool(decision.get("pass", False)):
        stage_warnings.append(
            {
                "id": "WARN-GATE-001",
                "tracking_id": "spread-science-debt-gate-regression",
                "warning": "Primary/secondary challenger gate did not pass.",
                "mitigation": "Address gate regressions listed in decision.reasons before promotion.",
                "target_stage": maturity_stage,
            }
        )

    if maturity_stage == "mvp_operational":
        science_debt_register.extend(
            [
                {
                    "debt_id": "SCI-DEBT-EXT-GT",
                    "tracking_id": "spread-science-debt-ext-ground-truth",
                    "description": "External ground-truth verification is not yet enforced in MVP gate.",
                    "target_stage": "science_grade",
                    "exit_criteria": "Validated against authoritative external ground-truth dataset.",
                },
                {
                    "debt_id": "SCI-DEBT-DM-SAL",
                    "tracking_id": "spread-science-debt-dm-sal-governance",
                    "description": "Science-grade DM significance and SAL threshold governance is deferred.",
                    "target_stage": "science_grade",
                    "exit_criteria": "DM significance and SAL thresholds enforced in promotion policy.",
                },
                {
                    "debt_id": "SCI-DEBT-DRIFT",
                    "tracking_id": "spread-science-debt-drift-monitoring",
                    "description": "Reliability/calibration drift monitoring controls are not yet mandatory.",
                    "target_stage": "science_grade",
                    "exit_criteria": "Drift monitors and alert thresholds are operational in production.",
                },
            ]
        )

    allow_promotion = bool(decision.get("pass", False)) and weighted_bss > 0.0 and not hard_stops
    promotion_decision = "promote_challenger" if allow_promotion else "hold_challenger"
    return {
        "maturity_stage": maturity_stage,
        "hard_stops": hard_stops,
        "stage_warnings": stage_warnings,
        "promotion_decision": promotion_decision,
        "science_debt_register": science_debt_register,
    }


def _collect_comparison_arrays(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    region_name = str(config["region_name"])
    bbox = tuple(float(v) for v in config["bbox"])
    start_time = datetime.fromisoformat(str(config["start_time"]))
    end_time = datetime.fromisoformat(str(config["end_time"]))
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    else:
        start_time = start_time.astimezone(timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    else:
        end_time = end_time.astimezone(timezone.utc)

    horizons = [int(h) for h in config.get("horizons_hours", [24, 48, 72])]
    min_detections = int(config.get("min_detections", 5))
    interval_hours = int(config.get("interval_hours", 24))
    label_window_hours = int(config.get("label_window_hours", 3))

    champion_cfg = config.get("champion", {})
    challenger_cfg = config.get("challenger", {})
    champ_name, champ_params = normalize_model_selection(
        champion_cfg.get("model_name"), champion_cfg.get("model_params")
    )
    chall_name, chall_params = normalize_model_selection(
        challenger_cfg.get("model_name"), challenger_cfg.get("model_params")
    )

    champion = get_spread_model(champ_name, champ_params)
    challenger = get_spread_model(chall_name, chall_params)

    # Pre-flight: validate grid alignment before sampling reference times.
    # Fail fast before any DB queries if the region grid diverges from the
    # canonical analysis-grid contract (CRS=EPSG:4326, cell_size=0.01°).
    preflight_grid = get_region_grid_spec(region_name)
    assert_grid_alignment(preflight_grid)

    engine = get_engine()
    ref_times = sample_fire_reference_times(
        engine=engine,
        bbox=bbox,
        start_time=start_time,
        end_time=end_time,
        min_detections=min_detections,
        interval_hours=interval_hours,
    )

    if not ref_times:
        raise ValueError("No reference times found for the provided config window.")

    acc: dict[int, dict[str, list[np.ndarray]]] = {
        h: {
            "y_true_flat": [],
            "champion_flat": [],
            "challenger_flat": [],
            "y_true_cases": [],
            "champion_cases": [],
            "challenger_cases": [],
        }
        for h in horizons
    }

    for ref_time in ref_times:
        inputs = build_spread_inputs(
            region_name=region_name,
            bbox=bbox,
            forecast_reference_time=ref_time,
            horizons_hours=horizons,
        )

        forecast_champion = champion.predict(inputs.to_model_input())
        forecast_challenger = challenger.predict(inputs.to_model_input())

        for i, h in enumerate(horizons):
            target_time = ref_time + timedelta(hours=int(h))
            target_start = target_time - timedelta(hours=label_window_hours)
            target_end = target_time + timedelta(hours=label_window_hours)

            obs = get_fire_cells_heatmap(
                region_name=region_name,
                bbox=bbox,
                start_time=target_start,
                end_time=target_end,
                mode="presence",
                clip=True,
            ).heatmap

            y_true_case = (np.asarray(obs) > 0).astype(np.float32)
            y_champion_case = np.asarray(
                forecast_champion.probabilities.isel(time=i).values, dtype=np.float32
            )
            y_challenger_case = np.asarray(
                forecast_challenger.probabilities.isel(time=i).values, dtype=np.float32
            )

            acc[h]["y_true_cases"].append(y_true_case)
            acc[h]["champion_cases"].append(y_champion_case)
            acc[h]["challenger_cases"].append(y_challenger_case)

            acc[h]["y_true_flat"].append(y_true_case.ravel())
            acc[h]["champion_flat"].append(y_champion_case.ravel())
            acc[h]["challenger_flat"].append(y_challenger_case.ravel())

    combined: dict[int, dict[str, Any]] = {}
    for h, payload in acc.items():
        if not payload["y_true_flat"]:
            continue
        combined[h] = {
            "y_true": np.concatenate(payload["y_true_flat"]),
            "champion": np.concatenate(payload["champion_flat"]),
            "challenger": np.concatenate(payload["challenger_flat"]),
            "y_true_cases": payload["y_true_cases"],
            "champion_cases": payload["champion_cases"],
            "challenger_cases": payload["challenger_cases"],
        }
    if not combined:
        raise ValueError("No paired evaluation arrays were collected.")
    return combined


def _plot_reliability_pair(
    *,
    horizon_hours: int,
    y_true: np.ndarray,
    y_champion: np.ndarray,
    y_challenger: np.ndarray,
    out_path: Path,
) -> None:
    bins = np.linspace(0.0, 1.0, 11)

    def curve(y_t: np.ndarray, y_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.digitize(y_p, bins[1:-1], right=False)
        xs = []
        ys = []
        for b in range(10):
            mask = idx == b
            if not np.any(mask):
                continue
            xs.append(float(np.mean(y_p[mask])))
            ys.append(float(np.mean(y_t[mask])))
        return np.asarray(xs), np.asarray(ys)

    y_true = (np.asarray(y_true) > 0.5).astype(float)
    x1, y1 = curve(y_true, np.clip(y_champion, 0.0, 1.0))
    x2, y2 = curve(y_true, np.clip(y_challenger, 0.0, 1.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    plt.plot(x1, y1, marker="o", linewidth=2, label="champion")
    plt.plot(x2, y2, marker="o", linewidth=2, label="challenger")
    plt.title(f"Reliability comparison (T+{int(horizon_hours)}h)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def run_eval(config: dict[str, Any], *, out_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight geo alignment check — run before any DB queries so that a CRS
    # or resolution mismatch lands in hard_stops rather than crashing the eval.
    geo_alignment_error: str | None = None
    try:
        preflight_grid = get_region_grid_spec(str(config["region_name"]))
        assert_grid_alignment(preflight_grid)
    except ValueError as exc:
        geo_alignment_error = str(exc)

    if geo_alignment_error is None:
        arrays = _collect_comparison_arrays(config)
        rows = []
        for h in sorted(arrays):
            row = summarize_comparison_for_horizon(
                horizon_hours=h,
                y_true=arrays[h]["y_true"],
                y_prob_champion=arrays[h]["champion"],
                y_prob_challenger=arrays[h]["challenger"],
                y_true_cases=arrays[h]["y_true_cases"],
                champion_cases=arrays[h]["champion_cases"],
                challenger_cases=arrays[h]["challenger_cases"],
                ece_bins=int(config.get("ece_bins", 10)),
            )
            rows.append(row)
    else:
        LOGGER.error("STOP-GEO-001: %s — skipping eval data collection.", geo_alignment_error)
        arrays = {}
        rows = []

    gate_cfg = config.get("gate", {}) or {}
    decision = compute_recommendation(
        rows,
        bss_improvement_min=float(gate_cfg.get("bss_improvement_min", 0.03)),
        bss_horizon_floor=float(gate_cfg.get("bss_horizon_floor", -0.005)),
        sal_regression_max=float(gate_cfg.get("sal_regression_max", 0.05)),
        dm_pvalue_max=float(gate_cfg.get("dm_pvalue_max", 0.05)),
        max_pr_auc_drop=float(gate_cfg.get("max_pr_auc_drop", 0.01)),
        max_iou_drop=float(gate_cfg.get("max_iou_drop", 0.02)),
    )
    stage_governance = _build_stage_governance(
        config=config,
        decision=decision,
        summary_rows=rows,
        geo_alignment_error=geo_alignment_error,
    )

    pd.DataFrame(rows).sort_values("horizon_hours").to_csv(out_dir / "summary.csv", index=False)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": rows,
        "decision": decision,
        "maturity_stage": stage_governance["maturity_stage"],
        "hard_stops": stage_governance["hard_stops"],
        "stage_warnings": stage_governance["stage_warnings"],
        "promotion_decision": stage_governance["promotion_decision"],
        "science_debt_register": stage_governance["science_debt_register"],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    weighted_bss = float(decision.get("weighted_bss_improvement", 0.0) or 0.0)
    report_pass = bool(decision.get("pass", False)) and weighted_bss > 0.0
    report_pass = report_pass and len(stage_governance["hard_stops"]) == 0
    gate_report = {
        "pass": report_pass,
        "recommend_challenger": bool(decision.get("recommend_challenger", False)),
        "decision": decision,
        "maturity_stage": stage_governance["maturity_stage"],
        "hard_stops": stage_governance["hard_stops"],
        "stage_warnings": stage_governance["stage_warnings"],
        "promotion_decision": stage_governance["promotion_decision"],
        "science_debt_register": stage_governance["science_debt_register"],
    }
    (out_dir / "gate_report.json").write_text(
        json.dumps(gate_report, indent=2) + "\n", encoding="utf-8"
    )

    decision_lines = [
        "# Champion vs Challenger Decision",
        "",
        f"- recommendation: `{'promote_challenger' if decision['recommend_challenger'] else 'keep_champion'}`",
        f"- maturity_stage: `{stage_governance['maturity_stage']}`",
        f"- promotion_decision: `{stage_governance['promotion_decision']}`",
        f"- pass: `{decision['pass']}`",
        f"- primary_ok: `{decision.get('primary_ok')}`",
        f"- secondary_ok: `{decision.get('secondary_ok')}`",
        f"- weighted_bss_improvement: `{decision.get('weighted_bss_improvement')}`",
        "",
        "## Reasons",
    ]
    for reason in decision["reasons"]:
        decision_lines.append(f"- {reason}")
    if stage_governance["hard_stops"]:
        decision_lines.append("")
        decision_lines.append("## Hard Stops")
        for item in stage_governance["hard_stops"]:
            decision_lines.append(f"- {item['id']}: {item['message']}")
    if stage_governance["stage_warnings"]:
        decision_lines.append("")
        decision_lines.append("## Stage Warnings")
        for item in stage_governance["stage_warnings"]:
            decision_lines.append(f"- {item['id']}: {item['warning']}")
    (out_dir / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    plots_cfg = config.get("plots", {}) or {}
    if bool(plots_cfg.get("enabled", True)):
        for h in sorted(arrays):
            _plot_reliability_pair(
                horizon_hours=h,
                y_true=arrays[h]["y_true"],
                y_champion=arrays[h]["champion"],
                y_challenger=arrays[h]["challenger"],
                out_path=out_dir / "plots" / f"reliability_h{int(h):03d}.png",
            )

    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate spread champion vs challenger.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/spread_champion_challenger",
        help="Output root directory.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = run_eval(config=config, out_root=Path(args.out_dir))
    LOGGER.info("Champion/challenger evaluation complete: %s", out_dir)


if __name__ == "__main__":
    main()
