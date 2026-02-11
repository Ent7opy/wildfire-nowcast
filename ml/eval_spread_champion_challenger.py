"""Evaluate champion vs challenger spread models on identical reference cases.

Produces per-horizon comparison metrics and a conservative recommendation.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from api.db import get_engine
from api.fires.service import get_fire_cells_heatmap
from ml.spread.factory import get_spread_model, normalize_model_selection
from ml.spread.hindcast_dataset import sample_fire_reference_times
from ml.spread_features import build_spread_inputs

LOGGER = logging.getLogger(__name__)


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


def summarize_comparison_for_horizon(
    *,
    horizon_hours: int,
    y_true: np.ndarray,
    y_prob_champion: np.ndarray,
    y_prob_challenger: np.ndarray,
    ece_bins: int = 10,
) -> dict[str, Any]:
    y_true = (np.asarray(y_true) > 0.5).astype(np.float32, copy=False)
    p_champion = np.clip(np.asarray(y_prob_champion, dtype=np.float32), 0.0, 1.0)
    p_challenger = np.clip(np.asarray(y_prob_challenger, dtype=np.float32), 0.0, 1.0)

    champion_brier = float(np.mean((p_champion - y_true) ** 2))
    challenger_brier = float(np.mean((p_challenger - y_true) ** 2))

    champion_ece = expected_calibration_error(y_true, p_champion, n_bins=ece_bins)
    challenger_ece = expected_calibration_error(y_true, p_challenger, n_bins=ece_bins)

    champion_pr_auc = _safe_pr_auc(y_true, p_champion)
    challenger_pr_auc = _safe_pr_auc(y_true, p_challenger)

    champion_iou_03 = _binary_metrics(y_true, p_champion, 0.3)["iou"]
    challenger_iou_03 = _binary_metrics(y_true, p_challenger, 0.3)["iou"]
    champion_iou_05 = _binary_metrics(y_true, p_champion, 0.5)["iou"]
    challenger_iou_05 = _binary_metrics(y_true, p_challenger, 0.5)["iou"]

    return {
        "horizon_hours": int(horizon_hours),
        "n": int(y_true.size),
        "champion_brier": champion_brier,
        "challenger_brier": challenger_brier,
        "brier_improvement": champion_brier - challenger_brier,
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
    }


def compute_recommendation(
    summary_rows: list[dict[str, Any]],
    *,
    max_pr_auc_drop: float = 0.01,
    max_iou_drop: float = 0.02,
) -> dict[str, Any]:
    """Conservative gate for champion/challenger recommendation."""
    if not summary_rows:
        return {
            "recommend_challenger": False,
            "reasons": ["No summary rows available."],
        }

    reasons: list[str] = []
    primary_ok = True
    secondary_ok = True

    for row in summary_rows:
        h = int(row["horizon_hours"])
        if float(row["brier_improvement"]) <= 0:
            primary_ok = False
            reasons.append(f"T+{h}h: challenger did not improve Brier.")
        if float(row["ece_improvement"]) <= 0:
            primary_ok = False
            reasons.append(f"T+{h}h: challenger did not improve ECE.")

        pr_auc_improvement = row.get("pr_auc_improvement")
        if pr_auc_improvement is not None and float(pr_auc_improvement) < -abs(max_pr_auc_drop):
            secondary_ok = False
            reasons.append(
                f"T+{h}h: PR-AUC regression exceeds threshold ({pr_auc_improvement:.4f})."
            )

        for key in ("iou_03_improvement", "iou_05_improvement"):
            if float(row[key]) < -abs(max_iou_drop):
                secondary_ok = False
                reasons.append(f"T+{h}h: {key} regression exceeds threshold ({row[key]:.4f}).")

    recommend = primary_ok and secondary_ok
    if recommend:
        reasons.append("Challenger improves primary metrics on all horizons with no major secondary regressions.")

    return {
        "recommend_challenger": bool(recommend),
        "primary_ok": bool(primary_ok),
        "secondary_ok": bool(secondary_ok),
        "max_pr_auc_drop": float(max_pr_auc_drop),
        "max_iou_drop": float(max_iou_drop),
        "reasons": reasons,
    }


def _collect_comparison_arrays(config: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
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
        h: {"y_true": [], "champion": [], "challenger": []} for h in horizons
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

            y_true = (np.asarray(obs).ravel() > 0).astype(np.float32)
            y_champion = np.asarray(forecast_champion.probabilities.isel(time=i).values).ravel().astype(np.float32)
            y_challenger = np.asarray(forecast_challenger.probabilities.isel(time=i).values).ravel().astype(np.float32)

            acc[h]["y_true"].append(y_true)
            acc[h]["champion"].append(y_champion)
            acc[h]["challenger"].append(y_challenger)

    combined: dict[int, dict[str, np.ndarray]] = {}
    for h, payload in acc.items():
        if not payload["y_true"]:
            continue
        combined[h] = {
            "y_true": np.concatenate(payload["y_true"]),
            "champion": np.concatenate(payload["champion"]),
            "challenger": np.concatenate(payload["challenger"]),
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

    arrays = _collect_comparison_arrays(config)
    rows = []
    for h in sorted(arrays):
        row = summarize_comparison_for_horizon(
            horizon_hours=h,
            y_true=arrays[h]["y_true"],
            y_prob_champion=arrays[h]["champion"],
            y_prob_challenger=arrays[h]["challenger"],
            ece_bins=int(config.get("ece_bins", 10)),
        )
        rows.append(row)

    gate_cfg = config.get("gate", {}) or {}
    decision = compute_recommendation(
        rows,
        max_pr_auc_drop=float(gate_cfg.get("max_pr_auc_drop", 0.01)),
        max_iou_drop=float(gate_cfg.get("max_iou_drop", 0.02)),
    )

    pd.DataFrame(rows).sort_values("horizon_hours").to_csv(out_dir / "summary.csv", index=False)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": rows,
        "decision": decision,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    decision_lines = [
        "# Champion vs Challenger Decision",
        "",
        f"- recommendation: `{'promote_challenger' if decision['recommend_challenger'] else 'keep_champion'}`",
        f"- primary_ok: `{decision['primary_ok']}`",
        f"- secondary_ok: `{decision['secondary_ok']}`",
        f"- max_pr_auc_drop: `{decision['max_pr_auc_drop']}`",
        f"- max_iou_drop: `{decision['max_iou_drop']}`",
        "",
        "## Reasons",
    ]
    for reason in decision["reasons"]:
        decision_lines.append(f"- {reason}")
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
