"""Denoiser v2 drift monitor with optional registry rollback on hard violations."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.db import get_engine
from api.model_registry import resolve_active_model, rollback_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("denoiser_drift_monitor")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _score_histogram(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bin_edges)
    total = max(1, int(hist.sum()))
    return hist.astype(float) / float(total)


def compute_population_stability_index(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute PSI(current||baseline), stable for sparse buckets."""
    if len(baseline_scores) == 0 or len(current_scores) == 0:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, num=max(2, int(bins) + 1))
    edges = np.quantile(baseline_scores, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(0.0, 1.0, num=max(3, int(bins) + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    baseline_pct = _score_histogram(baseline_scores, edges)
    current_pct = _score_histogram(current_scores, edges)
    eps = 1e-9
    baseline_pct = np.clip(baseline_pct, eps, None)
    current_pct = np.clip(current_pct, eps, None)
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def _load_scores(
    engine: Engine,
    *,
    start_time: datetime,
    end_time: datetime,
    model_id: str | None = None,
) -> np.ndarray:
    model_predicate = ""
    params: dict[str, Any] = {
        "start_time": start_time,
        "end_time": end_time,
    }
    if model_id:
        model_predicate = "AND denoiser_model_id = :model_id"
        params["model_id"] = model_id

    stmt = text(
        f"""
        SELECT event_score
        FROM fire_detections
        WHERE denoiser_scored_at >= :start_time
          AND denoiser_scored_at < :end_time
          AND event_score IS NOT NULL
          {model_predicate}
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(stmt, params).fetchall()
    if not rows:
        return np.asarray([], dtype=float)
    scores = np.asarray([float(r[0]) for r in rows if r[0] is not None], dtype=float)
    return np.clip(scores, 0.0, 1.0)


def _insert_metric_row(
    engine: Engine,
    *,
    model_id: str | None,
    metric_name: str,
    metric_value: float,
    threshold_value: float | None,
    window_start: datetime,
    window_end: datetime,
    triggered_rollback: bool,
    payload: dict[str, Any],
) -> None:
    stmt = text(
        """
        INSERT INTO denoiser_drift_metrics (
            model_id,
            metric_name,
            metric_value,
            threshold_value,
            window_start,
            window_end,
            payload_json,
            triggered_rollback,
            created_at
        )
        VALUES (
            :model_id,
            :metric_name,
            :metric_value,
            :threshold_value,
            :window_start,
            :window_end,
            CAST(:payload_json AS JSONB),
            :triggered_rollback,
            NOW()
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(
            stmt,
            {
                "model_id": model_id,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "threshold_value": threshold_value,
                "window_start": window_start,
                "window_end": window_end,
                "payload_json": json.dumps(payload),
                "triggered_rollback": bool(triggered_rollback),
            },
        )


def monitor_denoiser_drift(
    *,
    window_hours: int = 24,
    baseline_days: int = 7,
    min_samples: int = 500,
    psi_warn_threshold: float = 0.2,
    psi_hard_threshold: float = 0.35,
    mean_delta_warn_threshold: float = 0.10,
    mean_delta_hard_threshold: float = 0.20,
    model_id: str | None = None,
    allow_rollback: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    engine = get_engine()
    active = resolve_active_model("denoiser", engine=engine)
    effective_model_id = model_id or (str(active["model_id"]) if active else None)

    now = _utc_now()
    current_end = now
    current_start = now - timedelta(hours=max(1, int(window_hours)))
    baseline_end = current_start
    baseline_start = baseline_end - timedelta(days=max(1, int(baseline_days)))

    baseline_scores = _load_scores(
        engine,
        start_time=baseline_start,
        end_time=baseline_end,
        model_id=effective_model_id,
    )
    current_scores = _load_scores(
        engine,
        start_time=current_start,
        end_time=current_end,
        model_id=effective_model_id,
    )

    summary: dict[str, Any] = {
        "as_of": now.isoformat(),
        "model_id": effective_model_id,
        "windows": {
            "baseline_start": baseline_start.isoformat(),
            "baseline_end": baseline_end.isoformat(),
            "current_start": current_start.isoformat(),
            "current_end": current_end.isoformat(),
        },
        "sample_counts": {
            "baseline": int(len(baseline_scores)),
            "current": int(len(current_scores)),
        },
        "metrics": {},
        "rollback": {"attempted": False, "triggered": False, "reason": None},
    }

    if len(baseline_scores) < int(min_samples) or len(current_scores) < int(min_samples):
        payload = {
            "status": "insufficient_samples",
            "min_samples": int(min_samples),
            "baseline_n": int(len(baseline_scores)),
            "current_n": int(len(current_scores)),
        }
        summary["metrics"] = {
            "psi_score": {"value": None, "status": "insufficient_samples"},
            "score_mean_delta": {"value": None, "status": "insufficient_samples"},
        }
        if not dry_run:
            _insert_metric_row(
                engine,
                model_id=effective_model_id,
                metric_name="psi_score",
                metric_value=0.0,
                threshold_value=psi_hard_threshold,
                window_start=current_start,
                window_end=current_end,
                triggered_rollback=False,
                payload=payload,
            )
            _insert_metric_row(
                engine,
                model_id=effective_model_id,
                metric_name="score_mean_delta",
                metric_value=0.0,
                threshold_value=mean_delta_hard_threshold,
                window_start=current_start,
                window_end=current_end,
                triggered_rollback=False,
                payload=payload,
            )
        return summary

    psi = compute_population_stability_index(baseline_scores, current_scores, bins=10)
    baseline_mean = float(np.mean(baseline_scores))
    current_mean = float(np.mean(current_scores))
    mean_delta = float(current_mean - baseline_mean)

    psi_severity = "ok"
    if psi >= psi_hard_threshold:
        psi_severity = "hard"
    elif psi >= psi_warn_threshold:
        psi_severity = "warn"

    mean_delta_abs = abs(mean_delta)
    mean_delta_severity = "ok"
    if mean_delta_abs >= mean_delta_hard_threshold:
        mean_delta_severity = "hard"
    elif mean_delta_abs >= mean_delta_warn_threshold:
        mean_delta_severity = "warn"

    hard_violation = psi_severity == "hard" or mean_delta_severity == "hard"
    summary["metrics"] = {
        "psi_score": {
            "value": psi,
            "warn_threshold": psi_warn_threshold,
            "hard_threshold": psi_hard_threshold,
            "severity": psi_severity,
        },
        "score_mean_delta": {
            "value": mean_delta,
            "warn_threshold": mean_delta_warn_threshold,
            "hard_threshold": mean_delta_hard_threshold,
            "severity": mean_delta_severity,
        },
    }

    rollback_triggered = False
    rollback_reason = None
    rollback_payload: dict[str, Any] | None = None
    if allow_rollback and hard_violation and not dry_run:
        summary["rollback"]["attempted"] = True
        try:
            rollback_reason = (
                f"denoiser_drift hard_violation psi={psi:.4f} "
                f"mean_delta={mean_delta:.4f}"
            )
            rollback_payload = rollback_model(
                family="denoiser",
                promoted_by="denoiser_drift_monitor",
                notes=rollback_reason,
                engine=engine,
            )
            rollback_triggered = True
        except Exception as exc:  # pragma: no cover - operational safety
            rollback_reason = f"rollback_failed: {exc}"
            LOGGER.exception("Denoiser drift rollback attempt failed")

    summary["rollback"]["triggered"] = rollback_triggered
    summary["rollback"]["reason"] = rollback_reason
    if rollback_payload is not None:
        summary["rollback"]["active"] = rollback_payload

    payload = {
        "baseline_mean": baseline_mean,
        "current_mean": current_mean,
        "baseline_n": int(len(baseline_scores)),
        "current_n": int(len(current_scores)),
        "psi_warn_threshold": psi_warn_threshold,
        "psi_hard_threshold": psi_hard_threshold,
        "mean_delta_warn_threshold": mean_delta_warn_threshold,
        "mean_delta_hard_threshold": mean_delta_hard_threshold,
        "severity": {
            "psi": psi_severity,
            "score_mean_delta": mean_delta_severity,
        },
        "hard_violation": hard_violation,
        "rollback": summary["rollback"],
    }
    if not dry_run:
        _insert_metric_row(
            engine,
            model_id=effective_model_id,
            metric_name="psi_score",
            metric_value=psi,
            threshold_value=psi_hard_threshold,
            window_start=current_start,
            window_end=current_end,
            triggered_rollback=rollback_triggered,
            payload=payload,
        )
        _insert_metric_row(
            engine,
            model_id=effective_model_id,
            metric_name="score_mean_delta",
            metric_value=mean_delta_abs,
            threshold_value=mean_delta_hard_threshold,
            window_start=current_start,
            window_end=current_end,
            triggered_rollback=rollback_triggered,
            payload=payload,
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor denoiser drift and optionally rollback champion.")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--baseline-days", type=int, default=7)
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--psi-warn", type=float, default=0.2)
    parser.add_argument("--psi-hard", type=float, default=0.35)
    parser.add_argument("--mean-delta-warn", type=float, default=0.10)
    parser.add_argument("--mean-delta-hard", type=float, default=0.20)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-rollback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = monitor_denoiser_drift(
        window_hours=args.window_hours,
        baseline_days=args.baseline_days,
        min_samples=args.min_samples,
        psi_warn_threshold=args.psi_warn,
        psi_hard_threshold=args.psi_hard,
        mean_delta_warn_threshold=args.mean_delta_warn,
        mean_delta_hard_threshold=args.mean_delta_hard,
        model_id=args.model_id,
        allow_rollback=not args.no_rollback,
        dry_run=bool(args.dry_run),
    )
    LOGGER.info("Denoiser drift summary: %s", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
