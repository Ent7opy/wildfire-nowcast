"""Evaluate the impact of probability calibration on spread hindcast quality.

This script compares raw vs calibrated probabilities on a hindcast dataset and
produces:
- per-horizon metrics (Brier score, ECE, reliability curve points),
- summary tables (CSV/JSON),
- reliability plots (raw vs calibrated).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

# Add project root to sys.path (mirrors other ml scripts).
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.calibration import SpreadProbabilityCalibrator

LOGGER = logging.getLogger(__name__)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10
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
    # Assign each prob to a bin index in [0, n_bins-1]
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


def _load_hindcast_manifest(hindcast_run_dir: Path) -> dict[str, Any]:
    index_path = hindcast_run_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Hindcast manifest not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def _iter_case_paths(hindcast_run_dir: Path) -> Iterable[Path]:
    manifest = _load_hindcast_manifest(hindcast_run_dir)
    for c in manifest.get("cases", []):
        p = Path(c["path"])
        if not p.is_absolute():
            p = hindcast_run_dir / p
        yield p


def _collect_horizon_arrays(
    hindcast_run_dir: Path,
    *,
    p_min: float = 1e-4,
) -> dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return {horizon_hours: (y_pred_flat, y_obs_flat)} over all cases."""
    out: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

    for case_path in _iter_case_paths(hindcast_run_dir):
        if not case_path.exists():
            LOGGER.warning("Case file missing: %s (skipping)", case_path)
            continue

        with xr.open_dataset(case_path) as ds:
            if "y_pred" not in ds or "y_obs" not in ds:
                LOGGER.warning("Case missing y_pred/y_obs: %s (skipping)", case_path)
                continue
            lead = ds.get("lead_time_hours")
            if lead is None:
                LOGGER.warning("Case missing lead_time_hours: %s (skipping)", case_path)
                continue

            fire_t0 = ds.get("fire_t0")
            fire_mask = (np.asarray(fire_t0.values) > 0) if fire_t0 is not None else None

            horizons = [int(x) for x in np.asarray(lead.values).astype(int).tolist()]
            for i, h in enumerate(horizons):
                y_pred = np.asarray(ds["y_pred"].isel(time=i).values, dtype=float)
                y_obs = np.asarray(ds["y_obs"].isel(time=i).values, dtype=float)

                # Match calibration training mask: (fire_t0 > 0) | (y_pred >= p_min)
                if fire_mask is not None:
                    mask = fire_mask | (y_pred >= float(p_min))
                else:
                    mask = y_pred >= float(p_min)

                out.setdefault(h, []).append((y_pred[mask], y_obs[mask]))

    combined: dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for h, pairs in out.items():
        if not pairs:
            continue
        y_pred = np.concatenate([p[0] for p in pairs]).astype(np.float32, copy=False)
        y_obs = np.concatenate([p[1] for p in pairs]).astype(np.float32, copy=False)
        combined[int(h)] = (y_pred, y_obs)
    return combined


def _reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = (y_true[valid] > 0.5).astype(float)
    y_prob = np.clip(y_prob[valid], 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)

    prob_true = []
    prob_pred = []
    counts = []
    for b in range(int(n_bins)):
        mask = idx == b
        if not np.any(mask):
            continue
        counts.append(int(np.sum(mask)))
        prob_true.append(float(np.mean(y_true[mask])))
        prob_pred.append(float(np.mean(y_prob[mask])))

    return {"prob_true": prob_true, "prob_pred": prob_pred, "counts": counts, "n_bins": int(n_bins)}


def _plot_reliability(
    *,
    horizon_hours: int,
    raw_curve: dict[str, Any],
    cal_curve: dict[str, Any],
    brier_raw: float,
    brier_cal: float,
    ece_raw: float,
    ece_cal: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 6.2))

    # Perfect calibration.
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, alpha=0.7, label="perfect")

    plt.plot(
        raw_curve["prob_pred"],
        raw_curve["prob_true"],
        marker="o",
        linewidth=2,
        label=f"raw (Brier={brier_raw:.4f}, ECE={ece_raw:.4f})",
    )
    plt.plot(
        cal_curve["prob_pred"],
        cal_curve["prob_true"],
        marker="o",
        linewidth=2,
        label=f"calibrated (Brier={brier_cal:.4f}, ECE={ece_cal:.4f})",
    )

    plt.title(f"Reliability diagram (T+{int(horizon_hours)}h)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def run_eval(args: argparse.Namespace) -> Path:
    hindcast_run_dir = Path(args.hindcast_run_dir)
    calibrator_run_dir = Path(args.calibrator_run_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    calibrator = SpreadProbabilityCalibrator.load(calibrator_run_dir)
    meta = calibrator.metadata or {}
    run_id = meta.get("run_id") or calibrator_run_dir.name

    out_dir = Path(args.out_dir) / f"{timestamp}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility.
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")

    horizon_arrays = _collect_horizon_arrays(hindcast_run_dir, p_min=float(args.p_min))
    if not horizon_arrays:
        raise ValueError(f"No usable hindcast cases found in {hindcast_run_dir}")

    summary_rows: list[dict[str, Any]] = []
    curves: dict[int, dict[str, Any]] = {}

    for h, (y_pred, y_obs) in sorted(horizon_arrays.items()):
        # Raw metrics
        y_true = (y_obs > 0.5).astype(np.float32, copy=False)
        y_raw = np.clip(y_pred, 0.0, 1.0)
        brier_raw = float(np.mean((y_raw - y_true) ** 2))
        ece_raw = expected_calibration_error(y_true, y_raw, n_bins=int(args.n_bins))
        raw_curve = _reliability_curve(y_true, y_raw, n_bins=int(args.n_bins))

        # Calibrated metrics
        y_cal = calibrator.calibrate_probs(y_raw, int(h))
        brier_cal = float(np.mean((y_cal - y_true) ** 2))
        ece_cal = expected_calibration_error(y_true, y_cal, n_bins=int(args.n_bins))
        cal_curve = _reliability_curve(y_true, y_cal, n_bins=int(args.n_bins))

        curves[int(h)] = {"raw": raw_curve, "calibrated": cal_curve}

        summary_rows.append(
            {
                "horizon_hours": int(h),
                "n": int(y_true.size),
                "brier_raw": brier_raw,
                "brier_cal": brier_cal,
                "brier_improvement": brier_raw - brier_cal,
                "ece_raw": float(ece_raw),
                "ece_cal": float(ece_cal),
                "ece_improvement": float(ece_raw - ece_cal),
            }
        )

        _plot_reliability(
            horizon_hours=int(h),
            raw_curve=raw_curve,
            cal_curve=cal_curve,
            brier_raw=brier_raw,
            brier_cal=brier_cal,
            ece_raw=float(ece_raw),
            ece_cal=float(ece_cal),
            out_path=out_dir / "plots" / f"reliability_h{int(h):03d}.png",
        )

    # Write artifacts
    import pandas as pd

    df = pd.DataFrame(summary_rows).sort_values("horizon_hours")
    df.to_csv(out_dir / "summary.csv", index=False)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hindcast_run_dir": str(hindcast_run_dir),
        "calibrator_run_dir": str(calibrator_run_dir),
        "calibrator_metadata": meta,
        "summary": summary_rows,
        "reliability_curves": {str(k): v for k, v in curves.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Friendly notes file.
    (out_dir / "notes.md").write_text(
        "\n".join(
            [
                "# Spread Calibration Evaluation",
                "",
                f"- **Hindcast**: `{hindcast_run_dir}`",
                f"- **Calibrator**: `{calibrator_run_dir}`",
                "",
                "## What to look for",
                "- Reliability curves closer to the diagonal after calibration.",
                "- Positive `brier_improvement` and `ece_improvement` in `summary.csv`.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Wrote calibration eval report to %s", out_dir)
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Evaluate spread probability calibration on a hindcast run.")
    p.add_argument("--hindcast-run-dir", type=str, required=True, help="Path to hindcast run directory (index.json + cases).")
    p.add_argument("--calibrator-run-dir", type=str, required=True, help="Path to calibrator run directory (calibrator.pkl).")
    p.add_argument("--p-min", type=float, default=1e-4, help="Candidate mask threshold (match training).")
    p.add_argument("--n-bins", type=int, default=10, help="Number of bins for reliability/ECE.")
    p.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "reports" / "spread_calibration_eval"), help="Output root directory.")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()

