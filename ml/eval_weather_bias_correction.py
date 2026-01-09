"""Evaluate weather bias correction impact (raw vs corrected vs truth).

This script compares a forecast NetCDF against a truth NetCDF (e.g. ERA5),
computing bias/MAE/RMSE and producing plots that make it easy to see whether
bias correction reduced systematic error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

# Add project root to sys.path (mirrors other ml scripts).
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.weather_bias_analysis import DEFAULT_VARS, VAR_NAMES_HUMAN, align_datasets, compute_metrics, normalize_coords, normalize_truth
from ml.weather_bias_correction import WeatherBiasCorrector

LOGGER = logging.getLogger(__name__)


def _parse_var_mapping(s: str | None) -> dict[str, str]:
    if not s:
        return {}
    mapping: dict[str, str] = {}
    for pair in s.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid --variables entry {pair!r}; expected key=value.")
        k, v = pair.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def _plot_bias_map(mean_bias: xr.DataArray, *, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    mean_bias.plot(cmap="RdBu_r", robust=True)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_bias_timeseries(
    bias_raw: xr.DataArray,
    bias_corr: xr.DataArray,
    *,
    title: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Domain mean
    raw_series = bias_raw.mean(dim=["lat", "lon"])
    corr_series = bias_corr.mean(dim=["lat", "lon"])
    plt.figure(figsize=(12, 6))
    raw_series.plot(marker="o", label="raw")
    corr_series.plot(marker="o", label="corrected")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title(title)
    plt.ylabel("Bias (Forecast - Truth)")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def run_eval(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")

    LOGGER.info("Loading forecast: %s", args.forecast_nc)
    fc = normalize_coords(xr.open_dataset(args.forecast_nc))

    LOGGER.info("Loading truth: %s", args.truth_nc)
    if "*" in str(args.truth_nc):
        tr_raw = xr.open_mfdataset(str(args.truth_nc), combine="by_coords")
    else:
        tr_raw = xr.open_dataset(args.truth_nc)
    tr = normalize_truth(tr_raw, _parse_var_mapping(args.variables))

    wanted = list(args.vars) if args.vars else list(DEFAULT_VARS)
    variables = [v for v in wanted if v in fc.data_vars and v in tr.data_vars]
    if not variables:
        raise ValueError(
            "No overlapping variables found between forecast and truth. "
            f"forecast_vars={list(fc.data_vars)!r} truth_vars={list(tr.data_vars)!r}"
        )

    fc_aligned, tr_aligned = align_datasets(fc, tr, variables)

    corrector = WeatherBiasCorrector.load_json(args.corrector_json)
    fc_corr = corrector.apply(fc_aligned)

    # Compute bias datasets
    bias_raw = fc_aligned - tr_aligned
    bias_corr = fc_corr - tr_aligned

    rows: list[dict[str, Any]] = []
    for v in variables:
        m_raw = compute_metrics(fc_aligned[v].values, tr_aligned[v].values)
        m_corr = compute_metrics(fc_corr[v].values, tr_aligned[v].values)
        rows.append(
            {
                "variable": v,
                **{f"raw_{k}": m_raw[k] for k in ("bias_mean", "bias_std", "mae", "rmse", "count")},
                **{f"corr_{k}": m_corr[k] for k in ("bias_mean", "bias_std", "mae", "rmse", "count")},
                "bias_mean_abs_reduction": float(abs(m_raw["bias_mean"]) - abs(m_corr["bias_mean"])),
                "mae_reduction": float(m_raw["mae"] - m_corr["mae"]),
                "rmse_reduction": float(m_raw["rmse"] - m_corr["rmse"]),
            }
        )

        # Plots: mean bias maps (raw, corrected, reduction) + time series overlay
        mean_raw = bias_raw[v].mean(dim="time")
        mean_corr = bias_corr[v].mean(dim="time")
        mean_reduction = mean_raw - mean_corr

        name = VAR_NAMES_HUMAN.get(v, v)
        _plot_bias_map(mean_raw, title=f"Mean Bias (raw): {name}", path=plots_dir / f"bias_map_raw_{v}.png")
        _plot_bias_map(mean_corr, title=f"Mean Bias (corrected): {name}", path=plots_dir / f"bias_map_corrected_{v}.png")
        _plot_bias_map(mean_reduction, title=f"Mean Bias Reduction (raw - corrected): {name}", path=plots_dir / f"bias_map_reduction_{v}.png")

        _plot_bias_timeseries(
            bias_raw=bias_raw[v],
            bias_corr=bias_corr[v],
            title=f"Domain-Mean Bias over Time: {name}",
            path=plots_dir / f"bias_ts_raw_vs_corrected_{v}.png",
        )

    df = pd.DataFrame(rows).sort_values("variable")
    df.to_csv(out_dir / "summary.csv", index=False)

    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "forecast_path": str(args.forecast_nc),
        "truth_path": str(args.truth_nc),
        "corrector_json": str(args.corrector_json),
        "variables": variables,
        "summary": rows,
        "corrector": corrector.to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    (out_dir / "notes.md").write_text(
        "\n".join(
            [
                "# Weather Bias Correction Evaluation",
                "",
                f"- **Forecast**: `{args.forecast_nc}`",
                f"- **Truth**: `{args.truth_nc}`",
                f"- **Corrector**: `{args.corrector_json}`",
                "",
                "## What to look for",
                "- In `summary.csv`, positive `mae_reduction` and `rmse_reduction`.",
                "- Bias maps moving toward 0 after correction.",
                "- Time-series curves closer to 0 after correction.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Wrote weather bias correction eval report to %s", out_dir)
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Evaluate weather bias correction vs truth.")
    p.add_argument("--forecast-nc", type=Path, required=True, help="Path to forecast NetCDF.")
    p.add_argument("--truth-nc", type=str, required=True, help="Path or glob for truth NetCDF.")
    p.add_argument("--corrector-json", type=Path, required=True, help="Path to weather_bias_corrector.json")
    p.add_argument("--variables", type=str, help="Comma-separated mapping of truth variables, e.g. u10=u10_era,t2m=t2m_era")
    p.add_argument("--vars", nargs="*", default=None, help="Variables to evaluate (defaults to u10 v10 t2m rh2m when available).")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "weather_bias_correction_eval", help="Output root directory.")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()

