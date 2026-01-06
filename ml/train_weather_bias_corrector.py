"""Train a lightweight weather bias corrector (affine per variable).

This script fits `WeatherBiasCorrector` using aligned forecast vs truth datasets
and writes JSON artifacts that can be loaded during spread inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr
import yaml

# Add project root to sys.path (mirrors other ml scripts).
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.weather_bias_analysis import (
    DEFAULT_VARS,
    align_datasets,
    compute_metrics,
    normalize_coords,
    normalize_truth,
)
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


def _time_split(ds: xr.Dataset, *, train_fraction: float) -> tuple[xr.Dataset, xr.Dataset]:
    n = int(ds.sizes.get("time", 0))
    if n < 2:
        raise ValueError("Need at least 2 time steps to do a train/validation split.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}.")
    n_train = max(1, min(n - 1, int(round(n * train_fraction))))
    return ds.isel(time=slice(0, n_train)), ds.isel(time=slice(n_train, None))


def run_training(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log")],
    )

    # 1) Load datasets
    LOGGER.info("Loading forecast: %s", args.forecast_nc)
    ds_forecast = normalize_coords(xr.open_dataset(args.forecast_nc))

    LOGGER.info("Loading truth: %s", args.truth_nc)
    if "*" in str(args.truth_nc):
        ds_truth_raw = xr.open_mfdataset(str(args.truth_nc), combine="by_coords")
    else:
        ds_truth_raw = xr.open_dataset(args.truth_nc)

    var_mapping = _parse_var_mapping(args.variables)
    ds_truth = normalize_truth(ds_truth_raw, var_mapping)

    # 2) Determine variables + align
    wanted = list(args.vars) if args.vars else list(DEFAULT_VARS)
    variables = [v for v in wanted if v in ds_forecast.data_vars and v in ds_truth.data_vars]
    if not variables:
        raise ValueError(
            "No overlapping variables found between forecast and truth. "
            f"forecast_vars={list(ds_forecast.data_vars)!r} truth_vars={list(ds_truth.data_vars)!r}"
        )

    LOGGER.info("Aligning datasets for variables: %s", variables)
    fc_aligned, tr_aligned = align_datasets(ds_forecast, ds_truth, variables)

    # 3) Split train/validation over time
    fc_train, fc_val = _time_split(fc_aligned, train_fraction=float(args.train_fraction))
    tr_train, tr_val = _time_split(tr_aligned, train_fraction=float(args.train_fraction))

    # 4) Fit + evaluate
    corrector = WeatherBiasCorrector.fit(forecast=fc_train, truth=tr_train, variables=variables)
    fc_val_corr = corrector.apply(fc_val)

    metrics_uncorr: dict[str, dict[str, float]] = {}
    metrics_corr: dict[str, dict[str, float]] = {}
    improvements: dict[str, dict[str, float]] = {}
    for v in variables:
        m0 = compute_metrics(fc_val[v].values, tr_val[v].values)
        m1 = compute_metrics(fc_val_corr[v].values, tr_val[v].values)
        metrics_uncorr[v] = m0
        metrics_corr[v] = m1
        improvements[v] = {
            "bias_mean_abs_reduction": float(abs(m0["bias_mean"]) - abs(m1["bias_mean"])),
            "mae_reduction": float(m0["mae"] - m1["mae"]),
            "rmse_reduction": float(m0["rmse"] - m1["rmse"]),
        }

    # 5) Write artifacts
    corrector_path = run_dir / "weather_bias_corrector.json"
    corrector.save_json(corrector_path)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "forecast_path": str(args.forecast_nc),
        "truth_path": str(args.truth_nc),
        "variables": variables,
        "train_fraction": float(args.train_fraction),
        "metrics_uncorrected_validation": metrics_uncorr,
        "metrics_corrected_validation": metrics_corr,
        "improvements": improvements,
        "artifact": {"weather_bias_corrector_json": str(corrector_path)},
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "config_resolved.yaml").write_text(yaml.dump(vars(args)), encoding="utf-8")

    LOGGER.info("Wrote corrector to %s", corrector_path)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weather bias correction model(s).")
    parser.add_argument("--forecast-nc", type=Path, required=True, help="Path to forecast NetCDF.")
    parser.add_argument("--truth-nc", type=str, required=True, help="Path or glob for truth NetCDF.")
    parser.add_argument(
        "--variables",
        type=str,
        help="Comma-separated mapping of truth variable names, e.g. u10=u10_era,t2m=t2m_era",
    )
    parser.add_argument(
        "--vars",
        nargs="*",
        default=None,
        help="Variables to fit (defaults to u10 v10 t2m rh2m when available).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of times to use for training; remainder is validation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "models" / "weather_bias_corrector",
        help="Output directory for run artifacts.",
    )

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

