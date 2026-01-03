"""Weather bias analysis: compare GFS forecasts with ERA5 reanalysis/truth.

Computes systematic biases, MAE, and RMSE for wind, temperature, and humidity.
Outputs summary tables and plots to a versioned run directory.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

# Add project root to sys.path
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from api.core.grid import GridSpec

LOGGER = logging.getLogger(__name__)

DEFAULT_VARS = ["u10", "v10", "t2m", "rh2m"]
VAR_NAMES_HUMAN = {
    "u10": "U-Wind (10m) [m/s]",
    "v10": "V-Wind (10m) [m/s]",
    "t2m": "Temperature (2m) [K]",
    "rh2m": "Relative Humidity (2m) [%]",
    "wind_speed": "Wind Speed [m/s]",
}


def compute_metrics(forecast: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Compute basic error metrics between forecast and truth arrays."""
    diff = forecast - truth
    # Filter NaNs
    valid = np.isfinite(diff)
    if not np.any(valid):
        return {
            "bias_mean": np.nan,
            "bias_std": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "count": 0,
        }

    diff_valid = diff[valid]
    return {
        "bias_mean": float(np.mean(diff_valid)),
        "bias_std": float(np.std(diff_valid)),
        "mae": float(np.mean(np.abs(diff_valid))),
        "rmse": float(np.sqrt(np.mean(diff_valid**2))),
        "count": int(np.sum(valid)),
    }


def normalize_truth(ds: xr.Dataset, var_mapping: Dict[str, str]) -> xr.Dataset:
    """Normalize truth dataset: rename variables and coordinates, sort by coords."""
    # Rename coordinates to canonical time, lat, lon
    rename_coords = {}
    if "latitude" in ds.coords:
        rename_coords["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_coords["longitude"] = "lon"
    if rename_coords:
        ds = ds.rename(rename_coords)

    # Rename variables based on mapping
    if var_mapping:
        ds = ds.rename({v: k for k, v in var_mapping.items() if v in ds.data_vars})

    # Ensure monotonic increasing lat/lon
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")
    if "time" in ds.coords:
        ds = ds.sortby("time")

    return ds


def align_datasets(
    forecast: xr.Dataset, truth: xr.Dataset, variables: List[str]
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Align truth dataset to forecast grid and time coordinates."""
    # Subset truth to forecast time range and bbox
    t_min, t_max = forecast.time.min().values, forecast.time.max().values
    lat_min, lat_max = forecast.lat.min().values, forecast.lat.max().values
    lon_min, lon_max = forecast.lon.min().values, forecast.lon.max().values

    truth_sub = truth.sel(
        time=slice(t_min, t_max),
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )

    # Interpolate truth onto forecast grid
    # We use linear interpolation as it's standard for reanalysis to analysis grid
    truth_aligned = truth_sub.interp(
        time=forecast.time,
        lat=forecast.lat,
        lon=forecast.lon,
        method="linear",
    )

    return forecast[variables], truth_aligned[variables]


def plot_bias_maps(
    bias_ds: xr.Dataset, variables: List[str], run_dir: Path
) -> List[Path]:
    """Generate and save mean bias maps for each variable."""
    plot_paths = []
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for var in variables:
        mean_bias = bias_ds[var].mean(dim="time")
        plt.figure(figsize=(10, 8))
        mean_bias.plot(cmap="RdBu_r", robust=True)
        plt.title(f"Mean Bias: {VAR_NAMES_HUMAN.get(var, var)}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        path = plots_dir / f"bias_map_{var}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        plot_paths.append(path)

    return plot_paths


def plot_bias_timeseries(
    bias_ds: xr.Dataset, variables: List[str], run_dir: Path
) -> List[Path]:
    """Generate and save domain-mean bias time series."""
    plot_paths = []
    plots_dir = run_dir / "plots"

    for var in variables:
        domain_mean = bias_ds[var].mean(dim=["lat", "lon"])
        plt.figure(figsize=(12, 6))
        domain_mean.plot(marker="o")
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.title(f"Domain-Mean Bias over Time: {VAR_NAMES_HUMAN.get(var, var)}")
        plt.ylabel("Bias (Forecast - Truth)")
        plt.xlabel("Time")
        
        path = plots_dir / f"bias_ts_{var}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        plot_paths.append(path)

    return plot_paths


def compute_quadrant_stats(bias_ds: xr.Dataset, variables: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute mean bias for 2x2 quadrants of the domain."""
    lats = bias_ds.lat.values
    lons = bias_ds.lon.values
    lat_mid = (lats.min() + lats.max()) / 2
    lon_mid = (lons.min() + lons.max()) / 2

    quadrants = {
        "NW": bias_ds.where((bias_ds.lat >= lat_mid) & (bias_ds.lon < lon_mid)),
        "NE": bias_ds.where((bias_ds.lat >= lat_mid) & (bias_ds.lon >= lon_mid)),
        "SW": bias_ds.where((bias_ds.lat < lat_mid) & (bias_ds.lon < lon_mid)),
        "SE": bias_ds.where((bias_ds.lat < lat_mid) & (bias_ds.lon >= lon_mid)),
    }

    stats = {}
    for q_name, q_ds in quadrants.items():
        q_stats = {}
        for var in variables:
            val = float(q_ds[var].mean().values)
            q_stats[var] = val
        stats[q_name] = q_stats
    return stats


def compute_elevation_stats(
    bias_ds: xr.Dataset, dem: xr.DataArray, variables: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute bias metrics binned by elevation."""
    # Align DEM to bias grid
    dem_aligned = dem.interp(lat=bias_ds.lat, lon=bias_ds.lon, method="nearest")
    
    # Define bins (e.g. every 500m)
    bins = [0, 500, 1000, 2000, 3000, 5000]
    labels = ["0-500m", "500-1000m", "1000-2000m", "2000-3000m", "3000m+"]
    
    stats = {}
    for i in range(len(bins) - 1):
        mask = (dem_aligned >= bins[i]) & (dem_aligned < bins[i+1])
        if i == len(bins) - 2:
            mask = dem_aligned >= bins[i]
            
        b_ds = bias_ds.where(mask)
        b_stats = {}
        for var in variables:
            val = float(b_ds[var].mean().values)
            b_stats[var] = val
        stats[labels[i]] = b_stats
    return stats


def run_analysis(args: argparse.Namespace) -> None:
    """Main analysis execution loop."""
    # 1. Setup run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging for this run
    log_file = run_dir / "analysis.log"
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # We use a specific logger for the analysis to avoid polluting root if imported
    # But basicConfig was used before. Let's attach to root but keep track of handlers to remove them.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    LOGGER.info("Starting weather bias analysis run: %s", run_id)

    # 2. Save metadata and config
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
        }
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(run_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # 3. Load datasets
    if not args.forecast_nc.exists():
        LOGGER.error("Forecast file not found: %s", args.forecast_nc)
        return

    LOGGER.info("Loading forecast: %s", args.forecast_nc)
    ds_forecast = xr.open_dataset(args.forecast_nc)
    
    # Check truth path existence (handling globs)
    truth_path = Path(args.truth_nc)
    if "*" not in str(args.truth_nc) and not truth_path.exists():
        LOGGER.error("Truth file not found: %s", args.truth_nc)
        LOGGER.info("Please provide a valid path to a NetCDF file containing truth/reanalysis data.")
        return

    LOGGER.info("Loading truth: %s", args.truth_nc)
    # Support multiple files if glob passed
    if "*" in str(args.truth_nc):
        ds_truth_raw = xr.open_mfdataset(str(args.truth_nc), combine="by_coords")
    else:
        ds_truth_raw = xr.open_dataset(args.truth_nc)

    # 4. Normalize and Align
    var_mapping = {}
    if args.variables:
        # Expected format: key=val,key2=val2
        for pair in args.variables.split(","):
            if "=" in pair:
                k, v = pair.split("=")
                var_mapping[k.strip()] = v.strip()

    ds_truth = normalize_truth(ds_truth_raw, var_mapping)
    
    # Check if we have required variables
    available_vars = [v for v in DEFAULT_VARS if v in ds_forecast.data_vars and v in ds_truth.data_vars]
    if not available_vars:
        LOGGER.error("No overlapping variables found between forecast and truth.")
        LOGGER.info("Forecast variables: %s", list(ds_forecast.data_vars))
        LOGGER.info("Truth variables: %s", list(ds_truth.data_vars))
        return

    LOGGER.info("Aligning datasets on variables: %s", available_vars)
    fc_aligned, tr_aligned = align_datasets(ds_forecast, ds_truth, available_vars)

    # 5. Compute Wind Speed if both u/v available
    if "u10" in available_vars and "v10" in available_vars:
        LOGGER.info("Computing derived wind speed...")
        fc_aligned["wind_speed"] = np.sqrt(fc_aligned.u10**2 + fc_aligned.v10**2)
        tr_aligned["wind_speed"] = np.sqrt(tr_aligned.u10**2 + tr_aligned.v10**2)
        available_vars.append("wind_speed")

    # 6. Compute Biases and Metrics
    bias_ds = fc_aligned - tr_aligned
    
    metrics_list = []
    for var in available_vars:
        m = compute_metrics(fc_aligned[var].values, tr_aligned[var].values)
        m["variable"] = var
        metrics_list.append(m)

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(run_dir / "summary.csv", index=False)
    
    quadrant_stats = compute_quadrant_stats(bias_ds, available_vars)
    
    elevation_stats = None
    if args.dem_path and args.dem_path.exists():
        LOGGER.info("Computing elevation-based stats using DEM: %s", args.dem_path)
        # Use rioxarray to open GeoTIFF if it's one
        import rioxarray
        dem_da = rioxarray.open_rasterio(args.dem_path, masked=True)
        if "band" in dem_da.dims:
            dem_da = dem_da.squeeze("band", drop=True)
        # Rename to lat/lon if needed
        if "y" in dem_da.coords: dem_da = dem_da.rename({"y": "lat"})
        if "x" in dem_da.coords: dem_da = dem_da.rename({"x": "lon"})
        elevation_stats = compute_elevation_stats(bias_ds, dem_da, available_vars)

    summary_json = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics_list,
        "quadrant_stats": quadrant_stats,
        "elevation_stats": elevation_stats,
        "forecast_path": str(args.forecast_nc),
        "truth_path": str(args.truth_nc),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # 7. Plots
    LOGGER.info("Generating plots...")
    plot_bias_maps(bias_ds, available_vars, run_dir)
    plot_bias_timeseries(bias_ds, available_vars, run_dir)

    # 8. Generate notes.md
    LOGGER.info("Generating notes.md...")
    with open(run_dir / "notes.md", "w") as f:
        f.write(f"# Weather Bias Analysis Report\n\n")
        f.write(f"- **Run ID**: `{run_id}`\n")
        f.write(f"- **Forecast**: `{args.forecast_nc}`\n")
        f.write(f"- **Truth**: `{args.truth_nc}`\n\n")
        f.write("## Summary Metrics\n\n")
        f.write(df_metrics.to_string(index=False))
        f.write("\n\n## Quadrant Mean Bias\n\n")
        f.write(pd.DataFrame(quadrant_stats).T.to_string())
        if elevation_stats:
            f.write("\n\n## Elevation-Binned Mean Bias\n\n")
            f.write(pd.DataFrame(elevation_stats).T.to_string())
        f.write("\n\n## Observations\n\n")
        f.write("- [ ] Identify systematic wind biases (e.g. consistently too strong/weak).\n")
        f.write("- [ ] Check temperature gradients vs elevation if DEM was provided.\n")
        f.write("- [ ] Note any temporal drift in errors (RMSE vs lead time).\n")

        LOGGER.info("Analysis complete. Results in: %s", run_dir)


def main():
    parser = argparse.ArgumentParser(description="Analyze weather forecast bias against truth.")
    parser.add_argument("--forecast-nc", type=Path, required=True, help="Path to ingested forecast NetCDF.")
    parser.add_argument("--truth-nc", type=str, required=True, help="Path or glob for truth/reanalysis NetCDF.")
    parser.add_argument("--variables", type=str, help="Comma-separated mapping of truth variables, e.g. u10=u10_era,t2m=t2m_era")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "weather_bias", help="Root directory for reports.")
    parser.add_argument("--dem-path", type=Path, help="Optional path to DEM for elevation-based analysis.")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), help="Optional bbox override.")

    args = parser.parse_args()
    
    try:
        run_analysis(args)
    except Exception as e:
        LOGGER.exception("Analysis failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

