# Weather Ingestion (GFS 0.25°)

This pipeline downloads NOAA NOMADS GFS 0.25° forecasts (0–72h, 3h steps), subsets to a configurable bounding box (snapped to the analysis grid), interpolates onto the canonical 0.01° EPSG:4326 grid, and stores the result as NetCDF for downstream ML and API usage.

## Data and Variables
- Model: `gfs_0p25` (public HTTP, no API key).
- Variables: `u10`, `v10`, `t2m`, `rh2m` (always), `tp` (optional precipitation).
- Coordinates: `time`, `lat`, `lon` plus `forecast_reference_time` and `lead_time_hours`. The saved NetCDF is on the shared 0.01° EPSG:4326 analysis grid with attributes `crs`, `cell_size_deg`, `origin_lat`, `origin_lon`, `n_lat`, `n_lon`.

## Storage Layout
```
data/weather/gfs_0p25/YYYY/MM/DD/HH/
  gfs_0p25_YYYYMMDDTHHZ_0-72h_<region>.nc
```
`region` is `global` or `bbox_<min_lon>_<min_lat>_<max_lon>_<max_lat>`.

## Metadata Table
`weather_runs` tracks each ingest run:
`id, model, run_time, horizon_hours, step_hours, bbox_min_lon/lat/max_lon/lat, file_format, storage_path, status, created_at, metadata`.

## Usage

### Standard ingestion (region or large AOI)
```
python -m ingest.weather_ingest --run-time 2025-12-06T00:00Z
```
Options: `--bbox min_lon min_lat max_lon max_lat`, `--horizon-hours`, `--step-hours`, `--include-precip`.

### Small AOI optimization (patch mode)
For small bounding boxes (<50km x 50km) or just-in-time forecasts, use `--patch-mode` to reduce download and processing time:
```
python -m ingest.weather_ingest --run-time 2025-12-06T00:00Z --bbox 20.0 40.0 20.1 40.1 --patch-mode
```
Or via make:
```
make ingest-weather ARGS="--run-time 2025-12-06T00:00Z --bbox 20.0 40.0 20.1 40.1 --patch-mode"
```

**Patch mode optimizations:**
- Reduces forecast horizon to 24h (vs 72h default)
- Uses 6h temporal steps (vs 3h default)
- Skips precipitation variable processing
- Adds 0.5° spatial margin during download for interpolation coverage
- Final output is cropped to original bbox

**When to use:**
- Small areas of interest (<50km x 50km)
- Just-in-time forecast scenarios requiring low latency
- Cases where 24h horizon is sufficient

**Target performance:** <10s for 10km x 10km AOI.

Example read:
```python
import xarray as xr

ds = xr.open_dataset(
    "data/weather/gfs_0p25/2025/12/06/00/gfs_0p25_20251206T00Z_0-72h_global.nc"
)
print(ds)
print(ds["u10"].isel(time=0))
```

> Note: `cfgrib` requires ecCodes installed on the host (system package).

