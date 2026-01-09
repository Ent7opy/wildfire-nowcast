# Spread Hindcast Builder

The `hindcast_builder.py` pipeline reconstructs historical fire situations and runs a spread model to produce **predicted vs observed** pairs on the analysis grid. This dataset is used for calibration and evaluation of the spread model.

## Pipeline Overview

1.  **Selection**: Samples historical reference times from `fire_detections` in a specified bbox and time range.
2.  **Grouping**: Contiguous samples are grouped into "events" to ensure representativeness.
3.  **Inputs**: For each reference time, it builds a `SpreadInputs` package (fires + weather + terrain) using `ml/spread_features.py`.
4.  **Prediction**: Runs the selected spread model (e.g., `HeuristicSpreadModelV0`) to get predicted probabilities (`y_pred`).
5.  **Observation**: Queries FIRMS detections around the target horizons to get binary fire presence grids (`y_obs`).
6.  **Artifacts**: Saves each case as an xarray-compatible NetCDF file and maintains an `index.json` manifest.

## Usage

Run the builder via `uv` or `make`:

```bash
# Using uv directly
uv run --project ml -m ml.spread.hindcast_builder --config configs/hindcast_smoke_grid_balkans_mvp.yaml

# Using make (if target is added)
make hindcast-build ARGS="--config configs/hindcast_smoke_grid_balkans_mvp.yaml"
```

## Dataset Structure

Each case is stored as a NetCDF file with the following structure:

### Dimensions
- `time`: The forecast horizon times (e.g., T+24h, T+48h, ...).
- `lat`: Analysis grid latitudes.
- `lon`: Analysis grid longitudes.

### Coordinates
- `time`: Absolute UTC timestamps.
- `lat` / `lon`: Cell-center coordinates.
- `lead_time_hours`: Integer hours from reference time (aligned with `time`).

### Variables
- `y_pred` (time, lat, lon): Predicted spread probabilities [0.0, 1.0].
- `y_obs` (time, lat, lon): Observed fire presence [0.0, 1.0].
- `fire_t0` (lat, lon): Binary fire mask at reference time (T=0).
- `slope` (lat, lon): Terrain slope in degrees.
- `aspect` (lat, lon): Terrain aspect in degrees.
- `u10` (time, lat, lon): Eastward wind component at 10m (if available).
- `v10` (time, lat, lon): Northward wind component at 10m (if available).

### Attributes
- `region`: The analysis region name.
- `bbox`: The AOI bounding box.
- `ref_time`: The forecast reference time (ISO format).
- `model`: The model used for prediction.

## Loading Data for Analysis

You can load a single case or combine them using `xarray`:

```python
import xarray as xr
import json

# Load manifest
with open("data/hindcasts/smoke_grid/run_.../index.json", "r") as f:
    manifest = json.load(f)

# Load a single case
case = xr.open_dataset(manifest["cases"][0]["path"])

# Example: Compare predicted vs observed at T+24h
y_pred_24 = case.y_pred.isel(time=0)
y_obs_24 = case.y_obs.isel(time=0)
```

## Calibration and bias correction (transparency)

Hindcasts are used for **evaluation and calibration**:
- **Calibration** uses hindcast `y_pred` vs `y_obs` to learn per-horizon calibration mappings (see `ml/calibration.py`).
- **Evaluation** scripts produce tables and plots so improvements are visible:
  - Spread calibration evaluation: `ml/eval_spread_calibration.py` (reliability diagrams, Brier, ECE)
  - Weather bias correction evaluation: `ml/eval_weather_bias_correction.py` (bias/MAE/RMSE + maps/time series)

For a high-level, non-magical explanation of both layers, see:
`docs/ml/calibration_and_weather_bias_correction.md`.

