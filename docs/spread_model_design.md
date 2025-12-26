# Spread Forecasting Model (v1): Design & Contract

This document defines the contract for the **Wildfire Spread Forecasting Model (24–72h v1)**. It specifies the inputs, outputs, and calling patterns to ensure consistency between ML development, API serving, and UI visualization.

## 1. Goal & Horizons

The spread model predicts the **probability of fire presence** at future time horizons given the current fire state, weather forecast, and static terrain features.

- **MVP Horizons (v1)**: **T+24h, T+48h, T+72h**.
- **Optional Horizons (interface-supported)**: T+6h, T+12h.
- **Resolution**: 0.01° (~1 km) on the canonical analysis grid (`EPSG:4326`).

## 2. Core Abstraction (`SpreadModel`)

The model is defined as a `Protocol` in `ml/spread/contract.py`. This allows different implementations (e.g., cell-based, convolutional, or ensemble) to be swapped without changing the calling code.

### 2.1 Inputs (`SpreadModelInput`)

| Input | Type | Source / Pattern |
| :--- | :--- | :--- |
| `grid` | `GridSpec` | `api.core.grid.GridSpec` |
| `window` | `GridWindow` | `api.core.grid.GridWindow` (the analysis bbox) |
| `active_fires` | `FireHeatmapWindow` | `api.fires.service.get_fire_cells_heatmap` (heatmap of current detections) |
| `weather_cube` | `xr.Dataset` | `data/weather/...` (GFS 0.25° interpolated to analysis grid) |
| `terrain` | `TerrainWindow` | `api.terrain.window.load_terrain_window` (slope, aspect, elevation) |
| `horizons` | `Sequence[int]` | e.g. `[24, 48, 72]` hours |

### 2.2 Outputs (`SpreadForecast`)

The primary output is an **`xarray.DataArray`** with the following specification:

- **Dimensions**: `("time", "lat", "lon")`.
    - This repo’s canonical raster naming is `lat/lon` (see `api.core.grid`). If you think of a generic ML tensor `(time, y, x)`, then `y ↔ lat` and `x ↔ lon`, and the canonical index order is `(i, j) = (lat_index, lon_index)`.
- **Coordinates**:
    - `time`: `datetime` objects (Reference Time + Horizon).
    - `lat` / `lon`: Cell-center coordinates matching the input window.
    - `lead_time_hours`: (optional) Coordinate mapping time to horizon hours.
- **Values**: Probability of fire presence in range `[0.0, 1.0]`.
- **Data Type**: `float32` (preferred) or `float64`.

## 3. AOI & Cluster Handling

- **Granularity**: The model runs per **AOI window** (bounding box).
- **Clusters**: If multiple fire clusters are active, the calling service (e.g., a background worker) determines the appropriate window size to cover the cluster and its expected spread area.
- **Clipping**: Use `api.terrain.window.load_terrain_for_aoi` to obtain a polygon-clipped mask if the user specifies a non-rectangular AOI. The model may use this mask to zero out probabilities or weight results.

## 4. Expected Calling Pattern (API/Worker)

```python
from datetime import datetime, timedelta, timezone
import xarray as xr

from ml.spread import SpreadModelInput
from api.core.grid import get_grid_window_for_bbox
from api.fires.service import get_fire_cells_heatmap, get_region_grid_spec
from api.terrain.window import load_terrain_window

# 1. Define window
bbox = (5.1, 35.4, 6.0, 36.0) # (min_lon, min_lat, max_lon, max_lat)
region = "smoke_grid"
now = datetime.now(timezone.utc)
start_time = now - timedelta(hours=24)

# 2. Gather Features
grid = get_region_grid_spec(region)
window = get_grid_window_for_bbox(grid, bbox)

fires = get_fire_cells_heatmap(region, bbox, start_time, now)
terrain = load_terrain_window(region, bbox, include_dem=True)

# Weather loading is intentionally left to the caller (worker/service), since the
# API package does not currently expose a single "latest weather cube" helper.
# The contract requires an xr.Dataset with coords/dims including: time, lat, lon.
# Example sketch:
# weather = xr.open_dataset(path_to_latest_weather_netcdf)
# weather = weather.sel(lat=window.lat, lon=window.lon, method="nearest")
# weather = weather.sel(time=desired_times)
weather = xr.Dataset(coords={"time": [], "lat": window.lat, "lon": window.lon})

# 3. Predict
inputs = SpreadModelInput(
    grid=grid,
    window=window,
    active_fires=fires,
    weather_cube=weather,
    terrain=terrain,
    forecast_reference_time=now,
    horizons_hours=[24, 48, 72]
)

forecast = model.predict(inputs)
forecast.validate()

# 4. Use
# probabilities = forecast.probabilities  # xr.DataArray (3, H, W)
```

## 5. Storage & Serving

- **Raster**: Forecast probability grids should be stored as **Cloud-Optimized GeoTIFFs (COGs)** or **Zarr** in `data/forecasts/`.
- **Vector**: Iso-probability contours (isochrones) can be extracted using `skimage.measure.find_contours` or similar and stored in PostGIS for fast UI rendering.

## 6. Training Pipeline (v1)

The training pipeline for learned spread models uses a **hindcast** approach, where historical FIRMS detections are used to build both features (at T=0) and labels (at T+24/48/72h).

### 6.1 Building the Dataset

The script `ml/spread/hindcast_dataset.py` assembles a tabular dataset by:
1. Sampling historical reference times with sufficient fire activity.
2. Building `SpreadInputs` for each time using the canonical feature pipeline.
3. Extracting future fire presence from the DB as the target label.
4. Flattening grid cells into rows with spatial and weather features.
5. Applying negative sampling to manage the extreme sparsity of fire spread.

### 6.2 Training

Use the `ml/train_spread_v1.py` script with a YAML config:

```bash
uv run --project ml -m ml.train_spread_v1 --config configs/spread_train_v1.yaml
```

This trains an ensemble of `HistGradientBoostingClassifier` models (one per horizon) and saves them to `models/spread_v1/<run_id>/`.

### 6.3 Inference

The `LearnedSpreadModelV1` class (`ml/spread/learned_v1.py`) implements the `SpreadModel` protocol and can be initialized from a trained run directory to produce forecasts.

