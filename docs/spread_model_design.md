# Spread Forecasting Models: v0 heuristic + v1 baseline (Design, Contract, and Limitations)

This document exists so the spread model is **not misused or oversold**.

It has two roles:
1. **Contract**: the stable interface (`ml/spread/contract.py`) that callers and UI can rely on.
2. **Reality check**: what the currently implemented models actually do (v0 heuristic, v1 learned baseline), what they assume, and what they do *not* model.

## 0. What is implemented today (and what “probability” means here)

### 0.1 Implemented models
- **Heuristic v0 (default runtime model)**: `ml/spread/heuristic_v0.py`.
  - Selected by default in `ml/spread/service.py::run_spread_forecast`.
  - Produces a relative probability footprint by convolving a fire mask with a wind/terrain-biased kernel, then normalizing by the maximum value.
- **Learned v1 baseline (implemented but not the default)**: training `ml/train_spread_v1.py`, inference `ml/spread/learned_v1.py`.
  - Trains per-horizon `HistGradientBoostingClassifier` models on a hindcast dataset built by `ml/spread/hindcast_dataset.py`.

### 0.2 What v0 probabilities are (and are not)
The v0 output is bounded to \([0, 1]\) and looks like a probability field, but **it is not calibrated** and it is **normalized per-horizon**.

Concretely, v0 computes an unnormalized score by convolution and then divides by the maximum value in the AOI for that horizon (`ml/spread/heuristic_v0.py`). That means:
- **Good use**: as a **relative spread footprint** / “where is more likely than elsewhere” overlay.
- **Bad use**: interpreting 0.7 as “70% chance of fire presence” in an absolute, calibrated sense.
- **Important**: because of per-horizon max-normalization, the same physical situation can yield different absolute values depending on AOI/window shape and kernel support.

## 1. Goal & Horizons (contract-level intent)

The spread model predicts the **probability of fire presence** at future time horizons given the current fire state, weather forecast, and static terrain features.

- **Default horizons in code**: T+24h, T+48h, T+72h (`ml/spread/contract.py::DEFAULT_HORIZONS_HOURS`).
- **Resolution**: region grid cell size (commonly 0.01° on `EPSG:4326`; see `api.core.grid` / `get_region_grid_spec`).
- **Granularity**: model runs per AOI window (bbox), typically a clipped subset of the region grid.

## 2. Core Abstraction (`SpreadModel`) — contract

The model is a `Protocol` in `ml/spread/contract.py`, enabling different implementations to be swapped without changing the caller.

### 2.1 Inputs (`SpreadModelInput`)

| Input | Type | Where it comes from in this repo |
| :--- | :--- | :--- |
| `grid` | `GridSpec` | `api.fires.service.get_region_grid_spec` |
| `window` | `GridWindow` | `api.core.grid.get_grid_window_for_bbox(..., clip=True)` |
| `active_fires` | `FireHeatmapWindow` | `api.fires.service.get_fire_cells_heatmap` (assembled via `ml/spread_features.py::build_spread_inputs`) |
| `weather_cube` | `xr.Dataset` | Loaded/aligned by `ml/spread_features.py::_load_weather_cube` (from `weather_runs`, with calm fallback) |
| `terrain` | `TerrainWindow` | `api.terrain.window.load_terrain_window` (assembled via `ml/spread_features.py::build_spread_inputs`) |
| `forecast_reference_time` | `datetime` | Request field (`ml/spread/service.py::SpreadForecastRequest`) |
| `horizons_hours` | `Sequence[int]` | Request field, defaults to `DEFAULT_HORIZONS_HOURS` |

#### Inputs used by v0 specifically
v0 uses:
- **Fires**: a binary/thresholded mask from `active_fires.heatmap` (`ml/spread/heuristic_v0.py`).
- **Weather**: **window-mean** `u10` and `v10`, selected at the nearest weather time to each horizon (`ml/spread/heuristic_v0.py`).
- **Terrain (optional)**: if enabled, v0 uses **window-mean** slope/aspect to bias spread upslope (`HeuristicSpreadV0Config.enable_slope_bias` in `ml/spread/heuristic_v0.py`).
- **Masks**: applies `terrain.valid_data_mask` and `terrain.aoi_mask` if present.

### 2.2 Outputs (`SpreadForecast`)

The primary output is an `xarray.DataArray`:
- **Dimensions**: `("time", "lat", "lon")`.
- **Coordinates**:
  - `time`: reference time + horizon.
  - `lat` / `lon`: cell-center coordinates matching the input window.
  - `lead_time_hours`: optional but recommended; aligned with the `time` dimension.
- **Values**: \([0.0, 1.0]\) floats.
- **Validation**: enforced by `SpreadForecast.validate()` in `ml/spread/contract.py`.

## 3. AOI & Cluster Handling (today)

- **AOI**: bbox → clipped window via `get_grid_window_for_bbox(..., clip=True)` (`ml/spread_features.py`).
- **Cluster support**: `SpreadForecastRequest.fire_cluster_id` exists but is currently **not implemented** (`ml/spread/service.py` raises `NotImplementedError`).
- **Performance guardrail**: synchronous forecasting rejects AOIs above `MAX_AOI_CELLS` (`ml/spread/service.py`).

## 4. How heuristic v0 derives probabilities

This section is intentionally grounded in `ml/spread/heuristic_v0.py`.

### 4.1 Fire input → source mask
v0 thresholds the active fire heatmap:
- `fire_mask = (inputs.active_fires.heatmap > fire_threshold)` where `fire_threshold` is configurable (`HeuristicSpreadV0Config.fire_threshold`).
- If there are **no active fire cells**, v0 returns all-zero probability grids.

### 4.2 Kernel generation (distance + wind + optional slope)
For each horizon \(h\) (hours), v0 constructs a 2D kernel over (lat, lon) pixels:
- **Base spread distance**: `base_dist = base_spread_km_h * h`.
- **Wind**:
  - Uses window-mean `u10`, `v10` from `weather_cube`.
  - Wind speed: \(\sqrt{u^2 + v^2}\).
  - Shapes the kernel by:
    - **Elongation**: downwind distances are “easier” than upwind using `wind_elongation_factor`.
    - **Wind speed scaling**: expands the footprint with wind speed (`wind_influence_km_h_per_ms`).
- **Optional slope bias** (if enabled): computes window-mean slope and a circular mean of aspect; then biases spread toward **upslope** direction (aspect + 180°), scaling strength with slope (`slope_influence`, `slope_reference_deg`).
- **Distance decay**: kernel values decay exponentially with effective distance:

  `kernel = exp(-eff_dist / (base_dist + distance_decay_km))`

  where `eff_dist` incorporates wind and (optional) slope bias.

### 4.3 Convolution + normalization
- v0 convolves the `fire_mask` with the kernel using `scipy.signal.fftconvolve(..., mode="same")`.
- It normalizes by the maximum value in the AOI for that horizon (if max > 0).
- It clamps to \([0,1]\) and applies `valid_data_mask` and `aoi_mask` if present.

## 5. Known limitations (v0 heuristic)

These are direct implications of the current implementation.

- **Not a physical fire behavior model**: no fuels, no moisture model, no energy balance, no fireline intensity.
- **No spotting / ember transport**: cannot create disconnected ignitions.
- **No barriers**: rivers/roads/firebreaks/urban areas are not modeled as discontinuities.
- **Wind is spatially collapsed**: v0 uses **mean wind over the AOI**, so it cannot represent spatial wind gradients.
- **Terrain is spatially collapsed (if enabled)**: slope/aspect bias uses **window-mean** slope/aspect, not per-cell effects.
- **No stepwise dynamics**: each horizon is computed independently; there is no iterative propagation or coupling between horizons.
- **Per-horizon normalization**: useful as a footprint, but prevents interpreting values as calibrated probabilities.
- **Grid-distance approximation**: converts degrees to km using mean latitude.

## 6. Expected calling pattern (repo entrypoints)

### 6.1 End-to-end orchestration
- Use `ml/spread/service.py::run_spread_forecast` for the standard “assemble inputs → run model → validate output” path.
- Inputs are assembled by `ml/spread_features.py::build_spread_inputs`.

### 6.2 Persistence of products (rasters/contours)
- The CLI `ingest/spread_forecast.py` saves per-horizon probability rasters (GeoTIFF/COG) and generates thresholded polygon contours for storage.

## 7. Learned v1 baseline (implemented): training data, features, metrics

This is a **baseline** learned model; treat it as experimental unless evaluated and calibrated for a region.

### 7.1 Training data (hindcast)
`ml/spread/hindcast_dataset.py` builds a dataset by:
1. Sampling reference times with sufficient detections in a bbox (`sample_fire_reference_times`, querying `fire_detections`).
2. Building aligned inputs via `ml/spread_features.py::build_spread_inputs`.
3. Defining labels by querying future fire presence in a small time window around each target horizon (`get_fire_cells_heatmap(..., mode="presence")` for \(T+h \pm 3\) hours).
4. Flattening each grid cell into a row (tabular dataset).
5. Negative sampling to handle sparsity (while always keeping cells with fire at \(T=0\)).

### 7.2 Features (current baseline)
The baseline feature set comes from the hindcast builder and training config (`configs/spread_train_v1.yaml`). Today it includes:
- `fire_t0`
- `slope_deg`, `aspect_sin`, `aspect_cos`
- `u10`, `v10`, `wind_speed`

Optional columns exist in code (`elevation_m`, `t2m`, `rh2m`) but are not included in the default YAML.

### 7.3 Model + metrics
`ml/train_spread_v1.py` trains one `HistGradientBoostingClassifier` per horizon and writes artifacts to `models/spread_v1/<run_id>/`.

It reports:
- **ROC-AUC** and **PR-AUC** (when defined), and
- thresholded metrics including **precision/recall/F1** and a simple mask **IoU (Jaccard)**.

### 7.4 Inference
`ml/spread/learned_v1.py::LearnedSpreadModelV1`:
- loads `model.pkl` (mapping horizon → classifier) and `feature_list.json`,
- builds per-horizon tabular features from `SpreadModelInput`,
- returns an `xr.DataArray` with the same contract dims/coords, and
- applies `valid_data_mask`/`aoi_mask` similarly to v0.

## 8. Future improvements

### 8.1 More physical modeling
- Stepwise propagation (time stepping) instead of independent per-horizon kernels.
- Incorporate rate-of-spread heuristics from fire behavior literature as features or constraints.
- Explicit handling of barriers and suppression (where data exists).

### 8.2 Better integration with fuels and land cover
- Add fuel/vegetation/land-cover layers and derive fuel-dependent spread modifiers.
- Add dryness proxies (precipitation history, fuel moisture indices) when available.
- Region-specific calibration and validation.

### 8.3 More advanced ML approaches
- Spatial models (CNN/UNet-style) operating directly on rasters.
- Sequence models that consume weather time series rather than a single horizon slice.
- Uncertainty: ensembles, quantile regression, or Bayesian-ish approximations.
- Probability calibration + reliability curves per region/horizon.
- Better evaluation: spatial metrics (distance-to-fireline), event-based metrics, and calibration metrics (ECE/Brier).

## 9. Calibration and weather bias correction (not magic)

This repo includes two optional post-processing layers:
- **Weather bias correction**: adjusts input weather fields to reduce systematic forecast error vs truth.
- **Probability calibration**: adjusts output probabilities to better match observed frequencies in hindcast.

These layers can improve operational usefulness, but they are not a substitute for:
- better model structure/features,
- physical fire behavior modeling,
- rigorous region/season evaluation.

Key points:
- Calibration cannot “invent skill”; it only remaps scores.
- Weather bias correction here is a global affine transform per variable; it will not fix spatially varying biases.
- Heuristic v0 probabilities are normalized per AOI per horizon, so “probability” can be AOI-dependent even after calibration.

See `docs/ml/calibration_and_weather_bias_correction.md` for details, artifacts, failure modes, and evaluation recipes.
