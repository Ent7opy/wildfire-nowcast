# Machine Learning for Wildfire Nowcast

This directory contains the machine learning pipelines for the Wildfire Nowcast project.

## Denoiser Classifier

The denoiser is a tabular classifier that distinguishes between real fires and noise (e.g., industrial heat sources, sensor noise) in FIRMS detections.

### Training the Baseline Model

To train the baseline denoiser model, use the `train_denoiser.py` script.

#### Prerequisites

1.  **Data Snapshot**: You need a Parquet snapshot produced by the labeling/feature pipeline. Snapshots are typically stored in `data/denoiser/snapshots/run_<timestamp>/`.
2.  **Configuration**: Create or modify a configuration file (e.g., `configs/denoiser_train_v1.yaml`).

#### Usage

```bash
# Using make (from repo root)
make denoiser-train CONFIG=configs/denoiser_train_v1.yaml

# Or using uv directly
python -m ml.train_denoiser --config configs/denoiser_train_v1.yaml
```

#### Artifacts

Each training run produces versioned artifacts in the output directory specified in the config (default: `models/denoiser_v1/`).

Artifacts saved:
- `model.pkl`: The trained model (using `joblib`).
- `metadata.json`: Information about the training run, including configuration, feature list, and training environment.
- `metrics.json`: ROC-AUC, PR-AUC, and threshold-based metrics (Precision, Recall, F1, Confusion Matrix).
- `feature_list.json`: The list of features used for training.
- `config_resolved.yaml`: The resolved config used for the run (for reproducibility).

Notes:
- AUC metrics are only defined when the eval split contains **both** classes. If a time-based split produces a single-class eval set, training will fall back to a reproducible stratified random split (configurable) so ROC-AUC/PR-AUC are meaningful.

### Project Structure

- `ml/denoiser/`: Feature engineering and labeling logic.
- `ml/train_denoiser.py`: Training entrypoint.
- `ml/weather_bias_analysis.py`: Systematic bias analysis script.
- `configs/`: Training configuration files.
- `models/`: Trained model artifacts.
- `reports/`: Analysis outputs and reports.

## Weather Bias Analysis

Quantify systematic biases in weather fields (wind, temperature, humidity) by comparing forecasts with reanalysis or ground truth datasets (e.g., ERA5).

### Usage

```bash
# Using make (from repo root)
make weather-bias ARGS="--forecast-nc data/weather/gfs_0p25/2025/12/06/12/gfs_0p25_20251206T12Z_0-24h_bbox_5.0_35.0_20.0_47.0.nc --truth-nc path/to/YOUR_ERA5_FILE.nc"

# Or using uv directly
uv run --project ml -m ml.weather_bias_analysis \
    --forecast-nc data/weather/gfs_0p25/2025/12/06/12/gfs_0p25_20251206T12Z_0-24h_bbox_5.0_35.0_20.0_47.0.nc \
    --truth-nc path/to/YOUR_ERA5_FILE.nc \
    --out-dir reports/weather_bias
```

> **Note**: `path/to/YOUR_ERA5_FILE.nc` is a placeholder. You must provide a path to a real NetCDF file containing reanalysis truth.

Options:
- `--variables`: Comma-separated mapping if truth variable names differ (e.g., `u10=u10_era,t2m=t2m_truth`).
- `--dem-path`: Optional path to a DEM GeoTIFF for elevation-stratified bias analysis.

### Artifacts

Reports are saved to `reports/weather_bias/<timestamp>/`:
- `summary.csv`: Per-variable bias, MAE, and RMSE.
- `summary.json`: Comprehensive metrics including quadrant and elevation-binned stats.
- `plots/`: Mean bias maps and time series plots.
- `notes.md`: Template for documenting findings and observations.
- `metadata.json` / `config_resolved.yaml`: Reproducibility metadata.

## Weather Bias Correction (for Spread Inference)

For spread inference, you can optionally apply a lightweight bias corrector to the weather cube.
The corrector is a **per-variable affine transform** (truth \(\approx \alpha + \beta \cdot forecast\)) fit on aligned forecast vs truth data.

### Training

```bash
python -m ml.train_weather_bias_corrector \
  --forecast-nc path/to/forecast.nc \
  --truth-nc path/to/truth.nc \
  --out-dir models/weather_bias_corrector
```

This writes a run directory containing:
- `weather_bias_corrector.json`: correction parameters (loadable in inference).
- `metrics.json`: before/after validation metrics on a held-out time slice.

### Using in spread code

`ml.spread_features._load_weather_cube` will apply the corrector automatically if you provide a path:
- pass `weather_bias_corrector_path=Path(".../weather_bias_corrector.json")` to `build_spread_inputs`, or
- set `WEATHER_BIAS_CORRECTOR_PATH=/abs/path/to/weather_bias_corrector.json` in the environment.

## Evaluation: Calibration & Weather Bias Correction

### Calibration evaluation (spread reliability)

Given a hindcast run (predicted vs observed grids) and a calibrator run dir, generate a report with:
- per-horizon **Brier score** and **ECE**
- **reliability diagrams** (raw vs calibrated)

```bash
python -m ml.eval_spread_calibration \
  --hindcast-run-dir path/to/hindcast_run \
  --calibrator-run-dir models/spread_calibration/<run_id>
```

Outputs are written to `reports/spread_calibration_eval/<timestamp>_<run_id>/`.

### Weather bias correction evaluation (systematic error reduction)

Compare forecast vs truth (raw and corrected) and generate:
- per-variable **bias/MAE/RMSE** tables (raw vs corrected)
- mean bias maps + bias reduction maps
- domain-mean bias time series overlays

```bash
python -m ml.eval_weather_bias_correction \
  --forecast-nc path/to/forecast.nc \
  --truth-nc path/to/truth.nc \
  --corrector-json models/weather_bias_corrector/<run_id>/weather_bias_corrector.json
```

Outputs are written to `reports/weather_bias_correction_eval/<timestamp>/`.

