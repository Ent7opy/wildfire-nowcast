## Calibration & Weather Bias Correction (Transparent, Non-Magical)

This project uses two “post-processing” layers:

- **Weather bias correction**: adjusts input weather fields (e.g. GFS) to reduce systematic error vs a truth source (e.g. ERA5).
- **Probability calibration**: adjusts output spread probabilities to better match observed frequencies in hindcast data.

These layers can improve usefulness, but they do **not** turn the system into a physical fire model, and they can fail silently if misapplied. This doc explains what they do and how to evaluate them.

---

## 1) Weather bias correction

### 1.1 What it does

Weather bias correction applies a simple per-variable affine transform:

\[
\text{corrected} = \alpha + \beta \cdot \text{forecast}
\]

In code this is `ml/weather_bias_correction.py::WeatherBiasCorrector`. It is intentionally lightweight:

- **Global fit**: one \(\alpha,\beta\) pair per variable (not per grid cell).
- **Fast inference**: just multiply/add per variable.
- **Serializable**: stored in JSON (`weather_bias_corrector.json`).
- **Safety clamp**: `rh2m` is clamped to \([0,100]\) after correction.

### 1.2 What it does *not* do

- **Not a weather model**: it cannot fix missing physics or errors that vary strongly with location, elevation, coastlines, etc.
- **Not “downscaling”**: it does not add spatial detail; it only rescales/shifts existing fields.
- **Not guaranteed to generalize**: if trained on a narrow time window/season, it may degrade performance outside that regime.

### 1.3 Training artifacts (how to reproduce)

Train a corrector from aligned forecast vs truth NetCDFs:

```bash
python -m ml.train_weather_bias_corrector \
  --forecast-nc path/to/forecast.nc \
  --truth-nc path/to/truth.nc \
  --out-dir models/weather_bias_corrector
```

Outputs:
- `weather_bias_corrector.json`: correction parameters used in inference
- `metrics.json`: raw vs corrected validation metrics on a time split
- `config_resolved.yaml`: exact args used

### 1.4 Evaluation (how to tell if it helped)

Run the evaluation report:

```bash
python -m ml.eval_weather_bias_correction \
  --forecast-nc path/to/forecast.nc \
  --truth-nc path/to/truth.nc \
  --corrector-json models/weather_bias_corrector/<run_id>/weather_bias_corrector.json
```

Outputs include:
- `summary.csv`: per-variable bias/MAE/RMSE (raw vs corrected) + reductions
- `plots/bias_map_raw_*.png`, `plots/bias_map_corrected_*.png`, `plots/bias_map_reduction_*.png`
- `plots/bias_ts_raw_vs_corrected_*.png`

Interpretation guide:
- If **mean bias maps move toward 0** and `rmse_reduction`/`mae_reduction` are positive, correction is helping.
- If some variables worsen, the corrector may be mismatched (season/region) or overfit.

---

## 2) Spread probability calibration

### 2.1 What it does

Calibration maps raw probabilities \(\hat{p}\) to calibrated probabilities \(p'\) so that, approximately:

> among cells with predicted probability ~0.3, about 30% should be observed positive (in hindcast).

In code this is `ml/calibration.py::SpreadProbabilityCalibrator`:
- **Per-horizon** calibration models (e.g. T+24, T+48, T+72).
- Default method is **isotonic regression** (non-parametric monotone mapping).
- Stored in a run directory containing `calibrator.pkl` + metadata.

### 2.2 What it does *not* do

- **Not a model improvement**: it cannot fix bad spatial structure or missing features. It only remaps output scores.
- **Not universal**: calibration learned on one region or data regime may not generalize to another.
- **Important for heuristic v0**: v0 outputs are **normalized per AOI per horizon** (see `docs/spread_model_design.md`). This means:
  - calibration can still produce more “reliable-looking” probabilities on similar AOIs,
  - but the mapping may shift if AOI size/shape changes significantly.

### 2.3 Training artifacts (how to reproduce)

Calibration is trained from a **hindcast run** (predicted vs observed grids):

```bash
python -m ml.calibration --hindcast_run path/to/hindcast_run
```

This writes a calibrator run dir under `models/spread_calibration/<timestamp>_<git_sha>/` containing:
- `calibrator.pkl`: per-horizon calibration models
- `metadata.json`: provenance (git SHA, horizons, etc.)
- `metrics.json`: Brier improvement and reliability curves (during training)

### 2.4 Evaluation (how to tell if it helped)

Given a hindcast run and a calibrator run:

```bash
python -m ml.eval_spread_calibration \
  --hindcast-run-dir path/to/hindcast_run \
  --calibrator-run-dir models/spread_calibration/<run_id>
```

Outputs include:
- `summary.csv`: per-horizon **Brier score** and **ECE** (raw vs calibrated)
- `plots/reliability_h*.png`: reliability diagrams (raw vs calibrated)

Interpretation guide:
- Reliability curves **closer to the diagonal** indicate better calibration.
- Positive `brier_improvement` / `ece_improvement` suggests calibration helped on that hindcast distribution.

---

## 3) Operational inference: how artifacts are applied (and how it fails safely)

### 3.1 Bias correction application

At runtime, weather data is loaded by `ml/spread_features.py::_load_weather_cube`.
If a corrector exists, it is applied to the `weather_cube` and the dataset is tagged:

- `weather_cube.attrs["weather_bias_corrected"] = True`
- `weather_cube.attrs["weather_bias_corrector_path"] = "..."`

If the corrector file is missing or fails to load:
- The system logs a **warning/error** and continues with **uncorrected** weather.

### 3.2 Calibration application

At runtime, the spread forecast orchestration (`ml/spread/service.py::run_spread_forecast`) applies calibration **by default** when possible:

- If the model already carries a valid `SpreadProbabilityCalibrator`, it uses that.
- Otherwise it tries to load an operational calibrator and applies it per horizon.

If calibration is missing:
- The system logs a **warning** and returns **raw** probabilities.
If calibration exists but lacks some horizons:
- The system logs a **warning** and returns raw probabilities for those horizons.

### 3.3 Provenance and “not magic”

The service annotates the output `SpreadForecast.probabilities.attrs` so downstream systems can store/inspect:
- whether bias correction was applied,
- whether calibration was applied,
- which run IDs/paths were used.

This is intentionally explicit so downstream consumers don’t assume the results are “magical” or always calibrated.

