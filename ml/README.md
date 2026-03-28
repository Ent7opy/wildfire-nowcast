# ML Pipelines

## Denoiser v2 (current standard)

XGBoost classifier (exported to ONNX) that distinguishes real fires from noise in FIRMS detections. v2 is event-based: detections are grouped into fire events before labeling.

### Pipeline steps

```bash
# 1. Eventize detections into fire events
make denoiser-eventize ARGS="--batch-id ..."

# 2. Label events
make denoiser-label-v2 ARGS="--start ... --end ..."

# 3. Build snapshot (feature table)
make denoiser-snapshot-v2 ARGS="--bbox ... --start ... --end ... --version ..."

# 4. Train
make denoiser-train-v2 CONFIG=configs/denoiser_train_v2.yaml

# 5. Evaluate
make denoiser-eval-v2 MODEL_RUN=models/denoiser_v2/<run_id> SNAPSHOT=... OUT=reports/denoiser_v2/<run_id>
```

### Artifacts (per run under `models/denoiser_v2/<run_id>/`)

- `model.onnx` — inference model
- `metrics.json` — ROC-AUC, PR-AUC, event recall/precision, F1
- `gate_report.json` — promotion gate result (`"pass": true/false`)
- `runtime_contract.json` — feature schema and threshold profile for inference
- `config_resolved.yaml` — reproducibility record

### Promotion gate requirements

- `gate_report.json` field `"pass": true`
- `coverage_data_freshness.fresh: true`
- Threshold profile must match `DENOISER_THRESHOLD_PROFILE` in env

### Full pipeline (train → eval → register → promote)

```bash
make train-denoiser TRAIN_DENOISER_PIPELINE=v2
```

---

## Spread Forecasting v2

Probabilistic 24-72 h fire spread model. 18-channel feature tensor; requires a gate report (champion-challenger eval) before promotion.

### Pipeline steps

```bash
make train-spread TRAIN_SPREAD_PIPELINE=v2
```

Or step-by-step (hindcast build → eval → register → promote via `ml/spread/`).

### Canonical feature channels (v2/v3, 18 channels, order fixed)

```
fire_t0, fire_t-6h, fire_t-12h,
u10, v10, t2m, rh2m, precip_24h,
slope_deg, aspect_sin, aspect_cos, elevation_m, ruggedness, tpi,
ndvi, lfmc, dfmc,
region_id_embedding_input
```

### Gate requirements

See `docs/spread_gate_requirements.md` for full spec (hard stops, stage warnings, science debt register, metric thresholds, and the `science_grade` checklist).

---

## Weather Bias Analysis

Quantify systematic biases in GFS weather fields vs reanalysis (e.g., ERA5):

```bash
make weather-bias ARGS="--forecast-nc data/weather/... --truth-nc path/to/era5.nc"
```

Outputs to `reports/weather_bias/<timestamp>/`: `summary.csv`, `summary.json`, bias maps.

## Weather Bias Correction

Train a per-variable affine corrector (`truth ≈ α + β·forecast`) for use in spread inference:

```bash
python -m ml.train_weather_bias_corrector \
  --forecast-nc path/to/forecast.nc \
  --truth-nc path/to/truth.nc \
  --out-dir models/weather_bias_corrector
```

Pass corrector at inference via `WEATHER_BIAS_CORRECTOR_PATH` env var or `weather_bias_corrector_path` arg to `build_spread_inputs`.

## Spread Calibration Eval

```bash
python -m ml.eval_spread_calibration \
  --hindcast-run-dir path/to/hindcast_run \
  --calibrator-run-dir models/spread_calibration/<run_id>
```

Outputs per-horizon Brier score, ECE, and reliability diagrams to `reports/spread_calibration_eval/`.
