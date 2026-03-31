# Task: Ignition Probability — ML model

**Location:** `ml/ignition/`, `ml/spread_features.py` (reference), `api/` (runtime contract)
**Impact:** High — the core scientific component; determines the quality of everything the user sees
**Maturity target:** `mvp_operational`
**Depends on:** Task 16 (drought index and lightning proxy must be ingested before features can be extracted)

## Problem

The system currently has no model for ignition probability. The existing risk grid (`api/risk/`) is a rule-based heuristic (land cover × weather multiplier) — it does not model ignition likelihood, and it does not use drought index, time-since-last-burn, or lightning signals. A trained probability model is needed to produce calibrated, spatially specific ignition estimates that update with GFS forecast cycles.

## Proposed Solution

Build an ignition probability model following the conventions established by the denoiser v2 pipeline:
- XGBoost binary classifier (or calibrated regressor) producing a probability in [0, 1]
- Exported to ONNX for runtime inference
- Registered via the model registry (`make model-register`)
- Gated by a `gate_report.json` before promotion

### Feature set

All features should be computed per grid cell (matching the forecast grid resolution, ~0.25° consistent with GFS):

| Feature | Source | Notes |
|---|---|---|
| `fuel_moisture` | `lfmc_ecland_ingest` | Already in system |
| `lulc_flammability` | `lulc_worldcover_ingest` | Already in system; encode as ordinal flammability class |
| `relative_humidity` | weather cube | Already in spread_features.py |
| `temperature_c` | weather cube | Already in spread_features.py |
| `wind_speed_kmh` | weather cube | Already in spread_features.py |
| `precip_last_7d_mm` | weather cube / archive | Sum precipitation over past 7 days |
| `drought_index` | `drought_ingest` (Task 16) | Weekly CDI/SMA value; treat as slow-moving background |
| `thunderstorm_active` | `lightning_proxy_ingest` (Task 16) | Boolean; 1 if active thunderstorm warning covers cell |
| `days_since_last_burn` | derived from `fire_events` + `fire_perimeters` | See below |

**Deriving `days_since_last_burn`:** For each grid cell centroid, query `fire_perimeters` and `fire_events` for the most recent confirmed fire event or perimeter that intersects or is within 5 km of the cell. Use `perimeter_date` or `fire_events.detected_at`. Cap at 3650 days (10 years) for cells with no recent fire history — long unburned areas have high accumulated fuel load. Cells burned within the last 12 months should have near-zero ignition probability regardless of other conditions (recently burned areas lack fuel).

### Training labels

Positive label (ignition = 1): grid cells where a new fire event was first detected (FIRMS `fire_detections` with `is_noise = False`) in a subsequent 24h window, with no prior detection at that cell in the preceding 48h (i.e., new ignition, not spread from an existing fire).

Negative label (ignition = 0): all other grid cells in the same time windows, sampled to balance class ratio. Use geographic stratification to avoid regional imbalance.

This is harder to validate than spread — ignitions that *don't* happen are the majority class. Use calibration (Platt scaling or isotonic regression post-processing) to ensure the output probabilities are meaningful, not just discriminative. The `ml/calibration.py` module already exists for this purpose.

### Snapshot pipeline

Create `ml/ignition/snapshot.py` (analogous to `ml/denoiser/snapshot_v2.py`) to:
1. Extract the training feature matrix from the database for a given bbox and date range
2. Construct positive/negative labels from fire detections
3. Write a parquet snapshot to `data/snapshots/ignition/`

Add a Makefile target: `make ignition-snapshot ARGS="--bbox ... --start ... --end ..."`

### Training pipeline

Create `ml/train_ignition.py`:
- Load snapshot parquet
- Train XGBoost classifier
- Apply calibration
- Evaluate: AUC-ROC, Brier score, calibration curve
- Export to ONNX
- Write `metrics.json` and `gate_report.json` to `models/ignition/<run_id>/`

Add Makefile targets: `make ignition-train CONFIG=configs/ignition_train.yaml`

### Runtime inference

Create `ml/ignition_inference.py` following the exact interface of `ml/denoiser_inference_v2.py`:
- Load ONNX model from the registered model path
- Validate the runtime contract (required features, expected input shape)
- Accept a feature matrix and return per-cell probabilities + categorical classification (`low` / `elevated` / `high` / `critical`)
- Categorical thresholds (configurable via env):
  - `low` < 0.25
  - `elevated` 0.25–0.50
  - `high` 0.50–0.75
  - `critical` ≥ 0.75

The runtime contract JSON must list all required features and their expected dtypes — the inference module must hard-stop (`BLOCKER`) if a required feature is missing at runtime, not silently substitute zeros.

### Model registration

Follow the existing model registry flow:
```
make ignition-train CONFIG=configs/ignition_train.yaml
make model-register FAMILY=ignition ARTIFACT=models/ignition/<run_id>/model.onnx METRICS=@models/ignition/<run_id>/metrics.json RUNTIME_CONTRACT=@models/ignition/<run_id>/contract.json
make model-promote FAMILY=ignition MODEL_ID=...
```

Add `IGNITION_REQUIRED`, `IGNITION_MODEL_PATH`, and `IGNITION_THRESHOLD_PROFILE` env vars (see denoiser equivalents as reference). If `IGNITION_REQUIRED=true` (default) and no promoted model exists, the ignition API endpoint should return a 503 rather than silently returning zeros.

## Acceptance Criteria

- [ ] `ml/ignition/snapshot.py` extracts a correctly labelled feature matrix and writes a parquet snapshot
- [ ] `ml/train_ignition.py` trains, calibrates, and exports an ONNX model with AUC-ROC > 0.65 on held-out data
- [ ] `gate_report.json` is written with `"pass": true/false` based on the AUC-ROC gate (threshold configurable in the training config)
- [ ] `ml/ignition_inference.py` loads the ONNX model, validates the runtime contract, and returns per-cell probabilities + categorical classification
- [ ] Missing required features at runtime produce a hard BLOCKER, not silent zeros
- [ ] `days_since_last_burn` is correctly derived: cells burned within 12 months return near-zero probability regardless of other features
- [ ] Calibration is applied — output probabilities should be roughly calibrated (check calibration curve in eval report)
- [ ] Model can be registered and promoted via the existing model registry
- [ ] Unit tests cover: feature extraction, `days_since_last_burn` derivation edge cases, runtime contract validation

## Notes

- The hardest part of this task is constructing clean training labels. Spend time getting the "new ignition vs spread" distinction right — a detection adjacent to an existing fire event is NOT a new ignition event. Use the `fire_events` clustering/grouping logic that already exists rather than raw fire_detections.
- Do not use the existing risk grid heuristic as a baseline for model targets — the risk grid is not ground truth. Use actual fire ignition events from the FIRMS archive.
- Calibration is not optional. The feature brief explicitly calls out that "confidence levels and uncertainty should be clearly communicated to users from the start." An uncalibrated XGBoost will produce overconfident probabilities.
- The `+48h` forecast horizon inherits lower confidence from the GFS forecast itself. The model does not need to model horizon-specific uncertainty — that caveat is communicated in the UI (Task 19). The model just runs on the +48h weather forecast inputs.
- Do not block on achieving high absolute accuracy — ignition events are sparse and hard to predict. A well-calibrated model with AUC > 0.65 and clear uncertainty communication is better than an overfit model.
