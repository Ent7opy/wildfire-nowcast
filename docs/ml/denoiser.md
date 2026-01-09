# Hotspot Denoiser (v1): Design, Training, and Workflow

**Audience**: Contributors working on ingest, ML, API, or UI who need to understand what the denoiser is doing and how to train/evaluate it.

**When to read this**: Before treating denoiser outputs as "ground truth", before changing thresholds/labels/features, or when training a new model.

This repo uses a **tabular ML classifier** ("denoiser") to assign each FIRMS detection a probability \(p = P(\text{real fire})\). The output can be used to **drop** likely-noise detections (precision-first) or **downweight** them (coverage-first).

---

## 1. Purpose & Role in the Pipeline

### What Problem It Solves
FIRMS detections contain a mix of:
- True wildfire detections
- False positives / "permanent hotspots" (industrial sources, repeating static locations, one-off sensor artifacts)

The denoiser's job is to **reduce noise before downstream aggregation and visualization**. Denoiser outputs are stored on each `fire_detections` row:
- `denoised_score`: float in \([0,1]\) (probability of being real fire)
- `is_noise`: boolean classification flag

### Where It Runs
- **Automatic (optional)**: FIRMS ingest can trigger denoiser inference after inserting new detections (`ingest/firms_ingest.py`). Controlled via environment variables in `ingest/config.py`.
- **Manual**: Run inference directly via `ml/denoiser_inference.py`.

### How the API Uses It

Two usage patterns:

1. **Default filtering ("drop mode")**: Most DB queries exclude detections marked as noise (`is_noise IS NOT TRUE`). See `api/fires/repo.py`.
2. **Weighting ("weight mode")**: Heatmap helper aggregates using `denoised_score` as the value being summed. See `api/fires/service.py` (`weight_by_denoised_score=True`).

The `/fires/detections` endpoint supports `include_denoiser_fields=true` and `include_noise=true`.

---

## 2. Labeling Strategy (Weak Labels)

### High-Level Approach
v1 uses **heuristic weak labels** to produce high-precision training examples. Labels are written to `fire_labels` and exported into immutable Parquet snapshots.

### Label Definitions (3-Way)

| Label | Definition | Rationale |
| :--- | :--- | :--- |
| **POSITIVE** | Likely real wildfire | High-precision signals with growth/persistence or clustering |
| **NEGATIVE** | Likely noise or permanent hotspot | Clear industrial sources or extremely long static persistence |
| **UNKNOWN** | Ambiguous or uncertain | Excluded from training in v1 |

**Rationale**: Training on high-precision POS/NEG labels reduces label noise.

### Implemented Heuristics (v1)

`ml/denoiser/label_v1.py` applies these (in order):

- **Industrial mask → NEGATIVE**: Within configurable radius of entries in `industrial_sources`
- **Low-confidence singleton → NEGATIVE**: `confidence < threshold` and no other detections within (distance, time window)
- **Chronic static → NEGATIVE**: Repeated detections in the same snapped cell on many distinct days
- **Persistent cluster → POSITIVE**: Detections with at least one nearby neighbor in a time window (≥2 detections within 2km/24h)

Everything else stays `UNKNOWN` and is **excluded from training**.

### External Data Used
- **Industrial sources catalog**: Detections near known industrial locations are labeled as negative. The ingest tool `ingest/industrial_sources_ingest.py` downloads/loads the WRI Global Power Plant Database.

### Important: Spec vs Implementation
The current v1 implementation is intentionally simpler than the full spec. When retraining, treat the code as canonical behavior.

---

## 3. Features (v1)

v1 is a **tabular** classifier. Features from:

### A) FIRMS Per-Detection Signals
- **Radiometric/intensity**: `frp`, `brightness`, `bright_t31`, `brightness_minus_t31`
- **Confidence**: `confidence_norm` (maps categorical to numeric)
- **Pixel geometry**: `scan`, `track`
- **Optional sensor/platform codes**: encoded categorical IDs (not used in default v1 config)

### B) Time Encodings
Cyclical time features:
- `sin_hour`, `cos_hour`
- `sin_doy`, `cos_doy`

### C) Local Spatiotemporal Context (Past-Only)
Counts and distances from *past and nearby* detections:
- Neighborhood counts: `n_2km_6h`, `n_2km_24h`, `n_5km_24h`
- Same-cell counts: `n_same_cell_24h`, `n_same_cell_7d`
- `dist_nn_24h_km` (distance to nearest neighbor in prior 24h)
- "Seen before" signals: `seen_same_cell_past_3d`, `days_with_detection_past_30d_in_cell`

**Designed to avoid future leakage at inference time.**

### D) Terrain Context (Optional)
If `region_name` is provided:
- `elevation_m`, `slope_deg`, `aspect_sin`, `aspect_cos`

See `configs/denoiser_train_v1.yaml` for the exact feature list.

---

## 4. Model (v1) and Hyperparameters

### Model Family
`sklearn.ensemble.HistGradientBoostingClassifier` - tree-based gradient boosting for tabular data.

### Class Imbalance Handling
Optional per-sample weights to balance positive/negative examples (`handle_imbalance: true` in config).

### Hyperparameters (Baseline)
Defined in `configs/denoiser_train_v1.yaml`:
- `max_iter: 100`
- `learning_rate: 0.1`
- `max_depth: 5`
- `l2_regularization: 0.01`

Resolved config written to each model run as `config_resolved.yaml`.

---

## 5. Evaluation and Operating Thresholds

### Metrics Produced
The evaluation tool (`ml/eval_denoiser.py`) produces:
- ROC / PR curves (`roc_curve.png`, `pr_curve.png`)
- Calibration plot (`calibration.png`)
- Threshold sweep CSV (`threshold_sweep.csv`)
- Summary JSON (`metrics_summary.json`)
- Human-readable report (`thresholds.md`)

Artifacts written to `reports/denoiser_v1/<run_id>/`.

### Two Thresholds, Two Modes

1. **`strong_filter_threshold`**: For "drop mode" (precision-first). Detections with \(p < t\) are treated as noise.
2. **`downweight_threshold`**: For "weight mode" (coverage-first). Detections below this should be downweighted rather than dropped.

**Note**: The ingestion hook uses a **single** configured threshold (`DENOISER_THRESHOLD`) that controls what the API hides by default.

### Example Report
See `reports/denoiser_v1/20251224_154221_7ef20ce10867454709419bcfcede93d76e3d58fb/thresholds.md` for an example.

### Caveat: Threshold Stability
Evaluation can be misleading with highly imbalanced or small validation sets. Treat AUC numbers as *diagnostics*, not guarantees.

---

## 6. Training Workflow

### Step 0: Ensure Data is Available

```bash
# Ingest industrial sources (for negatives)
make ingest-industrial ARGS="--wri --bbox <min_lon> <min_lat> <max_lon> <max_lat>"

# Ingest FIRMS detections for your AOI and time period
# For training windows beyond NRT retention, use backfill/archive sources
make ingest-firms ARGS="--day-range 7 --area <bbox>"
```

### Step 1: Generate Heuristic Labels

```bash
make denoiser-label ARGS="--bbox <min_lon> <min_lat> <max_lon> <max_lat> --start YYYY-MM-DD --end YYYY-MM-DD --version v1.0.0"
```

**Outputs**: Writes labels into `fire_labels` in the DB.

### Step 2: Export Training Snapshot

```bash
make denoiser-snapshot ARGS="--bbox <min_lon> <min_lat> <max_lon> <max_lat> --start YYYY-MM-DD --end YYYY-MM-DD --version v1.0.0 --aoi <region_name>"
```

**Outputs**:
- `data/denoiser/snapshots/run_<timestamp>/train.parquet`
- `data/denoiser/snapshots/run_<timestamp>/eval.parquet`
- `data/denoiser/snapshots/run_<timestamp>/metadata.json`

**Data Split**: Temporal split enforced:
- **Train**: First 11 months
- **Eval**: Last 1 month

This prevents leakage from the same fire events across multiple days.

### Step 3: Train the Model

Edit or copy `configs/denoiser_train_v1.yaml` to point `snapshot_path` at your snapshot directory.

```bash
make denoiser-train CONFIG=configs/denoiser_train_v1.yaml
```

**Outputs**:
- `models/denoiser_v1/<run_id>/model.pkl`
- `models/denoiser_v1/<run_id>/metadata.json`
- `models/denoiser_v1/<run_id>/feature_list.json`
- `models/denoiser_v1/<run_id>/metrics.json`
- `models/denoiser_v1/<run_id>/config_resolved.yaml`

### Step 4: Evaluate and Choose Thresholds

```bash
make denoiser-eval MODEL_RUN="models/denoiser_v1/<run_id>" SNAPSHOT="data/denoiser/snapshots/run_<timestamp>"
```

**Tuning options**:
- `ARGS="--target_precision 0.95"` for stricter filter
- `ARGS="--target_recall 0.95"` for more recall-preserving downweight
- `ARGS="--min_downweight_rate 0.02"` to avoid degenerate threshold

**Outputs**: `reports/denoiser_v1/<run_id>/...` (curves, sweeps, `thresholds.md`)

### Step 5: Deploy to Ingestion (Optional)

Set these env vars to enable automatic denoiser inference during ingest:

```env
DENOISER_ENABLED=true
DENOISER_MODEL_RUN_DIR=models/denoiser_v1/<run_id>
DENOISER_THRESHOLD=<strong_filter_threshold>
DENOISER_BATCH_SIZE=500
DENOISER_REGION=<region_name>  # to include terrain features
```

---

## 7. Known Limitations & Caveats (v1)

### Label Bias / Weak Supervision Bias
Model is trained on heuristic labels encoding current beliefs about "obvious noise" and "obvious fire". Can bias away from ambiguous-but-real events.

### Industrial Catalog Coverage
Depends on completeness/coverage of `industrial_sources`. Missing sources leak industrial hotspots into positives.

### Domain Shift
Feature distributions differ by region/season and FIRMS source variants. A model trained on one AOI/time span may not generalize. Slice metrics help identify issues.

### Calibration
`denoised_score` is a probability estimate but not guaranteed to be calibrated across domains. Use calibration plots and prefer conservative interpretation.

### Spatiotemporal Features Can Be Expensive
Some feature builders use DB queries. Inference uses batch SQL, but training snapshot generation can be slow for large datasets.

### "NULL Means Unscored"
Downstream defaults treat `NULL` as "not noise" (`is_noise IS NOT TRUE`). Unscored detections can slip through unless inference is enabled.

---

## 8. Bumping Model Version (v2+)

The model "version" is a convention and filesystem layout choice.

**Recommended approach**:
1. Copy `configs/denoiser_train_v1.yaml` → `configs/denoiser_train_v2.yaml`
2. Change `model_output_root` to `models/denoiser_v2`
3. Train with new config
4. Evaluate and keep reports in `reports/denoiser_v2/<run_id>/`
5. Point `DENOISER_MODEL_RUN_DIR` at new run directory
6. Update this doc with feature/label changes

---

## 9. Database Field Semantics

- `is_noise` (`BOOLEAN`): `TRUE` if classified as noise. `NULL` means not scored yet (treated as non-noise by default filters).
- `denoised_score` (`FLOAT`): Probability [0-1] that detection is a real fire.

---

## 10. Downstream Usage

### Drop Mode (Precision-First)
- **Intent**: Only show detections highly likely to be real fires
- **Implementation**: API/DB filter out noise by default (`is_noise IS NOT TRUE`)
- **Usage**: Default settings in `api.fires.repo.list_fire_detections_bbox_time` or API `/fires/detections`

### Weight Mode (Coverage-First)
- **Intent**: Keep all detections but reduce influence of likely noise
- **Implementation**: Request all detections and use `denoised_score` as weighting factor
- **Usage**:
  - **API**: Pass `include_noise=true` and `include_denoiser_fields=true`
  - **Service**: In `api.fires.service.get_fire_cells_heatmap`, set `weight_by_denoised_score=True`
