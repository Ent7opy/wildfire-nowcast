# Hotspot denoiser (v1): design, training, and limitations

**Audience**: contributors working on ingest, ML, API, or UI who need to understand what the denoiser is doing and what it is *not* doing.

**When to read this**: before treating denoiser outputs as “ground truth”, and before changing thresholds, labels, or features.

This repo uses a **tabular ML classifier** (“denoiser”) to assign each FIRMS detection a probability \(p = P(\text{real fire})\). The output can be used to **drop** likely-noise detections (precision-first) or **downweight** them (coverage-first).

For the exact labeling heuristics, see [`docs/denoiser_labels.md`](denoiser_labels.md). For the step-by-step training workflow, see [`docs/ml/denoiser.md`](ml/denoiser.md).

---

## 1) Purpose & role in the pipeline

### What problem it solves
FIRMS detections contain a mix of:
- true wildfire detections, and
- false positives / “permanent hotspots” (industrial sources, repeating static locations, one-off sensor artifacts).

The denoiser’s job is to **reduce noise before downstream aggregation and visualization**. In this repo, denoiser outputs are stored on each `fire_detections` row:
- `denoised_score`: float in \([0,1]\) (probability of being real fire)
- `is_noise`: boolean classification flag

These columns are part of the canonical schema (`api/migrations/versions/37962c109cd5_add_fire_detections_schema.py`) and are described in [`docs/data/data_schema_fires.md`](data/data_schema_fires.md).

### Where it runs (today)
- **Automatic (optional)**: FIRMS ingest can trigger denoiser inference after inserting new detections (`ingest/firms_ingest.py`). This is controlled via environment variables in `ingest/config.py` (see “How to run inference” below).
- **Manual**: you can run inference directly via the CLI module `ml/denoiser_inference.py`.

### How the API uses it

There are two key usage patterns already implemented in the API/DB helpers:

- **Default filtering (“drop mode”)**: most DB queries exclude detections explicitly marked as noise by adding `is_noise IS NOT TRUE` (this keeps unscored rows where `is_noise` is `NULL`). See `api/fires/repo.py`.
- **Weighting (“weight mode”)**: the heatmap helper can aggregate using `denoised_score` as the value being summed (instead of counting detections). See `api/fires/service.py` (`weight_by_denoised_score=True`). In this mode, `NULL denoised_score` is treated as full weight (1.0) to avoid NaN poisoning.
To surface denoiser fields to clients, the `/fires/detections` endpoint supports `include_denoiser_fields=true` and `include_noise=true` (see `api/routes/fires.py`).

---

## 2) Labeling strategy (weak labels)

### High-level approach
v1 uses **heuristic weak labels** to produce high-precision training examples. Labels are written to a DB labels table (`fire_labels`) and exported into immutable Parquet snapshots for training.

The conceptual spec and rationale are in [`docs/denoiser_labels.md`](denoiser_labels.md). The implemented labeling job is in `ml/denoiser/label_v1.py`.

### External data used
- **Industrial sources catalog**: detections near known industrial locations are labeled as negative. The ingest tool `ingest/industrial_sources_ingest.py` downloads/loads the WRI Global Power Plant Database and stores it in `industrial_sources`.
### Implemented heuristics (v1)
`ml/denoiser/label_v1.py` applies these (in order), assigning `POSITIVE`, `NEGATIVE`, or leaving `UNKNOWN`:

- **Industrial mask → NEGATIVE**: within a configurable radius of entries in `industrial_sources`.
- **Low-confidence singleton → NEGATIVE**: `confidence < threshold` and no other detections within (distance, time window).
- **Chronic static → NEGATIVE**: repeated detections in the same snapped cell on many distinct days (v1 implementation uses a simplified “same cell day-count” check).
- **Persistent cluster → POSITIVE**: detections with at least one nearby neighbor in a time window (v1 uses a simplified “≥2 detections within 2km/24h” signal).
Everything else stays `UNKNOWN` and is **excluded from training**.

### Important: spec vs implementation
The doc spec includes adjacency/growth concepts and more nuanced UNKNOWN handling; the current v1 implementation is intentionally simpler in places (see comments in `ml/denoiser/label_v1.py`). When retraining, treat the code as canonical behavior, and treat the spec as the intended direction.

---

## 3) Features (v1)

v1 is a **tabular** classifier. Features are computed from three main sources:

### A) FIRMS per-detection signals
From the detection row itself / `raw_properties` (`ml/denoiser/features.py`):
- **Radiometric/intensity**: `frp`, `brightness`, `bright_t31`, and `brightness_minus_t31`.
- **Confidence**: `confidence_norm` (maps categorical confidence to numeric proxies when needed).
- **Pixel geometry**: `scan`, `track`.
- **Optional sensor/platform codes**: encoded categorical IDs for `instrument`/`satellite` (not used in the default v1 config).
### B) Time encodings
Cyclical time features (`ml/denoiser/features.py`):
- `sin_hour`, `cos_hour`
- `sin_doy`, `cos_doy`
### C) Local spatiotemporal context (past-only)
Counts and distances derived from *past and nearby* detections (`ml/denoiser/features.py`):
- Neighborhood counts like `n_2km_6h`, `n_2km_24h`, `n_5km_24h`
- Same-cell counts like `n_same_cell_24h`, `n_same_cell_7d`
- `dist_nn_24h_km` (distance to nearest neighbor in the prior 24h)
- “Seen before” signals like `seen_same_cell_past_3d`, `days_with_detection_past_30d_in_cell`
These features are designed to avoid future leakage at inference time.

### D) Terrain context (optional)
If a `region_name` is provided, terrain features are interpolated from the region’s rasters (`ml/denoiser/features.py`):
- `elevation_m`, `slope_deg`, `aspect_sin`, `aspect_cos`
### Concrete v1 feature list
The default v1 training config enumerates the exact features used for the baseline model:
- See `configs/denoiser_train_v1.yaml` (`features:` list).
---

## 4) Model (v1) and key hyperparameters

### Model family
v1 trains a `sklearn.ensemble.HistGradientBoostingClassifier` (see `ml/train_denoiser.py`). This is a tree-based gradient boosting model that works well for tabular signals and is relatively robust to nonlinear interactions.

### Class imbalance handling
Training optionally uses per-sample weights to approximately balance positive/negative examples (see `ml/train_denoiser.py`, `handle_imbalance: true` in `configs/denoiser_train_v1.yaml`).

### Hyperparameters (baseline defaults)
The baseline config is defined in `configs/denoiser_train_v1.yaml` under `model_params`, e.g.:
- `max_iter: 100`
- `learning_rate: 0.1`
- `max_depth: 5`
- `l2_regularization: 0.01`
The actual resolved config is written into each model run directory as `config_resolved.yaml` by `ml/train_denoiser.py`.

---

## 5) Evaluation, operating thresholds, and reports

### What metrics are produced
The evaluation tool (`ml/eval_denoiser.py`) produces:
- ROC / PR curves (`roc_curve.png`, `pr_curve.png`)
- A calibration plot (`calibration.png`) when possible
- A threshold sweep CSV (`threshold_sweep.csv`) and summary JSON (`metrics_summary.json`)
- A human-readable threshold report (`thresholds.md`)
Artifacts are written to `reports/denoiser_v1/<run_id>/`.

### Two thresholds, two “modes”
The evaluation report defines two operating points:
- **`strong_filter_threshold`**: intended for “drop mode” (precision-first). Detections with \(p < t\) are treated as noise.
- **`downweight_threshold`**: intended for “weight mode” (coverage-first). Detections below this threshold should be downweighted rather than dropped.
Note: the ingestion hook currently writes a boolean `is_noise` using a **single** configured threshold (`DENOISER_THRESHOLD`) in `ingest/firms_ingest.py` / `ml/denoiser_inference.py`. That threshold effectively controls what the API will hide by default (because default queries exclude `is_noise = TRUE`).

### Example report (checked in)
One concrete run is checked in here:
- [`reports/denoiser_v1/20251224_154221_7ef20ce10867454709419bcfcede93d76e3d58fb/thresholds.md`](../reports/denoiser_v1/20251224_154221_7ef20ce10867454709419bcfcede93d76e3d58fb/thresholds.md)

Use this as an illustration of what the evaluation outputs look like (thresholds + rationale + slice notes), not as a universal truth for every region/time period.
### Caveat: threshold stability and dataset skew
Evaluation can become statistically misleading when the validation set is highly imbalanced (high prevalence of positives) or too small. `ml/eval_denoiser.py` emits warnings in `metrics_summary.json` when eval sets are small or single-class, and training may fall back from time-split to stratified-random split to make AUC defined (`ml/train_denoiser.py`). Treat AUC numbers as *diagnostics*, not a guarantee of real-world ranking quality across domains.
---

## 6) Known limitations & caveats (v1)

### Label bias / weak supervision bias
The model is trained on heuristic labels that encode our current beliefs about “obvious noise” and “obvious fire”. This can bias the model toward those heuristics and away from ambiguous-but-real events.
### Industrial catalog coverage and staleness
Industrial masking depends on the completeness and geographic coverage of `industrial_sources` (see `ingest/industrial_sources_ingest.py`). Missing sources (or mislocated ones) will leak industrial hotspots into positives and/or confuse the model.
### Domain shift (region, season, sensor, source)
The feature distributions differ by region/season and by FIRMS source variants (e.g., NRT vs archive sources). A model trained on one AOI/time span may not generalize. Slice metrics in evaluation (`slice_metrics_by_sensor.csv`, etc.) help identify these issues.
### Calibration and interpretation of probabilities
`denoised_score` is a model probability estimate, but it is not guaranteed to be calibrated across domains. Use calibration plots in reports and prefer conservative interpretation when moving to new regions.
### Spatiotemporal features can be expensive
Some feature builders use DB queries over nearby detections. Inference uses the batch SQL implementation (`add_spatiotemporal_context_batch`), but training snapshot generation uses a per-row approach (`add_spatiotemporal_context`), which can be slow for large datasets. Plan AOI/time windows accordingly.
### “NULL means unscored”
Many downstream defaults treat `NULL` as “not noise” (e.g., `is_noise IS NOT TRUE`). This is convenient operationally, but it means unscored detections can slip through filters unless inference is enabled and up to date.
---

## 7) How to retrain (v1 workflow)

This section focuses on the “what to run, where outputs go, and how to bump versions”. For more procedural notes, see [`docs/ml/denoiser.md`](ml/denoiser.md).
### Step 0: Ensure data is available
- Ingest industrial sources (for the industrial mask / negatives):
  - `make ingest-industrial ARGS="--wri --bbox <min_lon> <min_lat> <max_lon> <max_lat>"`
- Ingest FIRMS detections for your AOI and time period. For training windows beyond NRT retention, use backfill/archive sources (see [`docs/ingest/ingest_firms.md`](ingest/ingest_firms.md)).
### Step 1: Generate heuristic labels in the DB
Run the labeling job:
- `make denoiser-label ARGS="--bbox <min_lon> <min_lat> <max_lon> <max_lat> --start YYYY-MM-DD --end YYYY-MM-DD --version v1.0.0"`

Outputs:
- Writes labels into `fire_labels` in the DB (see `ml/denoiser/label_v1.py`).
### Step 2: Export an immutable training snapshot
Export labeled detections + features to Parquet:
- `make denoiser-snapshot ARGS="--bbox <min_lon> <min_lat> <max_lon> <max_lat> --start YYYY-MM-DD --end YYYY-MM-DD --version v1.0.0 --aoi <region_name>"`

Outputs:
- `data/denoiser/snapshots/run_<timestamp>/train.parquet`
- `data/denoiser/snapshots/run_<timestamp>/eval.parquet`
- `data/denoiser/snapshots/run_<timestamp>/metadata.json`
### Step 3: Train the model
Edit or copy the training config (baseline is `configs/denoiser_train_v1.yaml`) to point `snapshot_path` at your new snapshot directory.

Train:
- `make denoiser-train CONFIG=configs/denoiser_train_v1.yaml`

Outputs:
- `models/denoiser_v1/<run_id>/model.pkl`
- `models/denoiser_v1/<run_id>/metadata.json`
- `models/denoiser_v1/<run_id>/feature_list.json`
- `models/denoiser_v1/<run_id>/metrics.json`
- `models/denoiser_v1/<run_id>/config_resolved.yaml`
### Step 4: Evaluate and choose thresholds
Run evaluation using the trained run directory and the snapshot:
- `make denoiser-eval MODEL_RUN="models/denoiser_v1/<run_id>" SNAPSHOT="data/denoiser/snapshots/run_<timestamp>"`

Outputs:
- `reports/denoiser_v1/<run_id>/...` (curves, sweeps, `thresholds.md`)
### Step 5: Deploy to ingestion (optional) and pick the threshold
If you want the ingest pipeline to populate `denoised_score`/`is_noise` automatically, set these env vars (see `ingest/config.py` and [`docs/ingest/ingest_firms.md`](ingest/ingest_firms.md)):
- `DENOISER_ENABLED=true`
- `DENOISER_MODEL_RUN_DIR=models/denoiser_v1/<run_id>`
- `DENOISER_THRESHOLD=<strong_filter_threshold>` (commonly; this controls `is_noise = (p < threshold)`)
- `DENOISER_BATCH_SIZE=500` (or tuned)
- `DENOISER_REGION=<region_name>` (to include terrain features at inference)
---

## 8) How to bump model version (v2+)

The model “version” is mostly a **convention** and filesystem layout choice in this repo.
Recommended approach:
- Copy `configs/denoiser_train_v1.yaml` → `configs/denoiser_train_v2.yaml` and change `model_output_root` to something like `models/denoiser_v2`.
- Train with the new config; the run artifacts will land in `models/denoiser_v2/<run_id>/`.
- Evaluate with `make denoiser-eval MODEL_RUN="models/denoiser_v2/<run_id>" ...` and keep reports in a matching folder, e.g. `reports/denoiser_v2/<run_id>/` (you can pass `OUT=...` to the Make target).
- Point `DENOISER_MODEL_RUN_DIR` at the new run directory.
When you bump versions, also update this doc (and/or the workflow doc) with any feature/label changes so contributors can compare v1 vs v2 behavior.

