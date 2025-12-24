# Hotspot Denoiser (ML v1)

This document describes the workflow for training and evaluating the FIRMS hotspot denoiser.

## 1. Overview
The denoiser is a binary classifier that distinguishes real wildfires from false positives (noise, industrial heat sources, etc.).

## 2. Pipeline Workflow

### A. Ingest Data
Ingest both fire detections and industrial sources for your AOI.
```bash
# 1. Ingest recent fires (NRT feeds typically only retain ~7 days)
make ingest-firms ARGS="--day-range 7 --area world"

# 2. Ingest industrial sources for the Balkans
make ingest-industrial ARGS="--wri --bbox 13.0 40.0 23.0 49.0"
```

### B. Prepare Training Snapshot
Run the labeling job and export the snapshot in one go.
```bash
# 1. Generate heuristic labels in the DB
make denoiser-label ARGS="--bbox 13.0 40.0 23.0 49.0 --start 2024-01-01 --end 2024-12-31 --version v1.0.0"

# 2. Export to Parquet snapshot
make denoiser-snapshot ARGS="--bbox 13.0 40.0 23.0 49.0 --start 2024-01-01 --end 2024-12-31 --version v1.0.0 --aoi balkans"
```

### C. Training (Next Step)
Once the snapshot is exported, you can train a baseline classifier using the provided features.

> Note: If you are targeting a historical window (e.g. all of 2024), you must ingest historical detections first.
> The default FIRMS `*_NRT` sources are Near Real-Time feeds and typically won’t return older data.

## 3. Labeling Rules (v1)
Refer to `docs/denoiser_labels.md` for detailed logic. The current implementation in `ml/denoiser/label_v1.py` covers:
- **Industrial Mask**: Spatial join with industrial sources.
- **Chronic Static**: Frequency analysis of static grid cells over 90 days.
- **Low-Conf Singleton**: Isolated low-confidence points.
- **Persistent Cluster**: Spatial/temporal clustering for POSITIVE samples.

## 4. Data Split
The snapshot exporter enforces a **temporal split**:
- **Train**: First 11 months of the selected window.
- **Eval**: Last 1 month of the selected window.

This prevents leakage from the same fire events across multiple days.

## 5. Evaluation + default thresholds

Use the evaluation script to compute ROC/PR curves, threshold sweeps, and (optional) a calibration plot, and to auto-pick default operating thresholds:

```bash
# Example (from repo root)
make denoiser-eval MODEL_RUN="models/denoiser_v1/<run_id>" SNAPSHOT="data/denoiser/snapshots/run_<timestamp>"
```

You can tune the auto-picking behavior via `ARGS`, e.g.:
- `ARGS="--target_precision 0.95"` for a stricter strong filter.
- `ARGS="--target_recall 0.95"` for a more recall-preserving downweight split.
- `ARGS="--min_downweight_rate 0.02"` to avoid a degenerate downweight threshold (ensure at least ~2% are in the downweighted bucket).

Artifacts are written to `reports/denoiser_v1/<run_id>/`:
- `roc_curve.png`, `pr_curve.png`, `calibration.png`
- `threshold_sweep.csv` (precision/recall/F1 across thresholds)
- `metrics_summary.json`
- `thresholds.md` (**chosen thresholds + rationale + downstream interpretation contract**)

### Downstream interpretation (drop vs weight)

The evaluation report (`thresholds.md`) defines two intended operating modes, now supported by the API and DB helpers:

1.  **Drop mode (Precision-first)**:
    - **Intent**: Only show detections that are highly likely to be real fires.
    - **Implementation**: The API and DB helpers filter out noise by default (`is_noise IS NOT TRUE`).
    - **Usage**: Use the default settings in `api.fires.repo.list_fire_detections_bbox_time` or the API `/fires/detections`.

2.  **Weight mode (Coverage-first)**:
    - **Intent**: Keep all detections but reduce the influence of likely noise.
    - **Implementation**: Request all detections and use `denoised_score` as a weighting factor.
    - **Usage**: 
        - **API**: Pass `include_noise=true` and `include_denoiser_fields=true`.
        - **Service**: In `api.fires.service.get_fire_cells_heatmap`, set `weight_by_denoised_score=True` to automatically aggregate the sum of scores per grid cell.

#### Database Field Semantics
- `is_noise` (`BOOLEAN`): `TRUE` if the detection is classified as noise. `NULL` means the detection hasn't been scored yet (treated as non-noise by default filters).
- `denoised_score` (`FLOAT`): Probability [0-1] that the detection is a real fire.

