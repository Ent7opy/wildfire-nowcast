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

