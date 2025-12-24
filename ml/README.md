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
- `configs/`: Training configuration files.
- `models/`: Trained model artifacts.

