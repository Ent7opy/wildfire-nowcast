# Modal Cloud Training Integration

> **Status**: Research / Pre-implementation
> **Scope**: Denoiser v2 (XGBoost), Spread v2 (PyTorch U-Net), Ignition (XGBoost)
> **Audience**: Two sections — see below

---

## Overview

Modal (modal.com) is a cloud compute platform purpose-built for Python-native ephemeral batch jobs. It is the recommended platform for running wildfire-nowcast ML training workloads outside of local developer machines.

Key fit for this project:
- Ephemeral containers — no persistent infra to manage; maps cleanly to `make denoiser-train-v2` style invocations
- Python-first SDK — wraps existing training entry points with minimal code changes
- XGBoost CPU training costs ~$0.75/hr on 16 cores; well within the free tier ($30/mo) for routine retraining
- GPU available (L4 at $0.80/hr) for spread model U-Net training or GPU-accelerated XGBoost
- Persistent Volumes for model artifacts, replacing local `models/` directory

### Connectivity model: snapshot-first, DB-free training

The pipeline is split at the snapshot boundary to avoid any need for Modal containers to reach the database:

```
Local machine (DB access)         Modal cloud (no DB needed)
─────────────────────────         ──────────────────────────
make denoiser-snapshot-v2    →    upload snapshot parquet to Modal Volume
make denoiser-label-v2       →    (label stage also runs locally or uploads result)
                                  make modal-denoiser-train-v2   ← reads only parquet
                                  make modal-denoiser-eval-v2    ← reads parquet + model
```

This eliminates the need for static IPs, firewall rules, or a paid Modal Proxy. The Starter plan ($0/month base, $30 free compute credits) is sufficient. No plan upgrade required.

---

## Part 1: Human Operator Guide

This section covers all manual steps a human must perform in the Modal dashboard and CLI. These steps cannot be automated by code agents.

---

### 1.1 Account & Workspace Setup

1. Go to [modal.com](https://modal.com) and sign up via GitHub SSO.
2. Enable MFA on your account (Settings → Security).
3. Your workspace is auto-created from your GitHub username. Note the workspace name — it appears in all resource URLs and CLI output.
4. (Optional but recommended) Create a dedicated **environment** for ML training to isolate it from any future inference deployments:
   - Dashboard → Environments → New Environment → name it `training`
   - Reference it with `-e training` in all `modal` CLI calls

---

### 1.2 CLI Installation & Authentication

```bash
# Install into your global Python (or uv tool install modal)
pip install modal

# Authenticate — opens browser SSO, writes credentials to ~/.modal.toml
modal setup

# Verify
modal profile current
```

For non-interactive environments (CI, other machines), skip `modal setup` and use environment variables instead — see Section 1.6.

---

### 1.3 Secrets: Non-Sensitive Training Config

Because training runs entirely from pre-exported parquet snapshots (no live DB access), Modal containers need **no database credentials**. The only secret needed is a small bundle of non-sensitive ML configuration values.

#### Create the training config secret

```bash
modal secret create wildfire-ml-config \
  DENOISER_REQUIRED=false \
  DENOISER_PIPELINE_VERSION=v2 \
  DENOISER_THRESHOLD_PROFILE=env \
  FIRE_SCORING_WEATHER_TIME_TOLERANCE_HOURS=6 \
  DENOISER_MOISTURE_TIME_TOLERANCE_HOURS=48
```

Add more values as needed when adding spread or ignition training. These are all non-sensitive — they're ML tuning knobs, not credentials.

**Verification:** Dashboard → Secrets → `wildfire-ml-config` should appear. The secret name is referenced by name in code, so it must match exactly.

> **No database credentials in Modal.** Snapshot export and event labeling run locally (where you already have DB access). Only training and evaluation — which read exclusively from parquet files on the Modal Volume — run in the cloud.

---

### 1.4 Uploading Snapshots to Modal Volume

Before triggering a cloud training run, export snapshots locally and push them to the Modal Volume:

```bash
# 1. Export snapshot locally as usual
make denoiser-snapshot-v2 ARGS="--bbox -125 32 -100 50 --start 2024-01-01 --end 2024-12-31 --version v2_us"
make denoiser-label-v2 ARGS="--start 2024-01-01 --end 2024-12-31"

# 2. Push the resulting parquet files to Modal Volume
modal volume put wildfire-snapshots \
  ./data/denoiser/snapshots_v2/v2_us/ \
  /denoiser/snapshots_v2/v2_us/

# 3. Verify
modal volume ls wildfire-snapshots /denoiser/snapshots_v2/
```

The `snapshot_path` config key in your YAML must then point to the volume path (e.g., `/data/denoiser/snapshots_v2/v2_us`). The dev agent handles this path substitution in the Modal wrapper — see Part 2.

---

### 1.5 Volumes: Model Artifact Storage

Volumes are Modal's persistent distributed filesystem. They replace the local `models/` directory.

```bash
# Create once — persists across all runs
modal volume create wildfire-models
modal volume create wildfire-snapshots

# Verify
modal volume list
```

After training runs, artifacts in `wildfire-models` survive container shutdown. To download trained models locally:

```bash
# Download a specific model run
modal volume get wildfire-models \
  /denoiser_v2/<run_id>/model.onnx \
  ./models/denoiser_v2/<run_id>/model.onnx

# Browse the volume
modal volume ls wildfire-models /denoiser_v2/
modal volume ls wildfire-models /spread_v2/
modal volume ls wildfire-models /ignition/
```

---

### 1.6 Service Token for CI / Non-Interactive Use

For use in CI pipelines (GitHub Actions), other machines, or Makefiles without `~/.modal.toml`:

1. Dashboard → Settings → Service Users → Create Service User.
2. Name it `wildfire-training-ci`, assign **Contributor** role, scope to the `training` environment if you created one.
3. Generate a token. Copy `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
4. Add them to your `.env` (never commit) and to GitHub Actions secrets.

```bash
# In .env (already in .gitignore)
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...
```

With these set in the environment, all `modal` CLI commands authenticate automatically without `~/.modal.toml`.

---

### 1.7 Plan & Cost Monitoring

- **Starter plan** (free base): $30/month compute credits, 10 concurrent GPUs, 100 containers. This is the only plan needed — no upgrade required.
- Monitor spend: Dashboard → Usage. Set a spending alert under Settings → Billing.

**Estimated costs per training run:**

| Pipeline | Compute | Est. Duration | Est. Cost |
|----------|---------|---------------|-----------|
| Denoiser v2 (XGBoost, 16-core CPU) | 16 CPU cores | 30–90 min | $0.40–$1.20 |
| Denoiser v2 (XGBoost, L4 GPU) | 1× L4 GPU | 15–45 min | $0.20–$0.60 |
| Spread v2 (U-Net, A10 GPU) | 1× A10 GPU | 2–6 hr | $2.20–$6.60 |
| Ignition (XGBoost, 8-core CPU) | 8 CPU cores | 15–45 min | $0.10–$0.30 |
| Snapshot export (CPU-heavy) | 8 CPU cores | 30–60 min | $0.15–$0.30 |

At Starter tier, the full pipeline (all three models) fits within free credits for ~10–15 full training cycles per month.

---

### 1.8 Deploying the Training App

After the code agent has implemented the Modal app files, deploy once to make functions persistently callable:

```bash
# From project root
make modal-deploy

# Verify deployment
modal app list
modal app logs wildfire-denoiser-v2
```

Redeploy whenever the app code changes:
```bash
make modal-deploy
```

Deployed apps persist until explicitly stopped. For one-off training runs without a persistent deployment, use `modal run` instead.

---

### 1.9 Triggering Training Runs

Once deployed, trigger training runs from your local machine or CI:

```bash
# 1. Locally: export snapshot + upload to Modal Volume
make denoiser-snapshot-v2 ARGS="--bbox -125 32 -100 50 --start 2024-01-01 --end 2024-12-31 --version v2_us"
make denoiser-label-v2 ARGS="--start 2024-01-01 --end 2024-12-31"
make modal-upload-snapshot VERSION=v2_us

# 2. Train on Modal
make modal-denoiser-train-v2 CONFIG=denoiser_train_v2.yaml SNAPSHOT=/data/denoiser/snapshots_v2/v2_us

# Monitor live
make modal-logs-denoiser

# Download artifacts
make modal-get-model FAMILY=denoiser_v2 RUN_ID=<run_id>
```

After training, use the existing `make model-register` / `make model-promote` targets to register artifacts through the API as usual — these work with locally downloaded artifacts or with paths pointing to the Modal volume (if the API container can access it).

---

## Part 2: AI Dev Agent Implementation Guide

This section is addressed to AI coding agents that will implement the Modal integration in this codebase. It contains all technical context needed to write correct code without guesswork.

---

### 2.1 Repository Context

**ML training entry points** (all accept `--config <yaml_path>`):

| Script | Entry point | Primary output |
|--------|-------------|---------------|
| `ml/train_denoiser_v2.py` | `main()` at L1688, calls `train_denoiser_v2(config)` | `models/denoiser_v2/<run_id>/` |
| `ml/train_spread_v2.py` | `main()` at L470, calls `train_spread_v2(config)` | `models/spread_v2/<run_id>/` |
| `ml/train_ignition.py` | `main()`, calls `train_ignition(config)` | `models/ignition/<run_id>/` |
| `ml/denoiser/export_snapshot_v2.py` | `main()` at L552 | `data/denoiser/snapshots_v2/` |
| `ml/ignition/snapshot.py` | `main()` | `data/snapshots/ignition/` |

**Artifact layout per run** (denoiser example — spread/ignition follow same pattern):
```
models/denoiser_v2/<YYYYMMDD_HHMMSS_gitsha>/
  model_bundle.pkl       # joblib-pickled bundle (model + calibrators + thresholds)
  model.pkl              # bare XGBoost model
  feature_list.json      # ordered feature names
  metrics.json           # CV + holdout metrics + SHAP
  gate_report.json       # {"pass": true/false, ...}
  config_resolved.yaml   # full resolved config
  metadata.json          # run_id, git_sha, versions, env info
```

**What runs locally vs. on Modal:**

| Stage | Runs where | DB needed? |
|-------|-----------|-----------|
| `denoiser-snapshot-v2` | Local machine | Yes — queries fire_detections |
| `denoiser-label-v2` | Local machine | Yes — queries event data |
| `denoiser-train-v2` | **Modal** | No — reads parquet snapshots only |
| `denoiser-eval-v2` | **Modal** | No — reads parquet + model artifacts |
| `spread-train-v2` | **Modal** | No — reads parquet snapshots only |
| `ignition-snapshot` | Local machine | Yes — queries grid features |
| `ignition-train` | **Modal** | No — reads parquet snapshots only |

**Key internal imports** used by training code (must be importable in Modal container):
- `from api.core.grid import GridSpec, GridWindow, get_grid_window_for_bbox` — grid math only, no DB
- `from ingest.weather_repository import GFS_GRID_DEG` — constants only, no DB

Note: `api.db`, `api.fires.service`, and `api.terrain.window` are **not** imported during training — only during snapshot export, which stays local. The `api/` and `ingest/` packages still need to be in the container image because `api.core.grid` and `ingest.weather_repository` are imported.

**Environment variables consumed by Modal training containers** (all non-sensitive, from `wildfire-ml-config` secret):
```
DENOISER_REQUIRED=false
DENOISER_PIPELINE_VERSION=v2
DENOISER_THRESHOLD_PROFILE=env
FIRE_SCORING_WEATHER_TIME_TOLERANCE_HOURS=6   (default)
DENOISER_MOISTURE_TIME_TOLERANCE_HOURS=48     (default)
```

No database credentials. No API keys. The Postgres secret is never passed to training functions.

---

### 2.2 File Structure to Create

Create the following new files. Do **not** modify existing training scripts.

```
ml/
  modal/
    __init__.py          # empty
    images.py            # Image definitions (shared across apps)
    volumes.py           # Volume definitions (shared across apps)
    denoiser.py          # Modal app for denoiser pipeline
    spread.py            # Modal app for spread pipeline
    ignition.py          # Modal app for ignition pipeline
    pipeline.py          # Optional: orchestrated full pipeline entrypoint
```

---

### 2.3 Image Definition (`ml/modal/images.py`)

The image must:
1. Use Python 3.11 exactly (project requirement — no 3.12+)
2. Install `libgomp1` via apt (required by XGBoost on Linux)
3. Sync Python deps from `ml/pyproject.toml` using `uv_sync`
4. Add local packages `ml/`, `api/`, and `ingest/` as importable Python sources
5. Add `configs/` directory for YAML config access

```python
import modal
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # project root

def build_training_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(["libgomp1", "libstdc++6", "git"])
        .uv_sync(
            app_dir=str(ROOT / "ml"),       # reads ml/pyproject.toml + ml/uv.lock
        )
        # Add all three local packages — training code imports from all of them
        .add_local_python_source(str(ROOT / "ml"), remote_path="/pkg/ml")
        .add_local_python_source(str(ROOT / "api"), remote_path="/pkg/api")
        .add_local_python_source(str(ROOT / "ingest"), remote_path="/pkg/ingest")
        .add_local_dir(str(ROOT / "configs"), remote_path="/configs")
    )
```

**Note on `uv_sync`**: It installs dependencies without installing the project itself (equivalent to `--no-install-project`). This is intentional — source code is added separately via `add_local_python_source`, so source changes don't bust the dependency cache layer.

**If `uv_sync` is unavailable** (older Modal version): fall back to `pip_install_from_pyproject("ml/pyproject.toml")`.

---

### 2.4 Volume Definition (`ml/modal/volumes.py`)

```python
import modal

# Created once by operator: `modal volume create wildfire-models`
model_volume = modal.Volume.from_name("wildfire-models", create_if_missing=True)

# Created once by operator: `modal volume create wildfire-snapshots`
snapshot_volume = modal.Volume.from_name("wildfire-snapshots", create_if_missing=True)

MODELS_MOUNT = "/models"
SNAPSHOTS_MOUNT = "/data"
```

Volume mount paths must mirror the local path structure expected by training scripts:
- Local `models/denoiser_v2/<run_id>/` → container `/models/denoiser_v2/<run_id>/`
- Local `data/denoiser/snapshots_v2/` → container `/data/denoiser/snapshots_v2/`

The existing training scripts write artifacts using paths derived from config (`model_output_root`, `out` arg). The config values passed to Modal functions must use `/models/...` and `/data/...` prefixes.

---

### 2.5 Denoiser App (`ml/modal/denoiser.py`)

This app wraps **only the cloud stages**: training and evaluation. Snapshot export and event labeling stay local (they need DB access). The snapshot parquet files are uploaded to the Modal Volume by the operator before triggering a training run (see Part 1, Section 1.4).

```python
import modal
import sys

from ml.modal.images import build_training_image
from ml.modal.volumes import model_volume, snapshot_volume, MODELS_MOUNT, SNAPSHOTS_MOUNT

app = modal.App("wildfire-denoiser-v2")
image = build_training_image()

# No database secrets — training reads only from parquet on the Volume
_secrets = [modal.Secret.from_name("wildfire-ml-config")]

# ── Training ─────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    cpu=16.0,           # XGBoost parallelizes well; set nthread=16 in config
    memory=32768,       # 32 GiB for feature matrix; adjust down if config is small
    timeout=14400,      # 4 hours max; real runs typically 30–90 min
    retries=modal.Retries(max_retries=1, initial_delay=120.0),
    secrets=_secrets,
    volumes={
        MODELS_MOUNT: model_volume,
        SNAPSHOTS_MOUNT: snapshot_volume,
    },
)
def train(
    config_path: str,           # path inside container, e.g. "/configs/denoiser_train_v2.yaml"
    snapshot_path: str | None = None,  # override config snapshot_path if provided
    run_id: str | None = None,
) -> dict:
    """
    Runs train_denoiser_v2.train_denoiser_v2(config).
    Reads parquet from snapshot_volume. Writes artifacts to model_volume.
    Returns the metrics dict from metrics.json.
    """
    import json
    from pathlib import Path
    sys.path.insert(0, "/pkg/ml")
    sys.path.insert(0, "/pkg/api")
    sys.path.insert(0, "/pkg/ingest")

    from ml.train_denoiser_v2 import load_config, train_denoiser_v2

    config = load_config(config_path)

    # Override snapshot path to point at the volume mount
    if snapshot_path:
        config["snapshot_path"] = snapshot_path
    elif not config.get("snapshot_path", "").startswith(SNAPSHOTS_MOUNT):
        raise ValueError(
            f"snapshot_path in config must point to {SNAPSHOTS_MOUNT}/... "
            f"(the Modal Volume). Got: {config.get('snapshot_path')}. "
            "Upload snapshots first: make modal-upload-snapshot VERSION=..."
        )

    # Redirect model output to volume
    if not config.get("model_output_root", "").startswith(MODELS_MOUNT):
        config["model_output_root"] = f"{MODELS_MOUNT}/denoiser_v2"

    run_dir: str = train_denoiser_v2(config)  # returns str (not Path)
    model_volume.commit()

    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    metrics["run_dir"] = run_dir
    return metrics


# ── Evaluation ───────────────────────────────────────────────────────────────

@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    timeout=3600,
    secrets=_secrets,
    volumes={
        MODELS_MOUNT: model_volume,
        SNAPSHOTS_MOUNT: snapshot_volume,
    },
)
def evaluate(model_run: str, snapshot_path: str, out_dir: str | None = None) -> dict:
    """
    Runs ml/eval_denoiser_v2.py against a model run directory and snapshot.
    gate_report.json is written to out_dir (not model_run dir).
    Returns gate_report dict.
    """
    import json
    from pathlib import Path
    sys.path.insert(0, "/pkg/ml")
    sys.path.insert(0, "/pkg/api")
    sys.path.insert(0, "/pkg/ingest")

    resolved_out = out_dir or f"{MODELS_MOUNT}/denoiser_v2/eval_reports/{Path(model_run).name}"

    import subprocess
    cmd = [
        "python", "-m", "ml.eval_denoiser_v2",  # module is at ml/ root, not ml/denoiser/
        "--model_run", model_run,                 # underscore, not hyphen
        "--snapshot", snapshot_path,
        "--out", resolved_out,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)

    model_volume.reload()
    gate_path = Path(resolved_out) / "gate_report.json"  # written to out_dir, not model_run
    return json.loads(gate_path.read_text()) if gate_path.exists() else {}


# ── Full Pipeline Entrypoint ─────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    config: str = "/configs/denoiser_train_v2.yaml",
    snapshot: str | None = None,
    run_id: str | None = None,
):
    """
    Runs the full denoiser pipeline: train → evaluate.
    Use for one-shot runs: `modal run ml/modal/denoiser.py --config /configs/...`
    """
    import uuid
    rid = run_id or uuid.uuid4().hex[:8]
    print(f"[denoiser] Starting run {rid}")

    result = train.remote(config_path=config, snapshot_path=snapshot, run_id=rid)
    print(f"[denoiser] Training complete: AUC={result.get('holdout_auc', 'n/a')}")
    print(f"[denoiser] Artifacts at: {result['run_dir']}")

    gate = evaluate.remote(
        model_run=result["run_dir"],
        snapshot_path=snapshot or result.get("snapshot_path", ""),
    )
    gate_pass = gate.get("pass", False)
    print(f"[denoiser] Gate: {'PASS' if gate_pass else 'FAIL'}")

    return result
```

**Implementation notes for the agent:**

1. The `load_config` function in `train_denoiser_v2.py` reads a YAML file by path. Either copy the config YAML into the image (`add_local_file`) or pass config as a dict. The dict approach is cleaner for parameterized runs.

2. `train_denoiser_v2(config)` returns a `Path` (or string path) to the run directory. This is the `run_dir` that contains all artifacts.

3. `model_volume.commit()` **must** be called after any write to the volume mount, before returning, to make files visible to subsequent containers or to the operator downloading artifacts.

4. The `sys.path.insert` calls ensure `ml`, `api`, and `ingest` packages are importable regardless of how Modal resolves the container working directory. The `add_local_python_source` calls in the image definition place them under `/pkg/`; adjust the path if the image definition changes.

5. Do not monkey-patch or modify `train_denoiser_v2.py`. The only changes needed are in the wrapper functions above: overriding `model_output_root` in the resolved config, and redirecting output to volume mount paths.

---

### 2.6 Spread App (`ml/modal/spread.py`)

The spread model is a PyTorch U-Net and benefits significantly from GPU. Use A10 or A100 for training runs longer than 2 hours.

```python
import modal
import sys

from ml.modal.images import build_training_image
from ml.modal.volumes import model_volume, snapshot_volume, MODELS_MOUNT, SNAPSHOTS_MOUNT

app = modal.App("wildfire-spread-v2")
image = build_training_image()

# No database secrets — training reads only from parquet on the Volume
_secrets = [modal.Secret.from_name("wildfire-ml-config")]

@app.function(
    image=image,
    gpu="A10",              # U-Net training benefits from GPU
    cpu=4.0,
    memory=32768,
    timeout=28800,          # 8 hours; spread training can be long
    retries=modal.Retries(max_retries=1, initial_delay=120.0),
    secrets=_secrets,
    volumes={
        MODELS_MOUNT: model_volume,
        SNAPSHOTS_MOUNT: snapshot_volume,
    },
)
def train(config_path: str, run_id: str | None = None) -> dict:
    """
    Runs train_spread_v2.train_spread_v2(config).
    Returns metrics dict.
    Config must have model_output_root pointing to /models/spread_v2.
    """
    import json, uuid
    from pathlib import Path
    sys.path.insert(0, "/pkg/ml")
    sys.path.insert(0, "/pkg/api")
    sys.path.insert(0, "/pkg/ingest")

    from ml.train_spread_v2 import load_config, train_spread_v2

    config = load_config(config_path)
    if not config.get("model_output_root", "").startswith(MODELS_MOUNT):
        config["model_output_root"] = f"{MODELS_MOUNT}/spread_v2"

    run_dir = train_spread_v2(config)
    model_volume.commit()

    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    metrics["run_dir"] = str(run_dir)
    return metrics


@app.local_entrypoint()
def main(config: str = "/configs/spread_train_v2.yaml", run_id: str | None = None):
    import uuid
    rid = run_id or uuid.uuid4().hex[:8]
    print(f"[spread] Starting run {rid}")
    result = train.remote(config_path=config, run_id=rid)
    print(f"[spread] Training complete: best_brier={result.get('best_eval_brier', 'n/a')}")
    print(f"[spread] Artifacts at: {result['run_dir']}")
```

**GPU note**: The spread U-Net uses `torch.cuda.is_available()` in `metadata.json` to record GPU availability. No code change needed — PyTorch auto-detects CUDA in the Modal container when `gpu=` is set. If GPU is unavailable (e.g., `gpu=None` for testing), it falls back to CPU automatically.

**ONNX export note**: `train_spread_v2.py` exports `model.onnx` and `model.int8.onnx` inline after training (L383–L418). These write to the `run_dir` which is already on the volume mount. No additional wrapping needed.

---

### 2.7 Ignition App (`ml/modal/ignition.py`)

Snapshot export stays local (needs DB). Only training runs on Modal.

```python
import modal
import sys

from ml.modal.images import build_training_image
from ml.modal.volumes import model_volume, snapshot_volume, MODELS_MOUNT, SNAPSHOTS_MOUNT

app = modal.App("wildfire-ignition")
image = build_training_image()

# No database secrets — training reads only from parquet on the Volume
_secrets = [modal.Secret.from_name("wildfire-ml-config")]


@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    timeout=7200,
    retries=modal.Retries(max_retries=1, initial_delay=60.0),
    secrets=_secrets,
    volumes={
        MODELS_MOUNT: model_volume,
        SNAPSHOTS_MOUNT: snapshot_volume,
    },
)
def train(config_path: str, snapshot_path: str | None = None, run_id: str | None = None) -> dict:
    import json
    from pathlib import Path
    sys.path.insert(0, "/pkg/ml")
    sys.path.insert(0, "/pkg/api")
    sys.path.insert(0, "/pkg/ingest")

    from ml.train_ignition import load_config, train_ignition

    config = load_config(config_path)

    if snapshot_path:
        config["snapshot_path"] = snapshot_path
    elif not config.get("snapshot_path", "").startswith(SNAPSHOTS_MOUNT):
        raise ValueError(
            f"snapshot_path must point to {SNAPSHOTS_MOUNT}/... "
            "Upload snapshots first: make modal-upload-snapshot-ignition VERSION=..."
        )

    if not config.get("out_root", "").startswith(MODELS_MOUNT):
        config["out_root"] = f"{MODELS_MOUNT}/ignition"

    run_dir = train_ignition(config)
    model_volume.commit()

    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    metrics["run_dir"] = str(run_dir)
    return metrics


@app.local_entrypoint()
def main(config: str = "/configs/ignition_train.yaml", run_id: str | None = None):
    import uuid
    rid = run_id or uuid.uuid4().hex[:8]
    print(f"[ignition] Starting run {rid}")
    result = train.remote(config_path=config, run_id=rid)
    print(f"[ignition] AUC-ROC: {result.get('auc_roc', 'n/a')}")
    print(f"[ignition] Artifacts at: {result['run_dir']}")
```

---

### 2.8 Makefile Targets to Add

Add to the root `Makefile`. These mirror the existing `denoiser-*` and `spread-*` targets:

```makefile
# ─── Modal Cloud Training ───────────────────────────────────────────────────
# Requires: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env or environment
#
# Workflow:
#   1. Run snapshot export + labeling LOCALLY (needs DB access):
#        make denoiser-snapshot-v2 ...
#        make denoiser-label-v2 ...
#   2. Upload snapshots to Modal Volume:
#        make modal-upload-snapshot VERSION=v2_us
#   3. Train on Modal (no DB needed):
#        make modal-denoiser-train-v2

## Deploy all Modal training apps (run once after code changes)
modal-deploy:
	modal deploy ml/modal/denoiser.py
	modal deploy ml/modal/spread.py
	modal deploy ml/modal/ignition.py

## Upload denoiser snapshot parquet to Modal Volume (run after local snapshot export)
## Usage: make modal-upload-snapshot VERSION=v2_us
modal-upload-snapshot:
	modal volume put wildfire-snapshots \
		data/denoiser/snapshots_v2/$(VERSION)/ \
		/denoiser/snapshots_v2/$(VERSION)/

## Upload ignition snapshot to Modal Volume
## Usage: make modal-upload-snapshot-ignition VERSION=v1_us
modal-upload-snapshot-ignition:
	modal volume put wildfire-snapshots \
		data/snapshots/ignition/$(VERSION)/ \
		/ignition/snapshots/$(VERSION)/

## Denoiser v2 — training on Modal
## Usage: make modal-denoiser-train-v2 SNAPSHOT=/data/denoiser/snapshots_v2/v2_us
modal-denoiser-train-v2:
	modal run --timestamps ml/modal/denoiser.py::train \
		--config-path /configs/$(or $(CONFIG),denoiser_train_v2.yaml) \
		$(if $(SNAPSHOT),--snapshot-path $(SNAPSHOT),)

## Denoiser v2 — full pipeline on Modal (train + eval)
modal-train-denoiser:
	modal run --timestamps ml/modal/denoiser.py \
		--config /configs/$(or $(CONFIG),denoiser_train_v2.yaml) \
		$(if $(SNAPSHOT),--snapshot $(SNAPSHOT),)

## Spread v2 — training on Modal
## Usage: make modal-train-spread SNAPSHOT=/data/spread/snapshots_v2/v2_us
modal-train-spread:
	modal run --timestamps ml/modal/spread.py \
		--config /configs/$(or $(CONFIG),spread_train_v2.yaml) \
		$(if $(SNAPSHOT),--snapshot-path $(SNAPSHOT),)

## Ignition — training on Modal
## Usage: make modal-train-ignition SNAPSHOT=/data/ignition/snapshots/v1_us
modal-train-ignition:
	modal run --timestamps ml/modal/ignition.py \
		--config /configs/$(or $(CONFIG),ignition_train.yaml) \
		$(if $(SNAPSHOT),--snapshot-path $(SNAPSHOT),)

## Download a model artifact from Modal Volume to local models/
## Usage: make modal-get-model FAMILY=denoiser_v2 RUN_ID=20260101_123456_abc1234
modal-get-model:
	@mkdir -p models/$(FAMILY)/$(RUN_ID)
	modal volume get wildfire-models /$(FAMILY)/$(RUN_ID)/ models/$(FAMILY)/$(RUN_ID)/

## Browse Modal volumes
modal-ls-models:
	modal volume ls wildfire-models /
modal-ls-snapshots:
	modal volume ls wildfire-snapshots /

## Tail logs from a running or recent app
modal-logs-denoiser:
	modal app logs wildfire-denoiser-v2 -f
modal-logs-spread:
	modal app logs wildfire-spread-v2 -f
modal-logs-ignition:
	modal app logs wildfire-ignition -f
```

---

### 2.9 Key Implementation Constraints

These are hard requirements the implementation must satisfy. Do not work around them.

**1. Do not modify existing training scripts.**
`train_denoiser_v2.py`, `train_spread_v2.py`, `train_ignition.py` must remain unchanged. All Modal-specific logic lives in `ml/modal/`. Config overrides (output paths, etc.) happen by mutating the loaded config dict before passing to the training function.

**2. Volume paths must mirror local paths.**
The existing training scripts infer artifact paths from config values (`model_output_root`, `out_root`, `out`). Override these values in the config dict after loading, substituting `/models/` for the local `models/` prefix. This ensures artifact paths are predictable.

**3. `volume.commit()` after every write.**
Call `model_volume.commit()` and/or `snapshot_volume.commit()` at the end of every function that writes to a volume mount. Without this, files written in one container may not be visible in subsequent containers. Also call `volume.reload()` at the start of any read-only function that depends on a prior write.

**4. All three local packages must be importable — but only grid/constants are used.**
Training code imports `api.core.grid` (grid math, no DB) and `ingest.weather_repository` (constants). `api.db`, `api.fires.service`, and `api.terrain.window` are only used during snapshot export (which stays local). However, all three packages (`ml/`, `api/`, `ingest/`) must still be added to the container image and `sys.path` because Python resolves all imports at module load time — a missing `api` package will cause an `ImportError` even if the DB-using submodules are never called.

**5. Python 3.11 exactly.**
The `CLAUDE.md` mandate is explicit: Python 3.11, no 3.12+. Use `modal.Image.debian_slim(python_version="3.11")` always.

**6. Secrets must match Modal secret names — and contain no DB credentials.**
Training functions use only `modal.Secret.from_name("wildfire-ml-config")` (non-sensitive ML config). The operator creates this in the dashboard as documented in Part 1, Section 1.3. Do not add `wildfire-postgres` or any DB credentials to training function secret lists — the training containers must not have DB access.

**7. No mock data or placeholder values.**
Per `AGENTS.md` and `CLAUDE.md`: zero-tolerance for fake/dummy data. If any data source is missing (e.g., snapshot parquet not yet exported), the function should fail loudly with a clear error message, not proceed with synthetic data.

**8. Gate report drives promotion, not training success.**
A training run returning `metrics.json` does not mean the model is ready to promote. The `gate_report.json` field `"pass": true` is required before any `model-promote` call. This gate logic lives in the existing training scripts — do not replicate or override it.

---

### 2.10 Testing the Integration Locally

Before running on Modal cloud, test with `.local()` to run functions in the current process:

```python
# In a scratch script or test
from ml.modal.denoiser import train
result = train.local(config_path="configs/denoiser_train_v2.yaml")
print(result)
```

`.local()` runs the function body in the current Python process, skipping container creation. It uses local env vars and local filesystem paths (not volume mounts). This is useful for verifying imports and logic before paying for cloud compute.

For integration testing against a real Modal container (but without GPU cost):

```bash
modal run ml/modal/denoiser.py::train \
  --config-path /configs/denoiser_train_v2.yaml
```

---

### 2.11 Incremental Roll-Out Order

Implement in this order to minimize risk:

1. **`ml/modal/images.py`** — image definition only; verify it builds without error by running a trivial function
2. **`ml/modal/volumes.py`** — volume references only
3. **`ml/modal/denoiser.py`** — denoiser `train` function only (skip snapshot/label/eval initially)
4. **Makefile `modal-denoiser-train-v2` target** — wire up the simplest path first
5. Verify a real training run completes and artifacts appear in the volume
6. Add `ml/modal/spread.py` and `ml/modal/ignition.py`
7. Add remaining pipeline stages (snapshot export, labeling, evaluation)
8. Add `modal-deploy` target and test the deployed-function calling path

---

### 2.12 Common Failure Modes and Mitigations

| Failure | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: api` | Local packages not on `sys.path` in container | Add `sys.path.insert(0, "/pkg/api")` at function entry; verify `add_local_python_source` paths in image |
| `ValueError: snapshot_path must point to /data/...` | Snapshot not uploaded to volume before training | Run `make modal-upload-snapshot VERSION=...` first |
| `FileNotFoundError: /configs/denoiser_train_v2.yaml` | Configs not in container | Add `add_local_dir("./configs", remote_path="/configs")` to image definition |
| Volume files missing after training | `commit()` not called | Call `model_volume.commit()` before returning from any function that writes files |
| `TimeoutError` | Training exceeded `timeout=` | Increase timeout; for spread model set to `28800` (8h) |
| `OperationalError: could not connect to server` | DB credentials accidentally passed to training container | Training functions must not have DB secrets — check `_secrets` list in the Modal app file |
| `gate_report.json: pass=false` | Model did not pass quality gates | This is expected behavior — do not suppress or bypass. Investigate metrics and adjust config/data |
| Image rebuild on every run | Local source added before deps | Ensure `add_local_python_source` calls come **after** all `pip_install`/`uv_sync` calls in image definition |

---

## Appendix: Quick Reference

### One-time operator setup commands
```bash
pip install modal
modal setup
modal secret create wildfire-ml-config \
  DENOISER_REQUIRED=false \
  DENOISER_PIPELINE_VERSION=v2 \
  DENOISER_THRESHOLD_PROFILE=env \
  FIRE_SCORING_WEATHER_TIME_TOLERANCE_HOURS=6 \
  DENOISER_MOISTURE_TIME_TOLERANCE_HOURS=48
modal volume create wildfire-models
modal volume create wildfire-snapshots
```

> No database credentials needed in Modal. Snapshot export runs locally.

### Recurring operator commands
```bash
modal volume ls wildfire-models /denoiser_v2/
modal volume get wildfire-models /denoiser_v2/<run_id>/ ./models/denoiser_v2/<run_id>/
modal app logs wildfire-denoiser-v2 -f
```

### Dev agent test commands
```bash
modal run ml/modal/denoiser.py::train --config-path /configs/denoiser_train_v2.yaml
modal run --timestamps ml/modal/spread.py --config /configs/spread_train_v2.yaml
modal deploy ml/modal/denoiser.py
```
