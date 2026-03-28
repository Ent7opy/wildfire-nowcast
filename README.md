# Wildfire Nowcast

> **North Star**: The globally deployable, hourly-refreshed ground truth for active fire events — giving incident commanders, dispatchers, researchers, and emergency managers a single, trusted picture of where fires are burning, how confident we are, and where they are headed, so that every operational decision is grounded in the best available evidence rather than guesswork or stale data.

Wildfire Nowcast is a map-first decision support product for understanding active fires, near-term spread, and short-horizon risk using open geospatial signals.

## What We Are Building

- A reliable **nowcast** of active fire activity.
- A practical **24-72 hour spread outlook** with uncertainty.
- A clear **risk surface** for prioritization and response.
- Fast **area-based summaries** for analysts and operations teams.

## Product Principles

- **Operational clarity** over model novelty.
- **Uncertainty is explicit**, never hidden.
- **Global by default**, region-tunable where needed.
- **Human-in-the-loop** workflows, not full automation.

## Repository Areas

- `api/`: serving layer and product interfaces.
- `ui/`: map experience and analyst workflows.
- `ingest/`: external data acquisition and normalization.
- `ml/`: model training, evaluation, and inference assets.
- `infra/`: local/dev deployment scaffolding.

## Getting Started

```bash
make help
make install
make db-up
make migrate
make dev-api
make dev-ui
```

For documentation, see [`docs/README.md`](docs/README.md).

## Prepare Flow

Bootstrap the full local environment:

```bash
make prepare
```

What it does:
- Runs DB migrations.
- Cleans stale operational records.
- Runs one-shot orchestrated ingest for FIRMS + weather + terrain + perimeters.
- Applies incremental FIRMS watermark filtering so only new detections are processed.

Override defaults:

```bash
make prepare PREPARE_BBOX="-125 24 -66 50" PREPARE_FIRMS_AREA="-125,24,-66,50" PREPARE_REGION="conus"
make prepare PREPARE_JOBS="weather,terrain,perimeters" PREPARE_MAX_RETRIES=3
```

ML application during prepare:
- Denoiser v2 inference applied inline when enabled (`DENOISER_REQUIRED=true`).
- Default: `DENOISER_PIPELINE_VERSION=v2` + `DENOISER_THRESHOLD_PROFILE=strict_v1`.

## Ingestion Orchestrator

```bash
make ingest-orchestrator        # one-shot
make ops-start                  # continuous scheduler
make ingest-orchestrator ARGS="--loop --max-retries 3 --retry-backoff-seconds 20 --enforce-freshness"
```

Dashboard written to `data/ingest/orchestrator_dashboard.json`.

## Health & Model Lifecycle

```bash
curl http://localhost:8000/health/data-freshness
curl http://localhost:8000/internal/health/data-freshness
curl http://localhost:8000/internal/models/active
```

```bash
make model-register FAMILY=denoiser ARTIFACT=models/denoiser_v2/<run_id> \
  METRICS=@models/denoiser_v2/<run_id>/metrics.json \
  RUNTIME_CONTRACT=@configs/denoiser_runtime_contract_v1_strict_20260304_235923.json
make model-promote FAMILY=denoiser MODEL_ID=<model_id>
make model-rollback FAMILY=denoiser
```
