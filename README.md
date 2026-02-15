# Wildfire Nowcast

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

Use the default local workflow from the repository root:

```bash
make help
make install
make db-up
make migrate
make dev-api
make dev-ui
```

For project documentation, start at [`docs/README.md`](docs/README.md).

## Prepare Flow

Use `prepare` as the operational bootstrap command:

```bash
make prepare
```

What it does:
- Runs DB migrations.
- Cleans stale operational records.
- Runs one-shot orchestrated ingest for FIRMS + weather + terrain + perimeters.
- Applies incremental FIRMS watermark filtering (with grace window) so only new detections are processed.

Default local prep window is a Balkans smoke-grid bbox and can be overridden:

```bash
make prepare PREPARE_BBOX="-125 24 -66 50" PREPARE_FIRMS_AREA="-125,24,-66,50" PREPARE_REGION="conus"
```

You can also customize jobs/retries:

```bash
make prepare PREPARE_JOBS="weather,terrain,perimeters" PREPARE_MAX_RETRIES=3
```

ML application during prepare:
- FIRMS denoiser inference is applied for new detections when enabled, and can be enforced via `DENOISER_REQUIRED=true`.
- Active promoted models are resolved from the model registry (`/internal/models/active`) with env fallback compatibility.

## Ingestion Orchestrator

Run FIRMS + weather + terrain + perimeters in one command:

```bash
make ingest-orchestrator
```

Run as a continuous scheduler:

```bash
make ops-start
```

Reliability controls:

```bash
make ingest-orchestrator ARGS="--loop --max-retries 3 --retry-backoff-seconds 20 --enforce-freshness"
```

The orchestrator writes a JSON dashboard by default to:
`data/ingest/orchestrator_dashboard.json`

API/UI stale-data status endpoint:

```bash
curl http://localhost:8000/health/data-freshness
```

Active promoted models endpoint:

```bash
curl http://localhost:8000/internal/models/active
```

Model lifecycle commands:

```bash
make model-register FAMILY=denoiser ARTIFACT=models/denoiser_v1/<run_id> METRICS=@models/denoiser_v1/<run_id>/metrics.json
make model-promote FAMILY=denoiser MODEL_ID=<model_id>
make model-rollback FAMILY=denoiser
```
