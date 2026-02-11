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

## Ingestion Orchestrator

Run FIRMS + weather + terrain + perimeters in one command:

```bash
make ingest-orchestrator
```

Run as a continuous scheduler:

```bash
make ingest-orchestrator ARGS="--loop --poll-seconds 30"
```
