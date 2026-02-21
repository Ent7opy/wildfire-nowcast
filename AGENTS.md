# AGENTS.md

## Purpose
Guidance for coding agents working in this repository.

## Project Snapshot
- Monorepo with 4 Python projects: `api/`, `ui/`, `ingest/`, `ml/`.
- Core workflow: ingest signals, serve nowcast/forecast APIs, visualize in Streamlit UI, train/evaluate models.
- Prefer root `Makefile` targets for repeatable operations.

## First Steps
1. Check current workspace state: `git status --short`.
2. Read context in this order when needed:
   - `README.md`
   - `docs/README.md`
   - `docs/WILDFIRE_NOWCAST_101.md`
   - `docs/architecture.md`
   - `docs/SETUP.md`
   - `docs/OPS_RUNBOOK.md`
3. Keep changes scoped; do not revert unrelated local edits.

## Environment Contract
- Python: `3.11` (required by all subprojects).
- Package/tooling: `uv`, `pytest`, `ruff`, Docker for local DB/services.
- Bootstrap:
  - `make doctor`
  - `make install`
  - `make db-up`
  - `make migrate`
  - `make prepare`

## Common Commands
- Start backend: `make dev-api`
- Start UI: `make dev-ui`
- Run full lint: `make lint`
- Run full tests: `make test`
- Service health check: `make health-check`
- Start continuous ingest loop: `make ops-start` (long-running; use only when explicitly needed)

## Test Strategy (Agent Default)
Prefer fast, targeted validation first.

- Lint only touched project:
  - `cd api && uv run ruff check .`
  - `cd ui && uv run ruff check .`
  - `cd ingest && uv run ruff check .`
  - `cd ml && uv run ruff check .`
- Run focused tests for changed modules, then widen scope as needed.
- CI-equivalent default for each project:
  - `cd <project> && uv sync --dev`
  - `cd <project> && uv run pytest -m "not integration"`
- Integration tests (especially in `api/`) may require running DB/services.

## Coding Conventions
- Follow existing patterns and keep edits minimal.
- Ruff settings are per-project (`line-length = 100`, `target-version = py311`).
- Favor explicit typing and clear module boundaries over broad refactors.
- Preserve stable API contracts unless the task explicitly changes them.

## Stateful / Expensive Operations
Run only when the task requires it and call out impact:
- Ingestion commands (`make ingest-*`, `make prepare`, `make ops-start`) mutate DB and can hit external data sources.
- Training commands (`make denoiser-train`, `make train-denoiser`, `make train-spread`, `make denoiser-pipeline`) are compute-heavy and write artifacts under `models/`.
- Model registry actions (`make model-register`, `make model-promote`, `make model-rollback`) change active model state.

## Database & Schema Changes
If schema changes are required:
1. Create migration: `make revision msg="<description>"`
2. Apply migration locally: `make migrate`
3. Add/adjust tests for new behavior.

## Config & Secrets
- Never commit real secrets.
- Use `.env.example` as the contract and keep it updated when adding config.
- `FIRMS_MAP_KEY` is required for live FIRMS ingestion.

## Before Handoff
- Run lint/tests at least for touched projects.
- Report exactly what was run and what was not run.
- Note operational side effects (DB writes, external API calls, long-running jobs).
