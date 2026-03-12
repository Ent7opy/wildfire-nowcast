# Ops Runbook

## 1) Bootstrap (one-time per environment)

```bash
make install
make db-up
make migrate
make prepare
```

Validate:

```bash
curl http://localhost:8000/health/data-freshness
curl http://localhost:8000/internal/models/active
curl http://localhost:8000/internal/health/data-freshness
```

## 2) Continuous runtime operations

Run scheduler locally:

```bash
make ops-start
```

Or in compose (full stack, root repo):

```bash
docker compose up -d
```

For scheduler-only iteration:

```bash
docker compose up ingest_scheduler -d
```

Expected runtime profile:
- FIRMS bootstrap ingest on first run: last 6 hours (`FIRMS_INITIAL_LOOKBACK_MINUTES=360`).
- FIRMS recurring ingest: every 30 minutes, scoped to the last 30 minutes (`FIRMS_INCREMENTAL_LOOKBACK_MINUTES=30`).
- Perimeters refresh daily.
- Freshness checks enabled.
- Retries enabled (`max_retries=3`, `backoff=20s`).

## 3) Model lifecycle (register/promote/rollback)

Register artifact with strict runtime contract:

```bash
make model-register FAMILY=denoiser ARTIFACT=models/denoiser_v2/<run_id> \
  METRICS=@models/denoiser_v2/<run_id>/metrics.json \
  RUNTIME_CONTRACT=@configs/denoiser_runtime_contract_v1_strict_20260304_235923.json
```

Promote champion:

```bash
make model-promote FAMILY=denoiser MODEL_ID=<model_id>
```

Verify active runtime contract:

```bash
uv run --project api scripts/model_registry.py active
curl http://localhost:8000/internal/models/active
```

Patch contract on an already-registered model (if needed):

```bash
make model-update-contract FAMILY=denoiser MODEL_ID=<model_id> \
  RUNTIME_CONTRACT=@configs/denoiser_runtime_contract_v1_strict_20260304_235923.json
```

Rollback to previously promoted model:

```bash
make model-rollback FAMILY=denoiser
```

## 4) Training + auto-promotion helpers

```bash
make train-denoiser
make train-spread
```

These targets train, register the latest run, and promote it as champion.

## 5) Incident playbook: data recency / empty-map issues

Symptoms:
- Map shows no fires for recent windows.
- Forecast generation is slow due repeated on-demand weather/terrain ingestion.

Immediate checks:

```bash
curl http://localhost:8000/health/data-freshness
curl http://localhost:8000/internal/health/data-freshness
```

Actions:
- If FIRMS latest run has zero fetched rows, run a manual FIRMS ingest for the intended area/day window.
- If weather/terrain is missing for frequent forecast AOIs, prewarm those inputs (optional) to reduce click-to-forecast latency.
- If FIRMS ingest fails due denoiser policy, verify the promoted denoiser has `metrics_json.runtime_contract` and runtime env uses `DENOISER_PIPELINE_VERSION=v2` + `DENOISER_THRESHOLD_PROFILE=strict_v1`.
- Set `DENOISER_ALLOW_UNSAFE_THRESHOLD_OVERRIDE=true` only as an emergency local/dev override.

Recovery verification:
- `/health/data-freshness` reports recent `sources.*.last_seen_at` values.
- Scheduler dashboard updates at `data/ingest/orchestrator_dashboard.json`
