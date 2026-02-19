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

Or in compose (dedicated service):

```bash
docker compose up ingest_scheduler -d
```

Expected runtime profile:
- FIRMS poll every 30 minutes.
- Perimeters refresh daily.
- Freshness checks enabled.
- Retries enabled (`max_retries=3`, `backoff=20s`).

## 3) Model lifecycle (register/promote/rollback)

Register artifact:

```bash
make model-register FAMILY=denoiser ARTIFACT=models/denoiser_v2/<run_id> METRICS=@models/denoiser_v2/<run_id>/metrics.json
```

Promote champion:

```bash
make model-promote FAMILY=denoiser MODEL_ID=<model_id>
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
- If FIRMS ingest fails due denoiser policy, ensure promoted denoiser exists or set `DENOISER_REQUIRED=false` only for controlled local/dev fallback.

Recovery verification:
- `/health/data-freshness` reports recent `sources.*.last_seen_at` values.
- Scheduler dashboard updates at `data/ingest/orchestrator_dashboard.json`
