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
- Weather refresh every 3 hours.
- Perimeters refresh daily.
- Freshness checks enabled.
- Retries enabled (`max_retries=3`, `backoff=20s`).

## 3) Model lifecycle (register/promote/rollback)

Register artifact:

```bash
make model-register FAMILY=denoiser ARTIFACT=models/denoiser_v1/<run_id> METRICS=@models/denoiser_v1/<run_id>/metrics.json
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

## 5) Incident playbook: stale-critical forecast failures

Symptoms:
- `POST /forecast/jit` returns `503` with code `forecast_inputs_stale_or_missing`.
- UI forecast action is blocked with stale/missing source reasons.

Immediate checks:

```bash
curl http://localhost:8000/health/data-freshness
```

Actions:
- If `weather` is stale/missing: run/repair weather ingestion and retry.
- If `terrain` is missing for target area: trigger terrain ingest/prewarm and retry.
- If FIRMS ingest fails due denoiser policy, ensure promoted denoiser exists or set `DENOISER_REQUIRED=false` only for controlled local/dev fallback.

Recovery verification:
- `forecast_gate.can_run == true`
- Scheduler dashboard updates at `data/ingest/orchestrator_dashboard.json`
