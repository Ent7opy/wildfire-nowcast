# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## North Star

Wildfire Nowcast is the globally deployable, hourly-refreshed ground truth for active fire events — giving incident commanders, dispatchers, researchers, and emergency managers a single, trusted picture of where fires are burning, how confident we are, and where they are headed, so that every operational decision is grounded in the best available evidence rather than guesswork or stale data.

## Scientific Engineering Mandate

This is a production-grade wildfire decision support system. Per `AGENTS.md`:

- **Zero-tolerance for mocking**: Never use fake, dummy, or placeholder data unless explicitly requested. If a real data schema is missing, ask for it.
- **Hard stops are mandatory**: Use STOP/BLOCKER when an authoritative source is missing, feature contracts mismatch between train/infer, or geo alignment is invalid.
- **Warn, don't bypass**: Stage-gap WARNINGs must include a mitigation action and target stage (`science_grade`). A WARNING cannot replace a STOP/BLOCKER.
- **Push back on shortcuts**: If the user suggests a quick workaround that compromises scientific integrity, flag it: "We've had to rewrite this before because of shortcuts. Let's do it the real way now."

Maturity stages: `mvp_operational` (working end-to-end) → `science_grade` (promotion target).

## Commands

### Setup
```bash
make doctor           # Verify dev environment (Python, Node, uv, Docker, .env)
make install          # Install all deps: api, ui, ml, ingest
cp .env.example .env  # Then fill in FIRMS_MAP_KEY and DB credentials
make db-up            # Start PostGIS + Redis
make migrate          # Run Alembic migrations
make prepare          # Bootstrap: migrate + ingest + apply filters
```

### Development
```bash
make dev-api          # FastAPI on http://localhost:8000
make dev-ui           # Vite on http://localhost:8501
make health-check     # Verify running services
```

### Testing
```bash
make test                                          # All non-integration tests
cd api && uv run pytest -m "not integration"       # API only
cd api && uv run pytest tests/path/test_file.py   # Single test file
cd ui && npm run test:run                          # UI (vitest with coverage)
cd ml && uv run pytest -m "not integration"        # ML only
cd ingest && uv run pytest -m "not integration"    # Ingest only
```

### Linting
```bash
make lint             # ruff (Python) + eslint + tsc (UI)
make lint-fix         # Auto-fix
cd ui && npm run typecheck   # TypeScript only
```

### Database Migrations
```bash
make migrate                    # Apply migrations
make revision msg="description" # Create new Alembic revision
```

### ML Pipelines
```bash
# Denoiser v2 (current standard)
make denoiser-eventize ARGS="--batch-id ..."
make denoiser-label-v2 ARGS="--start ... --end ..."
make denoiser-snapshot-v2 ARGS="--bbox ... --start ... --end ... --version ..."
make denoiser-train-v2 CONFIG=configs/denoiser_train_v2.yaml
make denoiser-eval-v2 MODEL_RUN=models/denoiser_v2/<run_id> SNAPSHOT=... OUT=reports/denoiser_v2/<run_id>

# Full pipeline (train → eval → register → promote)
make train-denoiser TRAIN_DENOISER_PIPELINE=v2
make train-spread TRAIN_SPREAD_PIPELINE=v2
```

### Model Registry
```bash
make model-register FAMILY=denoiser ARTIFACT=... METRICS=@path/metrics.json RUNTIME_CONTRACT=@path/contract.json
make model-promote FAMILY=denoiser MODEL_ID=...
make model-rollback FAMILY=denoiser
```

### Ingest Operations
```bash
make ingest-orchestrator      # One-shot ingest (FIRMS + weather + terrain + perimeters)
make ops-start                # Continuous ingest scheduler
```

## Architecture

### System Loop
```
Ingest → Interpret → Deliver → Learn
```

1. **Ingest** (`ingest/`): Collect and normalize external geospatial signals
2. **Interpret** (`ml/`, `api/`): Derive nowcast, spread outlook, risk estimates
3. **Deliver** (`api/`, `ui/`): Expose map layers, queries, exports
4. **Learn** (`ml/`): Evaluate outcomes, retrain, promote models

### Services (docker-compose)
| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI backend |
| ui | 8501 | React/Vite frontend |
| db | 5433 | PostgreSQL 16 + PostGIS 3.5 |
| redis | 6379 | Cache + job queue |
| worker | — | RQ async workers |
| titiler | 8080 | COG raster tile server |
| tiles | 7800 | pg_tileserv vector tiles |
| ingest_scheduler | — | Continuous ingest loop |

### API (`api/`)
FastAPI with routers: fires, forecast, aois, exports, risk, tiles, internal, archive, assistant. Key modules:
- `routes/archive.py`: Multi-day historical replay — POST `/fires/archive/ingest-range`, GET status; validates against `MAX_ARCHIVE_RANGE_DAYS` (default 7) and FIRMS 10-day lookback
- `notifications.py`: Fire-and-forget webhook (Slack/Discord-compatible) + SMTP email alerts; rate-limited per event type; wired into orchestrator job failures, denoiser drift violations, data staleness, and model promote/rollback
- `industrial_coverage.py`: GET `/internal/health/industrial-coverage?bbox=...` — spread service emits a WARNING (never a BLOCKER) when forecasting a bbox with zero industrial sources
- `routes/forecast.py`: Forecast endpoint streams SSE via `StreamingResponse`; use `asyncio.get_running_loop()` not `get_event_loop()` in async generators here
- `model_registry.py`: Model lifecycle (register/promote/rollback)
- `data_status.py`: Data freshness monitoring
- `migrations/`: Alembic schema (run via `cd api && uv run alembic`)

### Ingest (`ingest/`)
Orchestrated by `orchestrator.py` with watermark-based incremental ingestion:
- **FIRMS**: `firms_ingest.py` — fire detections with denoiser v2 inference applied inline; uses threshold profiles (`strict_v1`, `env`, `unsafe`)
- **Weather**: `weather_ingest.py` — GFS 0.25° GRIB forecasts with bias correction
- **Terrain**: `terrain_features.py` — DEM stitching, elevation/aspect/slope
- **Perimeters**: `nifc_perimeters_ingest.py`, `cwfis_authority_ingest.py`, `wfigs_authority_ingest.py`, `copernicus_ems_authority_ingest.py`
- **Fuels**: `fuels_ingest.py`, `lulc_worldcover_ingest.py`, `lfmc_ecland_ingest.py`
- **Industrial sources**: `industrial_sources_ingest.py` — no-go zones for noise filtering

### ML (`ml/`)
**Denoiser** (fire vs. noise classification):
- v2 is the current standard: event-based labeling via `denoiser/eventize.py`, feature engineering, XGBoost classifier exported to ONNX
- `denoiser_inference_v2.py`: Runtime inference with runtime contract validation
- Training artifacts stored under `models/denoiser_v2/<run_id>/` with `metrics.json` and `gate_report.json`

**Spread Forecasting**:
- `spread/`: Hindcast builder, champion-challenger evaluation, calibration
- `spread_features.py`: Shared feature engineering (weather cube loading, bias correction)
- v2 requires a gate report (champion-challenger eval) before promotion

**Model Registry Gate**:
- Models must pass `gate_report.json` (field `"pass": true`) to be promoted
- Denoiser v2 also requires `coverage_data_freshness.fresh: true`

### UI (`ui/src/`)
React 18 + TypeScript with:
- **Map**: Deck.GL 9.1 (GPU layers) + MapLibre GL 5.3
- **State**: Zustand stores (`state/`)
- **Data fetching**: React Query (`@tanstack/react-query`)
- **Components**: MUI 7 + custom map/layer components
- `components/ReviewQueuePanel`: Operator triage UI — lists flagged detections, resolve as confirmed fire or noise; `review_required` events render amber on map
- `components/ArchiveRangeScrubber`: Day-range replay scrubber with play/pause/step/speed controls; polls ingest status every 5 s
- `components/AIChatAssistant`: Gemini-backed AI assistant (configured via `VITE_GEMINI_*`); circuit breaker CLOSED→OPEN→HALF_OPEN, health polled at `/assistant/health` (60 s healthy, 15 s degraded)

## Key Configuration

`.env` variables (from `.env.example`):
- `FIRMS_MAP_KEY` — required for FIRMS ingestion
- `POSTGRES_*` — database connection
- `DENOISER_REQUIRED`, `DENOISER_PIPELINE_VERSION`, `DENOISER_THRESHOLD_PROFILE` — inference gates
- `VITE_GEMINI_*` — AI assistant in UI
- `NOTIFICATION_WEBHOOK_URL` — Slack/Discord webhook for ops alerts (optional)
- `NOTIFICATION_EMAIL_TO`, `NOTIFICATION_SMTP_*` — SMTP email alerts (all required together to enable)
- `NOTIFICATION_RATE_LIMIT_SECONDS` — dedupe window per event type (default 900 s)
- `GEMINI_TIMEOUT_SECONDS`, `GEMINI_CIRCUIT_BREAKER_THRESHOLD`, `GEMINI_CIRCUIT_BREAKER_COOLDOWN_SECONDS` — AI assistant resilience
- `MAX_ARCHIVE_RANGE_DAYS` — cap on historical replay range (default 7; FIRMS hard limit is 10 days)
- `INTER_DAY_DELAY_SECONDS` — pacing between per-day RQ jobs in archive ingest

Configs in `configs/` drive ML training and ingestion parameters (YAML).

## Technology Stack

- **Python 3.11** (exact version — no 3.12+), managed with `uv`
- **Node.js 20+**, managed with `npm`
- **Ruff** for Python linting; **ESLint + tsc** for TypeScript
- **pytest** with `not integration` marker for unit tests
- **ONNX/ONNXRuntime** for model serialization and inference
- **xarray/rioxarray** for multidimensional geospatial data; **rasterio/Shapely** for vector/raster ops
