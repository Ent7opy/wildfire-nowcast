# RUNBOOK — Wildfire Nowcast MVP Execution Plan

This document is derived from `status_scan.txt` (the input Status Scan + Next Steps report). It is designed to be orchestrator-ready: tasks are split into “one fresh headless session” units, each with dependencies, success checks, and rollback.

## 1) MVP Gates

### Gate 0 — Contributor workflow is unblocked on Windows
- **Why it matters**: MVP tasks require running `make` and UI lint/tests; current Windows friction blocks iteration (see `status_scan.txt` lines 35–37, 146–153; tasks WN-001/WN-002).
- **Verify**
  - `make help`
    - **Pass looks like**: prints a target list including at least `install`, `dev-api`, `dev-ui`, `test`, `lint`.
  - `make lint`
    - **Pass looks like**: completes API + UI lint without the `ui\.venv\lib64` “Access is denied” error.
  - `make test`
    - **Pass looks like**: completes API + UI tests without the same UV removal error.
- **If red (fallback plan)**: document “use WSL/Git Bash” as the supported path and proceed with API-only tasks in a Linux-like environment; keep Windows fixes queued.

### Gate 1 — Real fires can be ingested and rendered on the map
- **Why it matters**: This is the MVP “it’s real” signal.
- **Verify**
  - Bring stack up: `docker compose up -d`
    - **Pass looks like**: services start (api/ui/db/tiles).
  - After following the repo’s documented ingest flow (`docs/GETTING_STARTED.md`), query:
    - `curl "http://127.0.0.1:8000/fires?min_lon=...&min_lat=...&max_lon=...&max_lat=...&start_time=...&end_time=..."`
      - **Pass looks like**: HTTP 200 and a non-empty response **after** ingest.
  - UI map renders fires tiles:
    - **Pass looks like**: Streamlit map loads and shows points from `/tiles/fires/{z}/{x}/{y}.pbf?...`.
- **If red (fallback plan)**: validate DB/migrations/ingest flow per `docs/GETTING_STARTED.md`; if API is up but map is blank, validate `/tiles/fires/...` proxy path.

### Gate 2 — On-demand baseline forecast produces a mappable overlay
- **Why it matters**: MVP requires a visible forecast overlay; UI is wired but backend returns unmappable results today.
- **Verify**
  - Generate run:
    - `curl -X POST "http://127.0.0.1:8000/forecast/generate" ...`
      - **Pass looks like**: response includes a **non-null** `run.id`.
  - Fetch forecast contour tiles:
    - `curl -I "http://127.0.0.1:8000/tiles/forecast_contours/{z}/{x}/{y}.pbf?run_id=<RUN_ID>"`
      - **Pass looks like**: HTTP 200 and non-empty tile payload.
  - UI: click “Generate Spread Forecast”
    - **Pass looks like**: overlay appears without manual steps.
- **If red (fallback plan)**: run the persisted forecast CLI path (`ingest/spread_forecast.py`) and validate that `GET /forecast` can see it.

### Gate 3 — Fires export is user-accessible from the UI (sync)
- **Why it matters**: Exports are a core “useful artifact” outcome for analysts.
- **Verify**
  - API export exists:
    - `curl -I "http://127.0.0.1:8000/fires/export?min_lon=...&min_lat=...&max_lon=...&max_lat=...&start_time=...&end_time=...&format=csv"`
      - **Pass looks like**: HTTP 200 with CSV content headers.
  - UI export exists:
    - **Pass looks like**: “Export fires” downloads a file (CSV/GeoJSON) for current bbox/time.
- **If red (fallback plan)**: keep API-only export and document how to call `/fires/export`.

### Gate 4 — Baseline “risk” layer exists and can be toggled in UI
- **Why it matters**: UI already advertises “Risk index”; backend risk endpoints are not present today.
- **Verify**
  - `curl "http://127.0.0.1:8000/risk?min_lon=...&min_lat=...&max_lon=...&max_lat=...&start_time=...&end_time=..."`
    - **Pass looks like**: HTTP 200 and a renderable response per the minimal contract implemented.
  - UI toggle “Risk index”
    - **Pass looks like**: layer renders and is labeled baseline.
- **If red (fallback plan)**: remove/disable the UI toggle until backend exists (prefer integrity over misleading UX).

### Gate 5 — Quality + docs cover MVP-critical surfaces
- **Why it matters**: prevent regressions and make the system runnable by others; CI currently omits `ml/` and `ingest/`.
- **Verify**
  - CI includes lint/tests for `ml` and `ingest` (WN-008)
    - **Pass looks like**: PR checks include them and succeed.
  - Forecast overlay operational doc exists (WN-009)
    - **Pass looks like**: copy/paste steps produce a visible overlay.
  - Remove environment-specific UI debug log path (WN-010)
    - **Pass looks like**: UI no longer writes to the hard-coded path.
- **If red (fallback plan)**: revert CI expansion and keep docs + core functionality stable.

## 2) Task Graph Summary (DAG)

- **WN-001** (infra) Make `make help` work on Windows → unblocks Gate 0 usability.
- **WN-002** (infra) Fix UI UV Windows permission failure → unblocks: WN-004/WN-005/WN-006/WN-010.
- **WN-003** (api) Persist `POST /forecast/generate` → unblocks: WN-004/WN-009.
- **WN-004** (ui) Use generated forecast `run_id` for overlay → blocked by: WN-002, WN-003 → unblocks WN-009.
- **WN-005** (api+ui) Add baseline `/risk` endpoint + UI layer → blocked by: WN-002.
- **WN-006** (ui) Add “Export fires” download → blocked by: WN-002.
- **WN-007** (api) Implement real async export worker → optional enhancement; independent.
- **WN-008** (qa) Extend CI to run `ml/` and `ingest/` → independent.
- **WN-009** (docs) Document forecast overlay workflow → blocked by: WN-003, WN-004.
- **WN-010** (ui) Remove hard-coded debug log path → blocked by: WN-002.

Parallelizable (safe to run in separate sessions): WN-001, WN-003, WN-007, WN-008.

## 3) Run Order (grouped by Gate) + checkpoints

### Gate 0 tasks
- Run: WN-001 → WN-002

Checkpoint 0 (QA)
- Run: `make help`, `make lint`, `make test`
- **Green**: all succeed on Windows PowerShell without the UV `.venv\lib64` error.
- **Red**: document WSL/Git Bash fallback and continue with API-only tasks while keeping WN-002 queued.

### Gate 1 tasks
- Run: (No new code required; follow `docs/GETTING_STARTED.md` ingest flow)

Checkpoint 1 (QA)
- Run: `docker compose up -d`
- Run: `curl "http://127.0.0.1:8000/fires?..."`
- **Green**: non-empty results after ingest; UI shows fires tiles.
- **Red**: validate migrations/ingest and `/tiles/fires` proxy behavior.

### Gate 2 tasks
- Run: WN-003 → WN-004

Checkpoint 2 (QA)
- Run: `cd api && uv run pytest`
- Run: generate run + fetch contour tiles (see Gate 2 verify)
- **Green**: non-null `run.id`; contour tiles return HTTP 200; UI overlay appears after generation.
- **Red**: use `ingest/spread_forecast.py` persisted run path and validate via `GET /forecast`.

### Gate 3 tasks
- Run: WN-006 (optional: WN-007)

Checkpoint 3 (QA)
- Run: `curl -I "http://127.0.0.1:8000/fires/export?...&format=csv"`
- UI: click “Export fires”
- **Green**: download works and matches bbox/time filters.
- **Red**: keep API-only export and document usage.

### Gate 4 tasks
- Run: WN-005

Checkpoint 4 (QA)
- Run: `curl "http://127.0.0.1:8000/risk?..."`
- UI: toggle “Risk index”
- **Green**: layer renders and is labeled baseline.
- **Red**: disable/remove UI toggle until backend exists.

### Gate 5 tasks
- Run: WN-010 → WN-008 → WN-009

Checkpoint 5 (QA)
- Run: `make lint`, `make test`
- CI: verify new jobs exist and pass
- **Green**: CI covers api/ui/ml/ingest; docs match behavior.
- **Red**: revert CI expansion, keep docs updated for current supported flow.

## 4) Task Specs

### WN-001 — Make `make help` work on Windows
- **Type**: infra
- **Blocked by**: none
- **Unblocks**: WN-002
- **Outcome**: `make help` prints targets on Windows PowerShell without `grep/awk`.
- **Scope**
  - Do: modify only the `help` target in `Makefile`.
  - Won’t do: refactor other targets.
- **Where to look**: `Makefile` (help target uses `grep` + `awk` today).
- **Implementation notes**: Use a portable Python one-liner to parse `##` comments.
- **Commands**: `make help`
- **Acceptance criteria**: Succeeds and lists at least `install/dev/test/lint` targets.
- **Rollback plan**: revert `Makefile` help target to prior implementation.
- **Handoff**: note output format changes (if any).

### WN-002 — Fix UI UV Windows permission failure
- **Type**: infra
- **Blocked by**: none
- **Unblocks**: WN-004, WN-005, WN-006, WN-010
- **Outcome**: `make test` and `make lint` complete UI steps on Windows without `.venv\\lib64` removal “Access is denied”.
- **Scope**
  - Do: minimal changes to eliminate the failure.
  - Won’t do: restructure dependency management.
- **Where to look**: `ui/` venv layout; `Makefile` `test`/`lint` targets.
- **Implementation notes**: likely stale symlink/dir or filesystem lock; prefer deterministic fix.
- **Commands**: `make lint`, `make test`
- **Acceptance criteria**: UI lint/test run without the specific UV error.
- **Rollback plan**: revert changes; if docs/cleanup only, remove additions and document manual cleanup.
- **Handoff**: record root cause + fix rationale.

### WN-003 — Persist on-demand forecasts from `POST /forecast/generate`
- **Type**: api
- **Blocked by**: none
- **Unblocks**: WN-004, WN-009
- **Outcome**: `POST /forecast/generate` returns non-null `run.id` and produces DB-backed contours usable via `/tiles/forecast_contours?...run_id=...`.
- **Scope**
  - Do: implement persistence using existing DB tables + existing contour/raster code paths.
  - Won’t do: new architecture; model training.
- **Where to look**: `api/routes/forecast.py`, `api/forecast/repo.py`, `ingest/spread_forecast.py`.
- **Implementation notes**: endpoint currently returns empty rasters/contours; change it to create a run, insert rasters/contours, finalize status.
- **Commands**
  - `cd api && uv run pytest`
  - Manual:
    - `curl -X POST "http://127.0.0.1:8000/forecast/generate" ...` (expect non-null `run.id`)
    - `curl -I "http://127.0.0.1:8000/tiles/forecast_contours/{z}/{x}/{y}.pbf?run_id=<RUN_ID>"` (expect HTTP 200)
- **Acceptance criteria**: persisted records exist and contour tiles are fetchable by `run_id`.
- **Rollback plan**: revert endpoint changes to restore metadata-only behavior.
- **Handoff**: document minimal request payload and required inputs (weather selection, limits).

### WN-004 — UI: use generated forecast `run_id` for overlay
- **Type**: ui
- **Blocked by**: WN-002, WN-003
- **Unblocks**: WN-009
- **Outcome**: When a user generates a forecast, the map overlay renders contours for that run.
- **Scope**
  - Do: UI-only changes.
  - Won’t do: redesign UI.
- **Where to look**: `ui/components/click_details.py`, `ui/components/map_view.py`.
- **Implementation notes**: overlay already supports `run_id`; ensure session state is set and used.
- **Commands**: run `make dev-api` + `make dev-ui`; click “Generate Spread Forecast”.
- **Acceptance criteria**: overlay updates immediately after generation without manual refresh.
- **Rollback plan**: revert UI session-state wiring.
- **Handoff**: note session-state keys used.

### WN-005 — Add baseline `/risk` endpoint (no model training)
- **Type**: api + ui
- **Blocked by**: WN-002
- **Unblocks**: none
- **Outcome**: `/risk` exists and UI “Risk index” toggle renders a baseline layer labeled as baseline.
- **Scope**
  - Do: simplest viable baseline (explicitly labeled).
  - Won’t do: ML training or ambiguous semantics.
- **Where to look**: API routers pattern (`api/main.py` wiring); UI toggle exists (`ui/components/sidebar.py`, `ui/components/legend.py`).
- **Implementation notes**: define semantics explicitly (ignition risk vs spread risk proxy) and keep contract minimal.
- **Commands**: `make lint`, `make test`, `curl "http://127.0.0.1:8000/risk?..."`
- **Acceptance criteria**: endpoint returns renderable data; UI renders it and labels baseline.
- **Rollback plan**: remove route and UI wiring; optionally hide toggle.
- **Handoff**: document baseline risk definition in route docstring + UI legend.

### WN-006 — UI: add “Export fires” download
- **Type**: ui
- **Blocked by**: WN-002
- **Unblocks**: none
- **Outcome**: UI can download CSV/GeoJSON for current bbox/time via `GET /fires/export`.
- **Scope**
  - Do: UI-only.
  - Won’t do: API changes unless strictly required.
- **Where to look**: `api/routes/exports.py`, `ui/api_client.py`, `ui/components/sidebar.py` or `ui/app.py`.
- **Implementation notes**: use `API_PUBLIC_BASE_URL` for browser-accessible download URL.
- **Commands**: run `make dev-ui`; click “Export fires”.
- **Acceptance criteria**: file downloads; contents match bbox/time filters.
- **Rollback plan**: remove the UI button/link.
- **Handoff**: note how bbox/time is sourced from map state.

### WN-007 — Implement real async export worker (remove TODO)
- **Type**: api
- **Blocked by**: none
- **Unblocks**: none
- **Outcome**: async export worker produces a real file artifact and updates `export_jobs` so download works.
- **Scope**
  - Do: implement one export kind end-to-end (e.g., fires_csv).
  - Won’t do: generalized job framework.
- **Where to look**: `api/routes/exports.py`, `api/exports/worker.py`.
- **Commands**: `docker compose up --build` (worker included), then create an export job via API.
- **Acceptance criteria**: job transitions queued→running→succeeded and download works.
- **Rollback plan**: revert worker; keep sync exports.
- **Handoff**: record where artifacts are stored and how they’re served.

### WN-008 — Extend CI to run `ml/` and `ingest/`
- **Type**: qa
- **Blocked by**: none
- **Unblocks**: none
- **Outcome**: CI runs ruff/pytest for `ml` and `ingest`.
- **Scope**: minimal changes to `.github/workflows/ci.yml`.
- **Where to look**: `.github/workflows/ci.yml`.
- **Commands (local mimic)**:
  - `cd ml && uv sync --dev && uv run pytest`
  - `cd ingest && uv sync --dev && uv run pytest`
- **Acceptance criteria**: CI includes ml+ingest jobs and they pass.
- **Rollback plan**: revert workflow changes.
- **Handoff**: note any required skips/mocks for CI stability.

### WN-009 — Document the “forecast overlay” operational workflow
- **Type**: docs
- **Blocked by**: WN-003, WN-004
- **Unblocks**: none
- **Outcome**: doc section that is copy/paste runnable and matches current code paths for weather ingest + forecast + UI overlay.
- **Scope**: docs only.
- **Where to look**: `docs/GETTING_STARTED.md`, `docs/architecture.md`, `ingest/spread_forecast.py`.
- **Acceptance criteria**: instructions are runnable and match implemented endpoints.
- **Rollback plan**: revert doc changes.
- **Handoff**: keep secrets out; reference env vars without copying values.

### WN-010 — Remove hard-coded debug log path from UI
- **Type**: ui
- **Blocked by**: WN-002
- **Unblocks**: none
- **Outcome**: UI no longer writes to a hard-coded Windows-only debug path.
- **Scope**: minimal deletion of debug logging blocks only.
- **Where to look**: `ui/components/map_view.py`.
- **Commands**: `make lint`, `make test`, then run UI and confirm no file writes to that path.
- **Acceptance criteria**: no writes; behavior unchanged.
- **Rollback plan**: revert removal if required (prefer keeping it removed).
- **Handoff**: if logging still needed, use existing repo-standard logging approach (do not add new deps).

## 5) Global risks & fallbacks (top 5)

1. **Forecast persistence mismatch** (UI expects `run_id`, API returns `None` today).  
   Mitigation: WN-003 first; keep CLI persisted forecast fallback.
2. **Risk semantics ambiguity** (UI advertises risk, API has none).  
   Mitigation: minimal baseline definition, clearly labeled; keep contract simple.
3. **Windows filesystem locks** (UV permission error).  
   Mitigation: WN-002 early; fallback to WSL/Git Bash documented path.
4. **Weather run selection unclear** (operational selection path not visible in scan).  
   Mitigation: in WN-003/WN-009, document the selection behavior used (e.g., “latest completed run”) and keep it explicit.
5. **CI expansion may introduce flaky deps**.  
   Mitigation: keep CI lean, avoid integration tests requiring network downloads; run unit tests + lint.

---

## 6) JIT Forecast Pipeline Observability

This section documents how to monitor and troubleshoot the Just-In-Time (JIT) forecast pipeline worker (`api/forecast/worker.py`).

### Worker Architecture
- The JIT worker runs as a separate Docker Compose service (`worker`) that processes RQ tasks from Redis.
- Tasks are enqueued by the API when users trigger JIT forecasts via `POST /forecast/jit`.
- Each task updates the `jit_forecast_jobs` table with status transitions: `pending` → `ingesting_terrain` → `ingesting_weather` → `running_forecast` → `completed` or `failed`.

### Inspecting Worker Logs
All worker operations include structured logging with `job_id` context. View logs via:

```bash
docker compose logs worker -f
```

Example log entries:
```
JIT forecast pipeline started: job_id=<UUID>, bbox=(20.0, 40.0, 21.0, 41.0)
JIT job <UUID>: starting terrain ingestion
JIT job <UUID>: starting weather ingestion
JIT job <UUID>: starting forecast
JIT forecast pipeline completed: job_id=<UUID>
```

On failure, logs include full traceback:
```
JIT forecast pipeline failed: job_id=<UUID>, error=ValueError: ...
Traceback (most recent call last):
  ...
```

### Checking Worker Health
The worker service has a healthcheck configured but currently disabled (`healthcheck: disable: true` in `docker-compose.yml`). This is acceptable for development but should be enabled for production deployments.

To verify the worker is running:
```bash
docker compose ps worker
```

Expected output shows `worker` service status as `running`.

### Querying Job Status via API
Check the status of any JIT forecast job:
```bash
curl http://localhost:8000/forecast/jit/<JOB_ID>
```

Response includes:
- `status`: current pipeline stage or terminal state (`completed`, `failed`)
- `progress_message`: user-friendly description of current activity
- `result`: forecast outputs on success (raster URLs, contour GeoJSON)
- `error`: error message on failure

### Querying Job Status via Database
Directly inspect the `jit_forecast_jobs` table:
```sql
SELECT id, status, created_at, updated_at, error
FROM jit_forecast_jobs
WHERE id = '<JOB_ID>';
```

### Common Failure Scenarios
1. **Terrain ingestion timeout**: Job stuck in `ingesting_terrain` state. Check worker logs for Copernicus DEM download failures or network timeouts.
2. **Weather ingestion failure**: Job fails during `ingesting_weather` with GRIB download errors. Verify NOAA NOMADS availability and network connectivity.
3. **Forecast execution error**: Job fails during `running_forecast`. Check logs for missing weather/terrain data or model errors.

### Troubleshooting Workflow
1. Check job status via API: `GET /forecast/jit/<JOB_ID>`
2. If status is `failed`, inspect `error` field for high-level cause
3. View detailed worker logs: `docker compose logs worker -f`
4. Query database for job record: `SELECT * FROM jit_forecast_jobs WHERE id = '<JOB_ID>'`
5. For ingestion failures, verify external dependencies (Copernicus DEM, NOAA GFS)

### Restarting the Worker
If the worker becomes unresponsive:
```bash
docker compose restart worker
```

In-flight jobs will fail and must be retried by the user.

