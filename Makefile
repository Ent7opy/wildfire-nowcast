.PHONY: help doctor dev-api dev-ui install test lint lint-fix clean db-up db-down migrate revision db-cleanup ingest-firms ingest-firms-backfill ingest-weather ingest-dem ingest-industrial ingest-industrial-authoritative industrial-build-policy industrial-load-no-go-zones industrial-coverage-report br-build-hybrid-curated ingest-viirs ingest-fwi ingest-all repair-fire-detections recompute-fire-scores denoiser-data-coverage-report prepare ops-start smoke-grid smoke-terrain-features denoiser-label denoiser-snapshot denoiser-train denoiser-eval denoiser-eventize denoiser-label-v2 denoiser-snapshot-v2 denoiser-train-v2 denoiser-eval-v2 denoiser-association-report denoiser-drift-monitor denoiser-load-coverage-masks denoiser-build-coverage-masks denoiser-freeze-baseline denoiser-sweep-v2 denoiser-pipeline ingest-nifc-perimeters ingest-authoritative-perimeters ingest-orchestrator download-fuels model-register model-promote model-rollback train-denoiser train-spread hindcast-build spread-champion-challenger weather-bias ralph-init ralph-plan ralph-run ralph-status health-check

PYTHON ?= python3
UV ?= uv
RALPH_TASK_FILE ?=

# Avoid cross-OS venv collisions (e.g., WSL-created venvs on Windows).
ifeq ($(OS),Windows_NT)
    UV_PROJECT_ENVIRONMENT ?= .venv-win
else
    UV_PROJECT_ENVIRONMENT ?= .venv
endif
export UV_PROJECT_ENVIRONMENT

# Ralph detection
ifeq ($(OS),Windows_NT)
    # Windows (CMD or PowerShell)
    RALPH_CMD = @C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -ExecutionPolicy Bypass -File .ralph/ralph.ps1
else
    # Linux / WSL / macOS
    RALPH_CMD = @./.ralph/ralph.sh
endif

help: ## Show this help message
	@echo "Available commands:"
	@$(PYTHON) -c "import re; [print(f'  {m[0]:<20} {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open('Makefile').read(), re.MULTILINE)]"

doctor: ## Check development environment and dependencies
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -Command "Write-Host '[CHECK] Development environment' -ForegroundColor Cyan; Write-Host ''; \
	Write-Host 'Checking Python...'; \
	try { python --version | ForEach-Object { Write-Host \"  [OK] $_\" } } catch { Write-Host '  [FAIL] Python not found' -ForegroundColor Red }; \
	Write-Host ''; \
	Write-Host 'Checking uv...'; \
	try { uv --version | ForEach-Object { Write-Host \"  [OK] $_\" } } catch { Write-Host '  [FAIL] uv not found. Install: https://astral.sh/uv' -ForegroundColor Red }; \
	Write-Host ''; \
	Write-Host 'Checking Docker...'; \
	try { docker --version | ForEach-Object { Write-Host \"  [OK] Docker: $_\" } } catch { Write-Host '  [FAIL] Docker not found' -ForegroundColor Red }; \
	try { docker compose version | ForEach-Object { Write-Host \"  [OK] Docker Compose: $_\" } } catch { Write-Host '  [FAIL] Docker Compose not found' -ForegroundColor Red }; \
	Write-Host ''; \
	Write-Host 'Checking .env file...'; \
	if (Test-Path .env) { Write-Host '  [OK] .env file exists' } else { Write-Host '  [WARN] .env file missing (copy from .env.example)' -ForegroundColor Yellow }; \
	Write-Host ''; \
	Write-Host 'Checking FIRMS_MAP_KEY...'; \
	if (Test-Path .env) { \
		$$content = Get-Content .env -Raw; \
		if ($$content -match 'FIRMS_MAP_KEY=([^\s]+)' -and $$matches[1] -ne 'your_firms_api_key_here') { \
			Write-Host '  [OK] FIRMS_MAP_KEY is set' \
		} else { \
			Write-Host '  [WARN] FIRMS_MAP_KEY not configured' -ForegroundColor Yellow \
		} \
	} else { \
		Write-Host '  [WARN] Cannot check (no .env file)' -ForegroundColor Yellow \
	}; \
	Write-Host ''; \
	Write-Host 'Done.'"
else
	@echo "[CHECK] Development environment"
	@echo ""
	@echo "Checking Python..."
	@$(PYTHON) --version 2>/dev/null && echo "  [OK]" || echo "  [FAIL] Python not found"
	@echo ""
	@echo "Checking uv..."
	@$(UV) --version 2>/dev/null && echo "  [OK]" || echo "  [FAIL] uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
	@echo ""
	@echo "Checking Docker..."
	@docker --version 2>/dev/null && echo "  [OK] Docker installed" || echo "  [FAIL] Docker not found"
	@docker compose version 2>/dev/null && echo "  [OK] Docker Compose installed" || echo "  [FAIL] Docker Compose not found"
	@echo ""
	@echo "Checking .env file..."
	@if [ -f .env ]; then echo "  [OK] .env file exists"; else echo "  [WARN] .env file missing (copy from .env.example)"; fi
	@echo ""
	@echo "Checking FIRMS_MAP_KEY..."
	@if [ -f .env ]; then grep -q "FIRMS_MAP_KEY=" .env && grep "FIRMS_MAP_KEY=" .env | grep -qv "your_firms_api_key_here" && echo "  [OK] FIRMS_MAP_KEY is set" || echo "  [WARN] FIRMS_MAP_KEY not configured"; else echo "  [WARN] Cannot check (no .env file)"; fi
	@echo ""
	@echo "Done."
endif

health-check: ## Check if stack services are running (API, UI, DB)
	@$(PYTHON) scripts/health_check.py

install: ## Install dependencies for all subprojects (with dev extras)
	cd api && $(UV) sync --dev
	cd ui && $(UV) sync --dev
	cd ml && $(UV) sync --dev
	cd ingest && $(UV) sync --dev

dev-api: ## Start FastAPI development server (requires make install)
	cd api && $(UV) run python -m uvicorn api.main:app --app-dir .. --reload --host 127.0.0.1 --port 8000

dev-ui: ## Start Streamlit development server (requires make install)
	cd ui && $(UV) run streamlit run app.py

test: ## Run unit tests (API + UI + ML + Ingest)
	@echo "Running API tests..."
	cd api && $(UV) run pytest
	@echo "Running UI tests..."
ifeq ($(OS),Windows_NT)
	cd ui && $(UV) run pytest
else
	@if [ -L "ui/.venv/lib64" ]; then rm -rf ui/.venv; fi
	cd ui && $(UV) run pytest
endif
	@echo "Running ML tests..."
	cd ml && $(UV) run pytest
	@echo "Running Ingest tests..."
	cd ingest && $(UV) run pytest

lint: ## Run Ruff lint checks (API + UI + ML + Ingest)
	@echo "Linting API..."
	cd api && $(UV) run --no-sync ruff check .
	@echo "Linting UI..."
ifeq ($(OS),Windows_NT)
	cd ui && $(UV) run --no-sync ruff check .
else
	@if [ -L "ui/.venv/lib64" ]; then rm -rf ui/.venv; fi
	cd ui && $(UV) run --no-sync ruff check .
endif
	@echo "Linting ML..."
	cd ml && $(UV) run --no-sync ruff check .
	@echo "Linting Ingest..."
	cd ingest && $(UV) run --no-sync ruff check .

lint-fix: ## Auto-fix Ruff lint errors (API + UI + ML + Ingest)
	@echo "Fixing API..."
	cd api && $(UV) run --no-sync ruff check --fix .
	@echo "Fixing UI..."
	cd ui && $(UV) run --no-sync ruff check --fix .
	@echo "Fixing ML..."
	cd ml && $(UV) run --no-sync ruff check --fix .
	@echo "Fixing Ingest..."
	cd ingest && $(UV) run --no-sync ruff check --fix .

clean: ## Remove Python caches and build artifacts
	@$(PYTHON) scripts/clean.py
	@echo "Clean complete."

clean-venv: ## Remove .venv directories (fixes Windows permission issues)
	@$(PYTHON) scripts/clean.py --include-venv

ralph-init: ## Initialize Ralph loop (.ralph/)
	$(RALPH_CMD) init "$(RALPH_TASK_FILE)"

ralph-plan: ## Generate .ralph/plan.json + .ralph/state.json (optional: RALPH_TASK_FILE=...)
	$(RALPH_CMD) plan "$(RALPH_TASK_FILE)"

ralph-run: ## Run Ralph loop (optional: RALPH_TASK_FILE=...)
	$(RALPH_CMD) run "$(RALPH_TASK_FILE)"

ralph-status: ## Show Ralph loop status
	$(RALPH_CMD) status

db-up: ## Start the database service
	@echo "Starting database service..."
	docker compose up db -d

db-down: ## Stop the database service
	@echo "Stopping database service..."
	docker compose stop db

migrate: ## Run database migrations
	@echo "Running database migrations..."
	cd api && uv run alembic upgrade head

revision: ## Create a new migration revision (usage: make revision msg="description")
	@echo "Creating new migration revision..."
	$(if $(msg),,$(error Please provide a message with msg='your message'))
	cd api && uv run alembic revision -m "$(msg)"

ingest-firms: ## Run NASA FIRMS ingestion (pass ARGS="--day-range 3")
	$(UV) run --project ingest -m ingest.firms_ingest $(ARGS)

ingest-firms-backfill: ## Backfill historical FIRMS detections (pass ARGS="--start YYYY-MM-DD --end YYYY-MM-DD --area w,s,e,n --sources ...")
	$(UV) run --project ingest -m ingest.firms_backfill $(ARGS)

repair-fire-detections: ## Repair synthetic rows, thermal fields, stale running batches, and batch metadata
	$(UV) run --project api scripts/repair_fire_detections.py $(ARGS)

recompute-fire-scores: ## Recompute scoring fields for batches with incomplete derived columns
	$(UV) run --project api scripts/recompute_fire_scores.py $(ARGS)

denoiser-data-coverage-report: ## Export denoiser data coverage/neutral report (pass ARGS="--start ... --end ...")
	$(UV) run --project api scripts/denoiser_data_coverage_report.py $(ARGS)

ingest-weather: ## Run NOAA GFS weather ingestion (pass ARGS="--run-time 2025-12-06T00:00Z")
	$(UV) run --project ingest -m ingest.weather_ingest $(ARGS)

ingest-dem: ## Run Copernicus DEM preprocessing (pass ARGS="--cog")
	$(UV) run --project ingest -m ingest.dem_preprocess $(ARGS)

smoke-grid: ## Run DEM + weather smoke check for grid alignment (pass ARGS="--bbox 5.1 35.4 6.0 36.0")
	$(UV) run --project ingest scripts/smoke_grid_alignment.py $(ARGS)

smoke-terrain-features: ## Run DEM + slope/aspect smoke check (pass ARGS="--bbox ... --region smoke_grid")
	$(UV) run --project ingest scripts/smoke_terrain_features.py $(ARGS)

ingest-forecast: ## Run spread forecast and persist (pass ARGS="--region ... --bbox ...")
	$(UV) run --project ingest -m ingest.spread_forecast $(ARGS)

ingest-industrial: ## Ingest industrial sources (pass ARGS="--wri --bbox ...")
	$(UV) run --project ingest -m ingest.industrial_sources_ingest $(ARGS)

ingest-industrial-authoritative: ## Ingest authoritative industrial profile (pass ARGS="--source-profile ... [--curated-file ...]")
	$(UV) run --project ingest -m ingest.industrial_sources_ingest $(ARGS)

industrial-build-policy: ## Build/update industrial mask policy (pass ARGS="--policy-version global_authoritative_industrial_v1")
	$(UV) run --project ingest -m ingest.industrial_policy_builder $(ARGS)

industrial-load-no-go-zones: ## Load industrial no-go zones (pass ARGS="--config configs/industrial_policy_global_v1.yaml")
	$(UV) run --project ingest -m ingest.industrial_no_go_loader $(ARGS)

industrial-coverage-report: ## Export denoiser data coverage report incl. industrial policy metrics
	$(UV) run --project api scripts/denoiser_data_coverage_report.py $(ARGS)

br-build-hybrid-curated: ## Build BR hybrid curated CSV from CTF identity + IBGE coordinate base
	$(UV) run --project ingest python scripts/build_br_ctf_ibge_hybrid_curated.py $(ARGS)

ingest-viirs: ## Alias for ingest-firms
	$(MAKE) ingest-firms ARGS="$(ARGS)"

ingest-fwi: ## Alias for ingest-forecast
	$(MAKE) ingest-forecast ARGS="$(ARGS)"

ingest-all: ingest-viirs ingest-fwi ingest-weather ## Run all primary ingestion pipelines

db-cleanup: ## Run database cleanup (14-day retention)
	$(UV) run --project api scripts/db_cleanup.py

# Default local bootstrap window for weather/terrain/perimeters.
# FIRMS defaults to `world` so the map has recent detections out of the box.
# Override these on demand:
#   make prepare PREPARE_BBOX="-125 24 -66 50" PREPARE_FIRMS_AREA="-125,24,-66,50" PREPARE_REGION="conus"
PREPARE_BBOX ?= 22 40 24 42
PREPARE_FIRMS_AREA ?= world
PREPARE_REGION ?= smoke_grid
PREPARE_JOBS ?= firms,weather,terrain,perimeters
PREPARE_MAX_RETRIES ?= 3
PREPARE_RETRY_BACKOFF_SECONDS ?= 20
PREPARE_FIRMS_DAY_RANGE ?= 1
PREPARE_PERIMETER_YEARS ?=
PREPARE_WEATHER_PATCH_MODE ?= --weather-patch-mode
PREPARE_EXTRA_ARGS ?=

prepare: ## Prepare DB + core context data (migrate, cleanup, orchestrated FIRMS/weather/terrain/perimeters)
	@echo "=== Step 1/3: Running migrations ==="
	$(MAKE) migrate
	@echo ""
	@echo "=== Step 2/3: Cleaning old operational records ==="
	$(MAKE) db-cleanup
	@echo ""
	@echo "=== Step 3/3: Running orchestrated ingestion ==="
	$(MAKE) ingest-orchestrator ARGS="--once --jobs $(PREPARE_JOBS) --enforce-freshness --max-retries $(PREPARE_MAX_RETRIES) --retry-backoff-seconds $(PREPARE_RETRY_BACKOFF_SECONDS) --firms-day-range $(PREPARE_FIRMS_DAY_RANGE) --firms-area $(PREPARE_FIRMS_AREA) --weather-bbox $(PREPARE_BBOX) $(PREPARE_WEATHER_PATCH_MODE) --terrain-bbox $(PREPARE_BBOX) --terrain-region-name $(PREPARE_REGION) --perimeters-bbox $(PREPARE_BBOX) $(PREPARE_PERIMETER_YEARS) $(PREPARE_EXTRA_ARGS)"
	@echo ""
	@echo "Prepare complete."
	@echo "Check freshness/status: curl http://localhost:8000/health/data-freshness"

OPS_JOBS ?= firms,perimeters
OPS_FIRMS_INTERVAL_MINUTES ?= 30
OPS_WEATHER_INTERVAL_MINUTES ?= 180
OPS_TERRAIN_INTERVAL_MINUTES ?= 1440
OPS_PERIMETERS_INTERVAL_MINUTES ?= 1440
OPS_MAX_RETRIES ?= 3
OPS_RETRY_BACKOFF_SECONDS ?= 20
OPS_DASHBOARD_PATH ?= data/ingest/orchestrator_dashboard.json

ops-start: ## Start continuous runtime scheduler profile (FIRMS 30m, weather/perimeters periodic)
	$(MAKE) ingest-orchestrator ARGS="--loop --jobs $(OPS_JOBS) --poll-seconds 30 --enforce-freshness --max-retries $(OPS_MAX_RETRIES) --retry-backoff-seconds $(OPS_RETRY_BACKOFF_SECONDS) --firms-interval-minutes $(OPS_FIRMS_INTERVAL_MINUTES) --weather-interval-minutes $(OPS_WEATHER_INTERVAL_MINUTES) --terrain-interval-minutes $(OPS_TERRAIN_INTERVAL_MINUTES) --perimeters-interval-minutes $(OPS_PERIMETERS_INTERVAL_MINUTES) --dashboard-path $(OPS_DASHBOARD_PATH)"

denoiser-label: ## Run ground-truth labeling (pass ARGS="--bbox ... --start ... --end ...")
	$(UV) run --project ml -m ml.denoiser.label $(ARGS)

denoiser-snapshot: ## Export training snapshot (pass ARGS="--bbox ... --start ... --end ... --version ...")
	$(UV) run --project ml -m ml.denoiser.export_snapshot $(ARGS)

denoiser-train: ## Train denoiser (pass CONFIG="configs/denoiser_train.yaml")
	$(UV) run --project ml -m ml.train_denoiser --config $(if $(CONFIG),$(CONFIG),configs/denoiser_train.yaml)

denoiser-eval: ## Evaluate denoiser and choose thresholds (pass MODEL_RUN="models/denoiser/<run_id>" SNAPSHOT="data/denoiser/snapshots/<run>" OUT="reports/denoiser/<run_id>" ARGS="--target_precision 0.95 ...")
	$(if $(MODEL_RUN),,$(error Please provide MODEL_RUN="models/denoiser/<run_id>"))
	$(if $(SNAPSHOT),,$(error Please provide SNAPSHOT="data/denoiser/snapshots/<run>" or a labeled parquet))
	$(UV) run --project ml -m ml.eval_denoiser --model_run $(MODEL_RUN) --snapshot $(SNAPSHOT) $(if $(OUT),--out $(OUT),) $(ARGS)

denoiser-eventize: ## Build front/event clusters for v2 (pass ARGS="--batch-id ... | --start ... --end ...")
	$(UV) run --project ml -m ml.denoiser.eventize $(ARGS)

denoiser-label-v2: ## Run v2 labeling (pass ARGS="--start ... --end ... [--bbox ...] --version ... --authority-profile wfigs_us --perimeter-source authoritative_perimeters --authoritative-tier both --industrial-policy-version global_authoritative_industrial_v1")
	$(UV) run --project ml -m ml.denoiser.label_v2 $(ARGS)

denoiser-snapshot-v2: ## Export v2 event snapshot (pass ARGS="--bbox ... --start ... --end ... --version ...")
	$(UV) run --project ml -m ml.denoiser.export_snapshot_v2 $(ARGS)

denoiser-train-v2: ## Train v2 denoiser (pass CONFIG="configs/denoiser_train_v2.yaml")
	$(UV) run --project ml -m ml.train_denoiser_v2 --config $(if $(CONFIG),$(CONFIG),configs/denoiser_train_v2.yaml)

denoiser-eval-v2: ## Evaluate v2 denoiser (pass MODEL_RUN="models/denoiser_v2/<run_id>" SNAPSHOT=".../run_<id>" OUT="reports/denoiser_v2/<run_id>" ARGS="--gate-scope covered")
	$(if $(MODEL_RUN),,$(error Please provide MODEL_RUN="models/denoiser_v2/<run_id>"))
	$(if $(SNAPSHOT),,$(error Please provide SNAPSHOT=<snapshot dir/parquet>))
	$(if $(OUT),,$(error Please provide OUT="reports/denoiser_v2/<run_id>"))
	$(UV) run --project ml -m ml.eval_denoiser_v2 --model_run $(MODEL_RUN) --snapshot $(SNAPSHOT) --out $(OUT) $(ARGS)

denoiser-association-report: ## Compare baseline vs updated event association quality (pass ARGS="--start ... --end ...")
	$(UV) run --project ml -m ml.denoiser.eval_event_association $(ARGS)

denoiser-drift-monitor: ## Run denoiser drift monitor (+ optional rollback) (pass ARGS="--dry-run")
	$(UV) run --project ingest -m ingest.denoiser_drift_monitor $(ARGS)

denoiser-load-coverage-masks: ## Load coverage masks (pass ARGS="--input ... --authority-profile wfigs_us --source-uri ... --source-version ...")
	$(UV) run --project api scripts/load_perimeter_coverage_masks.py $(ARGS)

denoiser-build-coverage-masks: ## Build coverage masks from authoritative geometry (pass ARGS="--input ... --authority-profile wfigs_us --source-uri ... --source-version ...")
	$(UV) run --project ingest -m ingest.coverage_mask_builder $(ARGS)

denoiser-freeze-baseline: ## Freeze baseline artifacts (pass ARGS="--model-run models/denoiser_v2/<run_id> --snapshot ...")
	$(UV) run --project api scripts/denoiser_v2_freeze_baseline.py $(ARGS)

denoiser-sweep-v2: ## Run/dry-run constrained PU-bagging sweep (pass ARGS="--base-config ... --snapshot ... [--execute]")
	$(UV) run --project api scripts/denoiser_v2_sweep.py $(ARGS)

ingest-nifc-perimeters: ## Ingest NIFC fire perimeters (pass ARGS="--year 2024 --year 2025")
	$(UV) run --project ingest -m ingest.nifc_perimeters_ingest $(ARGS)

ingest-authoritative-perimeters: ## Ingest authoritative WFIGS perimeters (pass ARGS="--source-profile ... [--start ... --end ...]")
	$(UV) run --project ingest -m ingest.wfigs_authority_ingest $(ARGS)

ingest-orchestrator: ## Run unified FIRMS/weather/terrain/perimeters orchestrator (pass ARGS="--loop --poll-seconds 30")
	$(UV) run --project ingest -m ingest.orchestrator $(ARGS)

download-fuels: ## Build/cache fuel-moisture feature cube (pass ARGS="--bbox ... --run-time ...")
	$(UV) run --project ingest -m ingest.fuels_ingest $(ARGS)

model-register: ## Register model artifact (usage: make model-register FAMILY=denoiser ARTIFACT=... METRICS=@path/or-json)
	$(if $(FAMILY),,$(error Please provide FAMILY=denoiser|spread))
	$(if $(ARTIFACT),,$(error Please provide ARTIFACT=<artifact path or URI>))
	$(UV) run --project api scripts/model_registry.py register --family "$(FAMILY)" --artifact "$(ARTIFACT)" $(if $(METRICS),--metrics '$(METRICS)',)

model-promote: ## Promote model champion (usage: make model-promote FAMILY=denoiser MODEL_ID=...)
	$(if $(FAMILY),,$(error Please provide FAMILY=denoiser|spread))
	$(if $(MODEL_ID),,$(error Please provide MODEL_ID=<registered model id>))
	$(UV) run --project api scripts/model_registry.py promote --family "$(FAMILY)" --model-id "$(MODEL_ID)" $(if $(PROMOTED_BY),--by "$(PROMOTED_BY)",) $(if $(NOTES),--notes "$(NOTES)",)

model-rollback: ## Rollback champion to previous promoted model (usage: make model-rollback FAMILY=denoiser)
	$(if $(FAMILY),,$(error Please provide FAMILY=denoiser|spread))
	$(UV) run --project api scripts/model_registry.py rollback --family "$(FAMILY)" $(if $(PROMOTED_BY),--by "$(PROMOTED_BY)",) $(if $(NOTES),--notes "$(NOTES)",)

TRAIN_DENOISER_PIPELINE ?= v1
TRAIN_DENOISER_CONFIG_V1 ?= configs/denoiser_train.yaml
TRAIN_DENOISER_CONFIG_V2 ?= configs/denoiser_train_v2.yaml
TRAIN_DENOISER_CONFIG ?= $(if $(filter v2,$(TRAIN_DENOISER_PIPELINE)),$(TRAIN_DENOISER_CONFIG_V2),$(TRAIN_DENOISER_CONFIG_V1))
TRAIN_DENOISER_FAMILY ?= denoiser
TRAIN_DENOISER_ROOT ?= $(if $(filter v2,$(TRAIN_DENOISER_PIPELINE)),models/denoiser_v2,models/denoiser)
TRAIN_DENOISER_METRICS_FILE ?= metrics.json
TRAIN_DENOISER_GATE_REPORT_FILE ?= gate_report.json
TRAIN_DENOISER_REQUIRE_GATE ?= false

train-denoiser: ## Train/eval/register/promote denoiser champion from latest run
	@echo "=== Training denoiser ($(TRAIN_DENOISER_PIPELINE)) ==="
	@if [ ! -f "$(TRAIN_DENOISER_CONFIG)" ]; then echo "Missing config: $(TRAIN_DENOISER_CONFIG)"; exit 1; fi
	@if [ "$(TRAIN_DENOISER_PIPELINE)" = "v2" ]; then \
		$(UV) run --project ml -m ml.train_denoiser_v2 --config $(TRAIN_DENOISER_CONFIG); \
	else \
		$(UV) run --project ml -m ml.train_denoiser --config $(TRAIN_DENOISER_CONFIG); \
	fi
	@latest_run=$$(ls -td $(TRAIN_DENOISER_ROOT)/* 2>/dev/null | head -n 1); \
	if [ -z "$$latest_run" ]; then echo "No denoiser run found under $(TRAIN_DENOISER_ROOT)"; exit 1; fi; \
	metrics_file="$$latest_run/$(TRAIN_DENOISER_METRICS_FILE)"; \
	gate_file="$$latest_run/$(TRAIN_DENOISER_GATE_REPORT_FILE)"; \
	if [ ! -f "$$metrics_file" ]; then echo "Missing metrics file: $$metrics_file"; exit 1; fi; \
	if [ "$(TRAIN_DENOISER_REQUIRE_GATE)" = "true" ] && [ ! -f "$$gate_file" ]; then echo "Missing required gate report: $$gate_file"; exit 1; fi; \
	gate_pass="true"; \
	coverage_fresh="true"; \
	if [ -f "$$gate_file" ]; then \
		gate_pass=$$($(PYTHON) -c 'import json,sys; payload=json.load(open(sys.argv[1], "r", encoding="utf-8")); print("true" if bool(payload.get("pass", False)) else "false")' "$$gate_file"); \
		if [ "$$gate_pass" != "true" ]; then echo "Promotion blocked: gate report failed ($$gate_file)"; exit 1; fi; \
		if [ "$(TRAIN_DENOISER_PIPELINE)" = "v2" ]; then \
			coverage_fresh=$$($(PYTHON) -c 'import json,sys; payload=json.load(open(sys.argv[1], "r", encoding="utf-8")); freshness=payload.get("coverage_data_freshness") or {}; print("true" if bool(freshness.get("fresh", False)) else "false")' "$$gate_file"); \
			if [ "$$coverage_fresh" != "true" ]; then echo "Promotion blocked: authoritative coverage freshness check failed ($$gate_file)"; exit 1; fi; \
		fi; \
	fi; \
	registry_metrics="$$latest_run/registry_metrics.json"; \
	$(PYTHON) -c 'import json, os, sys; metrics_path, gate_path, out_path, gate_pass, coverage_fresh = sys.argv[1:6]; metrics=json.load(open(metrics_path, "r", encoding="utf-8")); out=dict(metrics) if isinstance(metrics, dict) else {"metrics": metrics}; out["gate_pass"]=(gate_pass=="true"); out["coverage_fresh"]=(coverage_fresh=="true"); \
if gate_path and os.path.exists(gate_path): out["gate_report"]=json.load(open(gate_path, "r", encoding="utf-8")); \
json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)' "$$metrics_file" "$$gate_file" "$$registry_metrics" "$$gate_pass" "$$coverage_fresh"; \
	echo "Registering $$latest_run"; \
	model_id=$$($(UV) run --project api scripts/model_registry.py register --id-only --family $(TRAIN_DENOISER_FAMILY) --artifact "$$latest_run" --metrics "@$$registry_metrics"); \
	echo "Promoting $$model_id"; \
	$(UV) run --project api scripts/model_registry.py promote --family $(TRAIN_DENOISER_FAMILY) --model-id "$$model_id" --notes "auto-promote from make train-denoiser pipeline=$(TRAIN_DENOISER_PIPELINE)"

TRAIN_SPREAD_PIPELINE ?= v1
TRAIN_SPREAD_CONFIG_V1 ?= configs/spread_train_v1.yaml
TRAIN_SPREAD_CONFIG_V2 ?= configs/spread_train_v2.yaml
TRAIN_SPREAD_CONFIG ?= $(if $(filter v2,$(TRAIN_SPREAD_PIPELINE)),$(TRAIN_SPREAD_CONFIG_V2),$(TRAIN_SPREAD_CONFIG_V1))
TRAIN_SPREAD_FAMILY ?= spread
TRAIN_SPREAD_ROOT ?= $(if $(filter v2,$(TRAIN_SPREAD_PIPELINE)),models/spread_v2,models/spread_v1)
TRAIN_SPREAD_METRICS_FILE ?= metrics.json
TRAIN_SPREAD_GATE_REPORT_FILE ?= gate_report.json
TRAIN_SPREAD_GATE_CONFIG_V2 ?= configs/spread_champion_challenger.yaml
TRAIN_SPREAD_GATE_CONFIG ?= $(if $(filter v2,$(TRAIN_SPREAD_PIPELINE)),$(TRAIN_SPREAD_GATE_CONFIG_V2),)
TRAIN_SPREAD_REQUIRE_GATE ?= $(if $(filter v2,$(TRAIN_SPREAD_PIPELINE)),true,false)

train-spread: ## Train/eval/register/promote spread champion from latest run
	@echo "=== Training spread model ($(TRAIN_SPREAD_PIPELINE)) ==="
	@if [ ! -f "$(TRAIN_SPREAD_CONFIG)" ]; then echo "Missing config: $(TRAIN_SPREAD_CONFIG)"; exit 1; fi
	@if [ "$(TRAIN_SPREAD_PIPELINE)" = "v2" ]; then \
		$(UV) run --project ml -m ml.train_spread_v2 --config $(TRAIN_SPREAD_CONFIG); \
	else \
		$(UV) run --project ml -m ml.train_spread_v1 --config $(TRAIN_SPREAD_CONFIG); \
	fi
	@latest_run=$$(ls -td $(TRAIN_SPREAD_ROOT)/* 2>/dev/null | head -n 1); \
	if [ -z "$$latest_run" ]; then echo "No spread run found under $(TRAIN_SPREAD_ROOT)"; exit 1; fi; \
	metrics_file="$$latest_run/$(TRAIN_SPREAD_METRICS_FILE)"; \
	gate_file="$$latest_run/$(TRAIN_SPREAD_GATE_REPORT_FILE)"; \
	if [ ! -f "$$metrics_file" ]; then echo "Missing metrics file: $$metrics_file"; exit 1; fi; \
	if [ -n "$(TRAIN_SPREAD_GATE_CONFIG)" ]; then \
		gate_out="$$latest_run/gate_eval"; \
		$(UV) run --project ml -m ml.eval_spread_champion_challenger --config $(TRAIN_SPREAD_GATE_CONFIG) --out-dir "$$gate_out"; \
		latest_gate=$$(ls -td "$$gate_out"/* 2>/dev/null | head -n 1); \
		if [ -n "$$latest_gate" ] && [ -f "$$latest_gate/summary.json" ]; then \
			$(PYTHON) -c 'import json,sys; data=json.load(open(sys.argv[1], "r", encoding="utf-8")); decision=data.get("decision", {}); payload={"pass": bool(decision.get("recommend_challenger", False)), "decision": decision}; json.dump(payload, open(sys.argv[2], "w", encoding="utf-8"), indent=2)' "$$latest_gate/summary.json" "$$gate_file"; \
		fi; \
	fi; \
	if [ "$(TRAIN_SPREAD_REQUIRE_GATE)" = "true" ] && [ ! -f "$$gate_file" ]; then echo "Missing required gate report: $$gate_file"; exit 1; fi; \
	gate_pass="true"; \
	if [ -f "$$gate_file" ]; then \
		gate_pass=$$($(PYTHON) -c 'import json,sys; payload=json.load(open(sys.argv[1], "r", encoding="utf-8")); print("true" if bool(payload.get("pass", False)) else "false")' "$$gate_file"); \
		if [ "$$gate_pass" != "true" ]; then echo "Promotion blocked: gate report failed ($$gate_file)"; exit 1; fi; \
	fi; \
	registry_metrics="$$latest_run/registry_metrics.json"; \
	$(PYTHON) -c 'import json, os, sys; metrics_path, gate_path, out_path, gate_pass = sys.argv[1:5]; metrics=json.load(open(metrics_path, "r", encoding="utf-8")); out=dict(metrics) if isinstance(metrics, dict) else {"metrics": metrics}; out["gate_pass"]=(gate_pass=="true"); \
if gate_path and os.path.exists(gate_path): out["gate_report"]=json.load(open(gate_path, "r", encoding="utf-8")); \
json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)' "$$metrics_file" "$$gate_file" "$$registry_metrics" "$$gate_pass"; \
	echo "Registering $$latest_run"; \
	model_id=$$($(UV) run --project api scripts/model_registry.py register --id-only --family $(TRAIN_SPREAD_FAMILY) --artifact "$$latest_run" --metrics "@$$registry_metrics"); \
	echo "Promoting $$model_id"; \
	$(UV) run --project api scripts/model_registry.py promote --family $(TRAIN_SPREAD_FAMILY) --model-id "$$model_id" --notes "auto-promote from make train-spread pipeline=$(TRAIN_SPREAD_PIPELINE)"

# ── Denoiser end-to-end pipeline ─────────────────────────────────────
# Usage:
#   make denoiser-pipeline BBOX="-180 -90 180 90" START=2026-01-18 END=2026-01-30 YEARS="--year 2024 --year 2025 --year 2026"
#
# BBOX is intentionally global: labeling auto-restricts negatives to the
# perimeter coverage region, so non-US detections stay UNKNOWN (safe).
# This runs: migrate → ingest perimeters → label → snapshot → train.
# NOTE: START/END must match dates in fire_detections. Update when backfilling.
BBOX ?= -180 -90 180 90
START ?= 2026-01-18
END ?= 2026-01-30
YEARS ?= --year 2024 --year 2025 --year 2026
DENOISER_LABEL_VERSION ?= default

denoiser-pipeline: ## End-to-end denoiser: migrate → ingest perimeters → label → snapshot → train
	@echo "=== Step 1/5: Running migrations ==="
	$(MAKE) migrate
	@echo ""
	@echo "=== Step 2/5: Ingesting NIFC fire perimeters ==="
	$(MAKE) ingest-nifc-perimeters ARGS="$(YEARS) --bbox $(BBOX)"
	@echo ""
	@echo "=== Step 3/5: Labeling detections with ground truth ==="
	$(MAKE) denoiser-label ARGS="--bbox $(BBOX) --start $(START) --end $(END)"
	@echo ""
	@echo "=== Step 4/5: Exporting training snapshot ==="
	$(MAKE) denoiser-snapshot ARGS="--bbox $(BBOX) --start $(START) --end $(END) --version $(DENOISER_LABEL_VERSION)"
	@echo ""
	@echo "=== Step 5/5: Training denoiser (auto-detecting latest snapshot) ==="
	$(UV) run --project ml -m ml.train_denoiser \
		--config configs/denoiser_train.yaml \
		--snapshot-path latest

hindcast-build: ## Build spread hindcast predicted/observed dataset (pass CONFIG="configs/hindcast_smoke_grid_balkans_mvp.yaml")
	$(UV) run --project ml -m ml.spread.hindcast_builder --config $(if $(CONFIG),$(CONFIG),configs/hindcast_smoke_grid_balkans_mvp.yaml) $(ARGS)

spread-champion-challenger: ## Evaluate spread champion vs challenger (pass CONFIG="configs/spread_champion_challenger.yaml")
	$(UV) run --project ml -m ml.eval_spread_champion_challenger --config $(if $(CONFIG),$(CONFIG),configs/spread_champion_challenger.yaml) $(ARGS)

weather-bias: ## Run weather bias analysis (pass ARGS="--forecast-nc ... --truth-nc ...")
	$(UV) run --project ml -m ml.weather_bias_analysis $(ARGS)
