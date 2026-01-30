.PHONY: help doctor dev-api dev-ui install test lint lint-fix clean db-up db-down migrate revision db-cleanup ingest-firms ingest-firms-backfill ingest-weather ingest-dem ingest-industrial ingest-viirs ingest-fwi ingest-all prepare smoke-grid smoke-terrain-features denoiser-label denoiser-snapshot denoiser-train denoiser-eval denoiser-label-v2 denoiser-train-v2 denoiser-pipeline-v2 ingest-nifc-perimeters hindcast-build weather-bias ralph-init ralph-plan ralph-run ralph-status health-check

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

ingest-viirs: ## Alias for ingest-firms
	$(MAKE) ingest-firms ARGS="$(ARGS)"

ingest-fwi: ## Alias for ingest-forecast
	$(MAKE) ingest-forecast ARGS="$(ARGS)"

ingest-all: ingest-viirs ingest-fwi ingest-weather ## Run all primary ingestion pipelines

db-cleanup: ## Run database cleanup (14-day retention)
	$(UV) run --project api scripts/db_cleanup.py

prepare: ## Prepare the database and initial context data (FIRMS + Weather)
	@echo "Cleaning up database..."
	$(MAKE) db-cleanup
	@echo "Ingesting FIRMS data..."
	$(MAKE) ingest-firms
	@echo "Ingesting weather data..."
	$(MAKE) ingest-weather

denoiser-label: ## Run heuristic labeling (pass ARGS="--bbox ... --start ... --end ...")
	$(UV) run --project ml -m ml.denoiser.label_v1 $(ARGS)

denoiser-snapshot: ## Export training snapshot (pass ARGS="--bbox ... --start ... --end ... --version ...")
	$(UV) run --project ml -m ml.denoiser.export_snapshot $(ARGS)

denoiser-train: ## Train denoiser baseline (pass CONFIG="configs/denoiser_train_v1.yaml")
	$(UV) run --project ml -m ml.train_denoiser --config $(if $(CONFIG),$(CONFIG),configs/denoiser_train_v1.yaml)

denoiser-eval: ## Evaluate denoiser and choose thresholds (pass MODEL_RUN="models/denoiser_v1/<run_id>" SNAPSHOT="data/denoiser/snapshots/<run>" OUT="reports/denoiser_v1/<run_id>" ARGS="--target_precision 0.95 ...")
	$(if $(MODEL_RUN),,$(error Please provide MODEL_RUN="models/denoiser_v1/<run_id>"))
	$(if $(SNAPSHOT),,$(error Please provide SNAPSHOT="data/denoiser/snapshots/<run>" or a labeled parquet))
	$(UV) run --project ml -m ml.eval_denoiser --model_run $(MODEL_RUN) --snapshot $(SNAPSHOT) $(if $(OUT),--out $(OUT),) $(ARGS)

ingest-nifc-perimeters: ## Ingest NIFC fire perimeters (pass ARGS="--year 2024 --year 2025")
	$(UV) run --project ingest -m ingest.nifc_perimeters_ingest $(ARGS)

denoiser-label-v2: ## Run ground-truth labeling v2 (pass ARGS="--bbox ... --start ... --end ...")
	$(UV) run --project ml -m ml.denoiser.label_v2 $(ARGS)

denoiser-train-v2: ## Train denoiser v2 (pass CONFIG="configs/denoiser_train_v2.yaml")
	$(UV) run --project ml -m ml.train_denoiser --config $(if $(CONFIG),$(CONFIG),configs/denoiser_train_v2.yaml)

# ── Denoiser v2 end-to-end pipeline ─────────────────────────────────────
# Usage:
#   make denoiser-pipeline-v2 BBOX="-125 24 -66 50" START=2024-01-01 END=2025-01-01 YEARS="--year 2024 --year 2025"
#
# This runs: migrate → ingest perimeters → label → snapshot → train.
BBOX ?= -125 24 -66 50
START ?= 2024-01-01
END ?= 2025-01-01
YEARS ?= --year 2024 --year 2025
DENOISER_V2_VERSION ?= v2.0.0

denoiser-pipeline-v2: ## End-to-end denoiser v2: migrate → ingest perimeters → label → snapshot → train
	@echo "=== Step 1/5: Running migrations ==="
	$(MAKE) migrate
	@echo ""
	@echo "=== Step 2/5: Ingesting NIFC fire perimeters ==="
	$(MAKE) ingest-nifc-perimeters ARGS="$(YEARS) --bbox $(BBOX)"
	@echo ""
	@echo "=== Step 3/5: Labeling detections with ground truth (v2) ==="
	$(MAKE) denoiser-label-v2 ARGS="--bbox $(BBOX) --start $(START) --end $(END)"
	@echo ""
	@echo "=== Step 4/5: Exporting training snapshot ==="
	$(MAKE) denoiser-snapshot ARGS="--bbox $(BBOX) --start $(START) --end $(END) --version $(DENOISER_V2_VERSION)"
	@echo ""
	@echo "=== Step 5/5: Training denoiser v2 (auto-detecting latest snapshot) ==="
	$(UV) run --project ml -m ml.train_denoiser \
		--config configs/denoiser_train_v2.yaml \
		--snapshot-path "$$($(PYTHON) scripts/latest_snapshot.py)"

hindcast-build: ## Build spread hindcast predicted/observed dataset (pass CONFIG="configs/hindcast_smoke_grid_balkans_mvp.yaml")
	$(UV) run --project ml -m ml.spread.hindcast_builder --config $(if $(CONFIG),$(CONFIG),configs/hindcast_smoke_grid_balkans_mvp.yaml) $(ARGS)

weather-bias: ## Run weather bias analysis (pass ARGS="--forecast-nc ... --truth-nc ...")
	$(UV) run --project ml -m ml.weather_bias_analysis $(ARGS)
