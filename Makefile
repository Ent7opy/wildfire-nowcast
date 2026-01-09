.PHONY: help dev-api dev-ui install test lint clean db-up db-down migrate revision ingest-firms ingest-firms-backfill ingest-weather ingest-dem ingest-industrial smoke-grid smoke-terrain-features denoiser-label denoiser-snapshot denoiser-train denoiser-eval hindcast-build weather-bias

PYTHON ?= python
UV ?= uv

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

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
	cd ui && $(UV) run pytest
	@echo "Running ML tests..."
	cd ml && $(UV) run pytest
	@echo "Running Ingest tests..."
	cd ingest && $(UV) run pytest

lint: ## Run Ruff lint checks (API + UI + ML + Ingest)
	@echo "Linting API..."
	cd api && $(UV) run ruff check .
	@echo "Linting UI..."
	cd ui && $(UV) run ruff check .
	@echo "Linting ML..."
	cd ml && $(UV) run ruff check .
	@echo "Linting Ingest..."
	cd ingest && $(UV) run ruff check .

clean: ## Remove Python caches and build artifacts
	@$(PYTHON) scripts/clean.py
	@echo "Clean complete."

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

hindcast-build: ## Build spread hindcast predicted/observed dataset (pass CONFIG="configs/hindcast_smoke_grid_balkans_mvp.yaml")
	$(UV) run --project ml -m ml.spread.hindcast_builder --config $(if $(CONFIG),$(CONFIG),configs/hindcast_smoke_grid_balkans_mvp.yaml) $(ARGS)

weather-bias: ## Run weather bias analysis (pass ARGS="--forecast-nc ... --truth-nc ...")
	$(UV) run --project ml -m ml.weather_bias_analysis $(ARGS)

