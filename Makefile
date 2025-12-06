.PHONY: help dev-api dev-ui install test lint clean db-up db-down migrate revision ingest-firms ingest-weather ingest-dem

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
	cd api && $(UV) run uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-ui: ## Start Streamlit development server (requires make install)
	cd ui && $(UV) run streamlit run app.py

test: ## Run unit tests (API + UI)
	@echo "Running API tests..."
	cd api && $(UV) run pytest
	@echo "Running UI tests..."
	cd ui && $(UV) run pytest

lint: ## Run Ruff lint checks (API + UI)
	@echo "Linting API..."
	cd api && $(UV) run ruff check .
	@echo "Linting UI..."
	cd ui && $(UV) run ruff check .

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

ingest-weather: ## Run NOAA GFS weather ingestion (pass ARGS="--run-time 2025-12-06T00:00Z")
	$(UV) run --project ingest -m ingest.weather_ingest $(ARGS)

ingest-dem: ## Run Copernicus DEM preprocessing (pass ARGS="--cog")
	$(UV) run --project ingest -m ingest.dem_preprocess $(ARGS)

smoke-grid: ## Run DEM + weather smoke check for grid alignment (pass ARGS="--bbox 5.1 35.4 6.0 36.0")
	$(UV) run --project ingest scripts/smoke_grid_alignment.py $(ARGS)

