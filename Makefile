.PHONY: help dev-api dev-ui install test lint clean db-up db-down migrate revision

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

dev-api: ## Start FastAPI development server
	@echo "Starting FastAPI development server..."
	@echo "TODO: Implement when api/ is set up"
	# cd api && uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-ui: ## Start Streamlit development server
	@echo "Starting Streamlit development server..."
	@echo "TODO: Implement when ui/ is set up"
	# cd ui && streamlit run app.py

install: ## Install dependencies
	@echo "Installing dependencies..."
	@echo "TODO: Implement dependency installation"
	# pip install -r requirements.txt

test: ## Run tests
	@echo "Running tests..."
	@echo "TODO: Implement when tests are added"
	# pytest

lint: ## Run linters
	@echo "Running linters..."
	@echo "TODO: Implement when linting is configured"
	# ruff check .
	# mypy .

clean: ## Clean temporary files
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
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

