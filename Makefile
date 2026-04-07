.PHONY: help doctor health-check install test lint lint-fix clean clean-venv migrate revision widget-build ralph-init ralph-plan ralph-run ralph-status denoiser-label denoiser-snapshot denoiser-train denoiser-eval denoiser-eventize denoiser-label-v2 denoiser-snapshot-v2 denoiser-train-v2 denoiser-eval-v2 train-denoiser train-spread ignition-snapshot ignition-train train-ignition model-register model-promote model-rollback model-update-contract hindcast-build spread-champion-challenger weather-bias seed-ne-places ingest-orchestrator ops-start backup restore backup-list railway-up railway-down railway-down-all

PYTHON ?= python3
UV ?= uv

# Avoid cross-OS venv collisions (e.g., WSL-created venvs on Windows).
ifeq ($(OS),Windows_NT)
    UV_PROJECT_ENVIRONMENT ?= .venv-win
else
    UV_PROJECT_ENVIRONMENT ?= .venv
endif
export UV_PROJECT_ENVIRONMENT

# Ralph detection
ifeq ($(OS),Windows_NT)
    RALPH_CMD = @C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -ExecutionPolicy Bypass -File .ralph/ralph.ps1
else
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
	Write-Host 'Checking Node.js...'; \
	try { node --version | ForEach-Object { Write-Host \"  [OK] Node $_\" } } catch { Write-Host '  [FAIL] Node.js not found' -ForegroundColor Red }; \
	try { npm --version | ForEach-Object { Write-Host \"  [OK] npm $_\" } } catch { Write-Host '  [FAIL] npm not found' -ForegroundColor Red }; \
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
	@echo "Checking Node.js..."
	@node --version 2>/dev/null && echo "  [OK] Node installed" || echo "  [FAIL] Node.js not found"
	@npm --version 2>/dev/null && echo "  [OK] npm installed" || echo "  [FAIL] npm not found"
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

# ── Dependencies ───────────────────────────────────────────────────────────────

install: ## Install dependencies for all subprojects (with dev extras)
	cd api && $(UV) sync --dev
	cd ui && npm install
	cd ml && $(UV) sync --dev
	cd ingest && $(UV) sync --dev

# ── Quality ────────────────────────────────────────────────────────────────────

test: ## Run unit tests (API + UI + ML + Ingest)
	@echo "Running API tests..."
	cd api && $(UV) run pytest -m "not integration"
	@echo "Running UI tests..."
	cd ui && npm run test:run
	@echo "Running ML tests..."
	cd ml && $(UV) run pytest -m "not integration"
	@echo "Running Ingest tests..."
	cd ingest && $(UV) run pytest -m "not integration"

lint: ## Run lint and type checks (API + UI + ML + Ingest)
	@echo "Linting API..."
	cd api && $(UV) run --no-sync ruff check .
	@echo "Linting UI..."
	cd ui && npm run lint && npm run typecheck
	@echo "Linting ML..."
	cd ml && $(UV) run --no-sync ruff check .
	@echo "Linting Ingest..."
	cd ingest && $(UV) run --no-sync ruff check .

lint-fix: ## Auto-fix lint errors (API + UI + ML + Ingest)
	@echo "Fixing API..."
	cd api && $(UV) run --no-sync ruff check --fix .
	@echo "Fixing UI..."
	cd ui && npm run lint -- --fix
	@echo "Fixing ML..."
	cd ml && $(UV) run --no-sync ruff check --fix .
	@echo "Fixing Ingest..."
	cd ingest && $(UV) run --no-sync ruff check --fix .

widget-build: ## Build standalone embeddable forecast widget (ui/dist-widget/widget.js)
	cd ui && npm run build:widget

clean: ## Remove Python caches and build artifacts
	@$(PYTHON) scripts/clean.py
	@echo "Clean complete."

clean-venv: ## Remove .venv directories (fixes Windows permission issues)
	@$(PYTHON) scripts/clean.py --include-venv

# ── Database ───────────────────────────────────────────────────────────────────

migrate: ## Run database migrations
	@echo "Running database migrations..."
	cd api && $(UV) run alembic upgrade head

revision: ## Create a new migration revision (usage: make revision msg="description")
	@echo "Creating new migration revision..."
	$(if $(msg),,$(error Please provide a message with msg='your message'))
	cd api && $(UV) run alembic revision -m "$(msg)"

backup: ## Create a compressed database backup (stored in data/backups/)
	@echo "Creating database backup..."
	@scripts/backup_db.sh

restore: ## Restore database from a backup file (usage: make restore BACKUP=path/to/backup.sql.gz)
	@echo "Restoring database from $(BACKUP)..."
	@if [ -z "$(BACKUP)" ]; then echo "Error: BACKUP variable not set"; exit 1; fi
	@scripts/restore_db.sh "$(BACKUP)"

backup-list: ## List available database backups
	@echo "Available backups in data/backups/:"
	@ls -1 data/backups/*.sql.gz 2>/dev/null | head -20 || echo "No backups found."

# ── Ralph ──────────────────────────────────────────────────────────────────────

ralph-init: ## Initialize Ralph loop (.ralph/)
	$(RALPH_CMD) init "$(RALPH_TASK_FILE)"

ralph-plan: ## Generate .ralph/plan.json + .ralph/state.json (optional: RALPH_TASK_FILE=...)
	$(RALPH_CMD) plan "$(RALPH_TASK_FILE)"

ralph-run: ## Run Ralph loop (optional: RALPH_TASK_FILE=...)
	$(RALPH_CMD) run "$(RALPH_TASK_FILE)"

ralph-status: ## Show Ralph loop status
	$(RALPH_CMD) status

# ── ML research ────────────────────────────────────────────────────────────────

denoiser-label: ## Run ground-truth labeling (pass ARGS="--bbox ... --start ... --end ...")
	$(UV) run --project ml -m ml.denoiser.label $(ARGS)

denoiser-snapshot: ## Export training snapshot (pass ARGS="--bbox ... --start ... --end ... --version ...")
	$(UV) run --project ml -m ml.denoiser.export_snapshot $(ARGS)

denoiser-train: ## Train denoiser (pass CONFIG="configs/denoiser_train.yaml")
	$(UV) run --project ml -m ml.train_denoiser --config $(if $(CONFIG),$(CONFIG),configs/denoiser_train.yaml)

denoiser-eval: ## Evaluate denoiser and choose thresholds (pass MODEL_RUN="models/denoiser/<run_id>" SNAPSHOT="data/denoiser/snapshots/<run>" OUT="reports/denoiser/<run_id>")
	$(if $(MODEL_RUN),,$(error Please provide MODEL_RUN="models/denoiser/<run_id>"))
	$(if $(SNAPSHOT),,$(error Please provide SNAPSHOT="data/denoiser/snapshots/<run>" or a labeled parquet))
	$(UV) run --project ml -m ml.eval_denoiser --model_run $(MODEL_RUN) --snapshot $(SNAPSHOT) $(if $(OUT),--out $(OUT),) $(ARGS)

denoiser-eventize: ## Build fire/event clusters for v2 (pass ARGS="--batch-id ... | --start ... --end ...")
	$(UV) run --project ml -m ml.denoiser.eventize $(ARGS)

denoiser-label-v2: ## Run v2 labeling (pass ARGS="--start ... --end ... [--bbox ...]")
	$(UV) run --project ml -m ml.denoiser.label_v2 $(ARGS)

denoiser-snapshot-v2: ## Export v2 event snapshot (pass ARGS="--bbox ... --start ... --end ... --version ...")
	$(UV) run --project ml -m ml.denoiser.export_snapshot_v2 $(ARGS)

denoiser-train-v2: ## Train v2 denoiser (pass CONFIG="configs/denoiser_train_v2.yaml")
	$(UV) run --project ml -m ml.train_denoiser_v2 --config $(if $(CONFIG),$(CONFIG),configs/denoiser_train_v2.yaml)

denoiser-eval-v2: ## Evaluate v2 denoiser (pass MODEL_RUN="models/denoiser_v2/<run_id>" SNAPSHOT=".../run_<id>" OUT="reports/denoiser_v2/<run_id>")
	$(if $(MODEL_RUN),,$(error Please provide MODEL_RUN="models/denoiser_v2/<run_id>"))
	$(if $(SNAPSHOT),,$(error Please provide SNAPSHOT=<snapshot dir/parquet>))
	$(if $(OUT),,$(error Please provide OUT="reports/denoiser_v2/<run_id>"))
	$(UV) run --project ml -m ml.eval_denoiser_v2 --model_run $(MODEL_RUN) --snapshot $(SNAPSHOT) --out $(OUT) $(ARGS)

hindcast-build: ## Build spread hindcast predicted/observed dataset (pass CONFIG="configs/hindcast_smoke_grid_balkans_mvp.yaml")
	$(UV) run --project ml -m ml.spread.hindcast_builder --config $(if $(CONFIG),$(CONFIG),configs/hindcast_smoke_grid_balkans_mvp.yaml) $(ARGS)

spread-champion-challenger: ## Evaluate spread champion vs challenger (pass CONFIG="configs/spread_champion_challenger.yaml")
	$(UV) run --project ml -m ml.eval_spread_champion_challenger --config $(if $(CONFIG),$(CONFIG),configs/spread_champion_challenger.yaml) $(ARGS)

weather-bias: ## Run weather bias analysis (pass ARGS="--forecast-nc ... --truth-nc ...")
	$(UV) run --project ml -m ml.weather_bias_analysis $(ARGS)

ignition-snapshot: ## Export ignition training snapshot (pass ARGS="--bbox ... --start ... --end ... --version ...")
	$(UV) run --project ml -m ml.ignition.snapshot $(ARGS)

ignition-train: ## Train ignition probability model (pass CONFIG="configs/ignition_train.yaml")
	$(UV) run --project ml -m ml.train_ignition --config $(if $(CONFIG),$(CONFIG),configs/ignition_train.yaml)

TRAIN_IGNITION_CONFIG ?= configs/ignition_train.yaml
TRAIN_IGNITION_FAMILY ?= ignition
TRAIN_IGNITION_ROOT ?= models/ignition
TRAIN_IGNITION_METRICS_FILE ?= metrics.json
TRAIN_IGNITION_GATE_REPORT_FILE ?= gate_report.json

train-ignition: ## Train/register/promote ignition champion from latest run
	@echo "=== Training ignition probability model ==="
	@if [ ! -f "$(TRAIN_IGNITION_CONFIG)" ]; then echo "Missing config: $(TRAIN_IGNITION_CONFIG)"; exit 1; fi
	$(UV) run --project ml -m ml.train_ignition --config $(TRAIN_IGNITION_CONFIG)
	@latest_run=$$(ls -td $(TRAIN_IGNITION_ROOT)/* 2>/dev/null | head -n 1); \
	if [ -z "$$latest_run" ]; then echo "No ignition run found under $(TRAIN_IGNITION_ROOT)"; exit 1; fi; \
	metrics_file="$$latest_run/$(TRAIN_IGNITION_METRICS_FILE)"; \
	gate_file="$$latest_run/$(TRAIN_IGNITION_GATE_REPORT_FILE)"; \
	if [ ! -f "$$metrics_file" ]; then echo "Missing metrics file: $$metrics_file"; exit 1; fi; \
	if [ ! -f "$$gate_file" ]; then echo "Missing gate report: $$gate_file"; exit 1; fi; \
	gate_pass=$$($(PYTHON) -c 'import json,sys; payload=json.load(open(sys.argv[1], "r", encoding="utf-8")); print("true" if bool(payload.get("pass", False)) else "false")' "$$gate_file"); \
	if [ "$$gate_pass" != "true" ]; then echo "Promotion blocked: gate report failed ($$gate_file)"; exit 1; fi; \
	registry_metrics="$$latest_run/registry_metrics.json"; \
	$(PYTHON) -c 'import json, os, sys; metrics_path, gate_path, out_path = sys.argv[1:4]; metrics=json.load(open(metrics_path, "r", encoding="utf-8")); out=dict(metrics) if isinstance(metrics, dict) else {"metrics": metrics}; out["gate_report"]=json.load(open(gate_path, "r", encoding="utf-8")); json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)' "$$metrics_file" "$$gate_file" "$$registry_metrics"; \
	echo "Registering $$latest_run"; \
	model_id=$$($(UV) run --project api scripts/model_registry.py register --id-only --family $(TRAIN_IGNITION_FAMILY) --artifact "$$latest_run/model.onnx" --metrics "@$$registry_metrics" --runtime-contract "@$$latest_run/contract.json"); \
	echo "Promoting $$model_id"; \
	$(UV) run --project api scripts/model_registry.py promote --family $(TRAIN_IGNITION_FAMILY) --model-id "$$model_id" --notes "auto-promote from make train-ignition"

# ── Model registry ─────────────────────────────────────────────────────────────

model-register: ## Register model artifact (usage: make model-register FAMILY=denoiser ARTIFACT=... METRICS=@path/or-json RUNTIME_CONTRACT=@path/or-json)
	$(if $(FAMILY),,$(error Please provide FAMILY=denoiser|spread))
	$(if $(ARTIFACT),,$(error Please provide ARTIFACT=<artifact path or URI>))
	$(UV) run --project api scripts/model_registry.py register --family "$(FAMILY)" --artifact "$(ARTIFACT)" $(if $(METRICS),--metrics '$(METRICS)',) $(if $(RUNTIME_CONTRACT),--runtime-contract '$(RUNTIME_CONTRACT)',)

model-promote: ## Promote model champion (usage: make model-promote FAMILY=denoiser MODEL_ID=...)
	$(if $(FAMILY),,$(error Please provide FAMILY=denoiser|spread))
	$(if $(MODEL_ID),,$(error Please provide MODEL_ID=<registered model id>))
	$(UV) run --project api scripts/model_registry.py promote --family "$(FAMILY)" --model-id "$(MODEL_ID)" $(if $(PROMOTED_BY),--by "$(PROMOTED_BY)",) $(if $(NOTES),--notes "$(NOTES)",)

model-rollback: ## Rollback champion to previous promoted model (usage: make model-rollback FAMILY=denoiser)
	$(if $(FAMILY),,$(error Please provide FAMILY=denoiser|spread))
	$(UV) run --project api scripts/model_registry.py rollback --family "$(FAMILY)" $(if $(PROMOTED_BY),--by "$(PROMOTED_BY)",) $(if $(NOTES),--notes "$(NOTES)",)

model-update-contract: ## Update runtime contract on a registered model (usage: make model-update-contract FAMILY=denoiser MODEL_ID=... RUNTIME_CONTRACT=@path/or-json)
	$(if $(FAMILY),,$(error Please provide FAMILY=denoiser|spread))
	$(if $(MODEL_ID),,$(error Please provide MODEL_ID=<registered model id>))
	$(if $(RUNTIME_CONTRACT),,$(error Please provide RUNTIME_CONTRACT=@path/or-json))
	$(UV) run --project api scripts/model_registry.py update-contract --family "$(FAMILY)" --model-id "$(MODEL_ID)" --runtime-contract '$(RUNTIME_CONTRACT)' $(if $(REPLACE),--replace,)

seed-ne-places: ## Load Natural Earth populated places (usage: make seed-ne-places NE_PLACES=/path/to/ne_10m_populated_places.geojson)
	$(if $(NE_PLACES),,$(error Please provide NE_PLACES=/path/to/ne_10m_populated_places.geojson or .shp))
	$(UV) run --project api scripts/seed_ne_populated_places.py "$(NE_PLACES)" $(if $(TRUNCATE),--truncate,)

# ── Full train pipelines ───────────────────────────────────────────────────────

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

# ── Ingest Operations ──────────────────────────────────────────────────────────

ingest-orchestrator: ## One-shot ingest (FIRMS + weather + perimeters)
	$(UV) run --project ingest -m ingest.orchestrator \
	  --jobs firms,weather,perimeters \
	  --weather-include-precip \
	  $(if $(ARGS),$(ARGS),)

ops-start: ## Start continuous ingest scheduler
	$(UV) run --project ingest -m ingest.orchestrator \
	  --loop \
	  --jobs firms,weather,perimeters,lfmc,lulc,cleanup \
	  --enforce-freshness \
	  --max-retries 3 \
	  --retry-backoff-seconds 20 \
	  --firms-interval-minutes 30 \
	  --weather-interval-minutes 60 \
	  --weather-include-precip \
	  --perimeters-interval-minutes 1440 \
	  --lfmc-interval-minutes 360 \
	  --lulc-interval-minutes 10080 \
	  --cleanup-interval-minutes 1440 \
	  $(if $(ARGS),$(ARGS),)

# ── Railway ───────────────────────────────────────────────────────────────────

railway-up: ## Start Railway services (scale replicas to 1)
	@scripts/railway_scale.sh up

railway-down: ## Stop Railway services (scale replicas to 0, keeps databases running)
	@scripts/railway_scale.sh down

railway-down-all: ## Stop ALL Railway services including databases (scale replicas to 0)
	@scripts/railway_scale.sh down all
