## Python & `uv` Development Environment

This repo standardizes on **Python 3.11.x** (pinned in `.python-version`) and uses `uv` for dependency management across `api/`, `ui/`, and `ml/`. `uv sync` will download a matching 3.11 interpreter automatically if your system default is newer.

---

## 1. Prerequisites

1. Install Python 3.11 and confirm with `python --version`. The `.python-version` file shows the exact patch (e.g. `3.11.9`), but any `3.11.x` interpreter should work.
2. Install `uv` (see https://pypi.org/project/uv/ for instructions):
   ```bash
   python3 -m pip install --upgrade uv    # POSIX
   py -3 -m pip install --upgrade uv       # Windows (uses the launcher’s 3.11)
   ```
3. Clone the repo and `cd` into `wildfire-nowcast`.

---

## 2. Canonical workflow

Each top-level project manages its own `.venv` under the directory. The pattern is identical for `api`, `ui`, and `ml`:

```bash
cd <project>            # e.g. cd api
uv sync                # creates/updates .venv and installs deps
# run commands via uv to avoid manual activation
uv run <command>       # e.g. uv run uvicorn api.main:app
```

If you prefer activating manually, use:

```bash
source .venv/bin/activate         # POSIX
```

```powershell
.venv\Scripts\activate             # Windows PowerShell
```

Do not commit `.venv/` — this directory is local only.

---

## 3. Hello-world flows

### API

1. Sync the dependencies:
   ```bash
   cd api
   uv sync
   ```
2. Start the FastAPI hello world (defined in `api/main.py`):
   ```bash
   uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```
3. Visit `http://localhost:8000/health` to see `{"status": "ok"}`.

### Verifying the API surface

Once the server is running, a quick smoke test is to hit the two builtin status endpoints:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/version
```

The `/version` endpoint returns the configured app version, git commit (when available), and environment. These endpoints exist in both the host workflow and the Docker Compose stack, so the same requests validate each runtime.

### Adding new API routes

New FastAPI routes belong under the `api/routes/` package. Each module should expose an `APIRouter` and be wired into `api/main.py` with `app.include_router(...)`. Domain-specific routers (e.g., `api/routes/fires.py`) can later register their own prefixes and tags while the `api/routes/internal.py` router keeps internal/operational endpoints centralized.

### UI

1. Sync dependencies:
   ```bash
   cd ui
   uv sync
   ```
2. Launch Streamlit:
   ```bash
   uv run streamlit run app.py
   ```
3. Open the URL printed by Streamlit (typically `http://localhost:8501/`).

### ML & ingest scripts

1. The same workflow applies under `ml`:
   ```bash
   cd ml
   uv sync
   ```
2. Use `uv run python <script>.py` or a REPL to iterate on models, data prep, or ingestion logic.

> **Note:** The `ingest/` code currently shares the `ml` environment. When ingest-specific dependencies become formalized, we will either extend the `ml` deps or give `ingest/` its own `pyproject.toml` and lockfile. The canonical `uv` workflow above still applies once that happens.

---

## 4. Testing & CI

GitHub Actions runs `.github/workflows/ci.yml` on every push or pull request to `master`. The workflow
uses a simple matrix to run the same checks for `api/` and `ui/`:

1. Install Python 3.11 and `uv`.
2. `uv sync --dev` to create/update the per-project virtual environment (pulls runtime + dev deps).
3. `uv run ruff check .` for linting.
4. `uv run pytest` to execute the suite (`api/tests/test_health.py` exercises the FastAPI `/health`
   endpoint, while `ui/tests/test_app_imports.py` ensures the Streamlit app imports cleanly).

Running the same checks locally keeps CI green:

```bash
# API
cd api
uv sync --dev
uv run ruff check .
uv run pytest

# UI
cd ../ui
uv sync --dev
uv run ruff check .
uv run pytest
```

Re-run these commands after touching either codebase (or before pushing) to match the CI behaviour.

