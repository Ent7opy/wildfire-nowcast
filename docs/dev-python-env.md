## Python & `uv` Development Environment

This repo standardizes on **Python 3.11.x** (pinned in `.python-version`) and uses `uv` for dependency management across `api/`, `ui/`, `ml/`, and `ingest/`. `uv sync` will download a matching 3.11 interpreter automatically if your system default is newer.

For a complete from-scratch setup (tool installation, `.env`, Docker, `make` workflows), see [`docs/SETUP.md`](./SETUP.md). This document dives deeper into the Python/`uv` flows that `make` ultimately calls.

---

## 1. Prerequisites

1. Install Python 3.11 and confirm with `python --version`. The `.python-version` file shows the exact patch (e.g. `3.11.9`), but any `3.11.x` interpreter should work.
2. Install `uv` (see https://pypi.org/project/uv/ for instructions):
   ```bash
   python3 -m pip install --upgrade uv    # POSIX
   py -3 -m pip install --upgrade uv       # Windows (uses the launcher’s 3.11)
   ```
3. Install Docker Desktop / Engine and confirm `docker compose version` works.
4. Install GNU `make` if your OS does not include it:
   - macOS: `brew install make` (exposes `gmake`; symlink as `make` if desired).
   - Windows: `choco install make` (restart terminal afterwards).
   - Linux: `sudo apt-get install build-essential` (or distro equivalent).
5. Clone the repo and `cd` into `wildfire-nowcast`.

---

## 2. Canonical workflow

Each top-level project manages its own `.venv` under the directory. The easiest way to sync everything (runtime + dev dependencies) is:

```bash
make install
```

Under the hood, each directory runs `uv sync --dev`. If you prefer to manage environments manually, the pattern is identical for `api`, `ui`, `ml`, and `ingest`:

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

1. Sync the dependencies (or rely on `make install`):
   ```bash
   cd api
   uv sync
   ```
2. Start the FastAPI hello world (defined in `api/main.py`):
   ```bash
   uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```
3. Visit `http://localhost:8000/health` to see `{"status": "ok"}`.

**Shortcut:** `make dev-api` will execute the same command (after `make install`).

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

**Shortcut:** `make dev-ui` handles the sync/run combo once environments exist.

### ML & ingest scripts

- **ML (`ml/`)**
  1. `cd ml`
  2. `uv sync`
  3. `uv run python <script>.py`

- **FIRMS ingestion (`ingest/`)**
  1. `cd ingest`
  2. `uv sync`
  3. `uv run -m ingest.firms_ingest --day-range 1 --area world`

  The ingestion project has its own `pyproject.toml`, so dependencies (HTTP client, SQLAlchemy, etc.) stay isolated from the API/UI. Shortcut: `make ingest-firms ARGS="--day-range 3"` runs the same module with optional arguments.

---

## 4. Testing & CI

GitHub Actions runs `.github/workflows/ci.yml` on every push or pull request to `master`. The workflow
uses a simple matrix to run the same checks for `api/` and `ui/`:

1. Install Python 3.11 and `uv`.
2. `uv sync --dev` to create/update the per-project virtual environment (pulls runtime + dev deps).
3. `uv run ruff check .` for linting.
4. `uv run pytest` to execute the suite (`api/tests/test_health.py` exercises the FastAPI `/health`
   endpoint, while `ui/tests/test_app_imports.py` ensures the Streamlit app imports cleanly).

Running the same checks locally keeps CI green. You can either call the commands below directly or rely on `make test` / `make lint` which wrap the same invocations.

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

---

## 5. Makefile helpers

All common workflows are exposed through the root `Makefile`. Use `make help` to list them. Key targets:

- `make install` – sync dependencies for every subproject.
- `make db-up` / `make db-down` – manage the Postgres/PostGIS container.
- `make migrate` – run Alembic migrations from the `api` project.
- `make dev-api`, `make dev-ui` – start dev servers.
- `make test`, `make lint` – run pytest / Ruff for API + UI.
- `make ingest-firms ARGS="..."` – run the FIRMS ingestion CLI.
- `make clean` – remove Python caches and build artifacts via `scripts/clean.py`.

These targets work cross-platform as long as GNU `make`, `uv`, Python 3.11, and Docker are installed (see `docs/SETUP.md` for installation details).

