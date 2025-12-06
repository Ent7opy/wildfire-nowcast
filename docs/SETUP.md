## Project Setup Guide

This guide walks through everything needed to run the **Wildfire Nowcast & Forecast** stack from scratch on Windows, macOS, or Linux. It covers prerequisite tooling, environment setup, `make`-based workflows, and basic validation checks.

---

## 1. Prerequisites

| Tool | Windows | macOS | Linux |
| --- | --- | --- | --- |
| **Git** | Install from [git-scm.com](https://git-scm.com/download/win) | `brew install git` | `sudo apt-get install git` (or distro equivalent) |
| **Python 3.11** | Install from [python.org](https://www.python.org/downloads/windows/) and enable “Add to PATH” | `brew install python@3.11` | `sudo apt-get install python3.11 python3.11-venv` |
| **uv** (Python env manager) | `py -3 -m pip install --upgrade uv` | `python3 -m pip install --upgrade uv` | `python3 -m pip install --upgrade uv` |
| **Docker Desktop / Engine** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) | [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Docker Engine (`sudo apt-get install docker.io`) |
| **GNU make** | `choco install make` *(Chocolatey)*<br/>or use WSL | `brew install make` (installs as `gmake`, symlink if desired) | `sudo apt-get install build-essential` |

> **Tip:** After installing `make` on Windows with Chocolatey you might need to restart your terminal so `make` is on `PATH`.

---

## 2. Repository bootstrap

```bash
git clone https://github.com/<your-org>/wildfire-nowcast.git
cd wildfire-nowcast
```

Create an `.env` file in the repo root with the required secrets (NASA FIRMS MAP key, database overrides, etc.):

```env
FIRMS_MAP_KEY=your_firms_api_key
POSTGRES_USER=wildfire
POSTGRES_PASSWORD=wildfire
POSTGRES_DB=wildfire
```

> `POSTGRES_*` variables default to the values shown above; override only if needed.

---

## 3. Install project dependencies

Run once after cloning (or anytime dependencies change):

```bash
make install
```

This executes `uv sync --dev` inside `api/`, `ui/`, `ml/`, and `ingest/`, installing runtime + dev dependencies into per-project virtual environments.

---

## 4. Core workflows via `make`

All day-to-day tasks are exposed as `make` targets:

| Command | Description |
| --- | --- |
| `make db-up` | Start the Postgres/PostGIS container (detached). |
| `make db-down` | Stop the database container. |
| `make migrate` | Run Alembic migrations (`api` project) against the current database. |
| `make dev-api` | Start the FastAPI dev server at `http://localhost:8000`. |
| `make dev-ui` | Start the Streamlit UI at `http://localhost:8501`. |
| `make ingest-firms ARGS="--day-range 1"` | Run the FIRMS ingestion CLI with optional args. |
| `make test` | Run API + UI test suites via `pytest`. |
| `make lint` | Run Ruff lint checks for API + UI. |
| `make clean` | Remove Python caches and build artifacts. |

> Many targets assume `make install` has been run at least once so the `uv` environments exist.

---

## 5. Recommended “from scratch” workflow

1. **Install tools** (Section 1) and clone the repo.
2. **Bootstrap deps:** `make install`
3. **Start database:** `make db-up`
4. **Run migrations:** `make migrate`
5. **(Optional) seed data:** `make ingest-firms`
6. **Start API:** `make dev-api`
7. **Start UI:** `make dev-ui`
8. **Verify:**  
   - API health: `curl http://localhost:8000/health`  
   - UI: open `http://localhost:8501/`
9. **Shut down:** `Ctrl+C` to stop API/UI, `make db-down` to stop Postgres.

---

## 6. Docker Compose alternative

If you prefer running everything via Docker:

```bash
docker compose up --build
```

This brings up API, UI, Postgres/PostGIS, and Redis in one command. Use `docker compose down` to stop the stack.

---

## 7. FIRMS ingestion quick test

After the database and migrations are ready:

```bash
# Fetch last 24h (default sources) into Postgres
make ingest-firms

# Or customize time window / sources
make ingest-firms ARGS="--day-range 3 --sources VIIRS_SNPP_NRT"
```

Then inspect the database:

```bash
docker compose exec db psql -U ${POSTGRES_USER:-wildfire} -d ${POSTGRES_DB:-wildfire} \
  -c "SELECT COUNT(*) FROM fire_detections;"
```

---

## 8. Troubleshooting

| Issue | Fix |
| --- | --- |
| `make: command not found` | Install GNU make (see Section 1) and restart your terminal. |
| `uv: command not found` | Re-install `uv` (`python -m pip install --upgrade uv`) and ensure it’s on `PATH`. |
| Docker commands fail | Make sure Docker Desktop / Engine is running and you have permission to run `docker compose`. |
| FIRMS ingestion fails with missing key | Ensure `.env` contains `FIRMS_MAP_KEY` and restart your shell so env vars are picked up. |
| Permission denied when cleaning | On Windows, run your terminal as Administrator if files are read-only. |

---

## 9. Helpful references

- [`Makefile`](../Makefile) – Source of all commands (`make help` lists them).
- [`docs/dev-python-env.md`](./dev-python-env.md) – Additional context on Python + `uv` workflows.
- [`README.md`](../README.md) – High-level project overview.
- [`docs/architecture.md`](./architecture.md) – System architecture and data flow.

If you run into anything missing from this guide, please open an issue or PR so we can keep onboarding smooth for the next contributor.

