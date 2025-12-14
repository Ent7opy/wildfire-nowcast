## Wildfire Nowcast & Forecast

**Wildfire Nowcast & Forecast** is a small Python project to **watch active wildfires on a map** and eventually **predict short‑term spread (24–72 hours)** using open data (satellites, weather, terrain) and ML.

Right now the repo has:
- **`api/`** – FastAPI backend with health/version endpoints and a Postgres/PostGIS connection wired up.
- **`ui/`** – Streamlit map UI with placeholder layers (fires, forecast, risk) and simple controls.
- **`ml/`** – Placeholder Python project for future models and experiments.
- **`ingest/`** – Reserved for data ingest pipelines (FIRMS, weather, terrain, etc.).
- **`infra/`** – Docker Compose and infra docs for running the full stack locally.

If you want the deep product/ML vision and original longform overview, see `docs/WILDFIRE_NOWCAST_101.md`. For grid/terrain conventions and alignment guarantees, see [`docs/terrain_grid.md`](docs/terrain_grid.md) (and `docs/grid_choice.md`). For a full from-scratch setup (prerequisites, `.env`, `make` workflows) see [`docs/SETUP.md`](docs/SETUP.md). For navigation to all docs, start at [`docs/README.md`](docs/README.md).

---

## What the app will do (high level)

The goal is a **map‑first web app** where users can:
- Browse **recent fires** worldwide.
- Click a fire/area to see **context** (time history, terrain, weather).
- Toggle **forecast** and **risk** layers for the next 1–3 days.
- Get a short **text summary** for a selected area, driven by model outputs.

The target users are analysts, journalists, NGOs, and operations teams who need a **simple but serious** tool, not a toy demo.

---

## Quickstart (Make workflow) (RECOMMENDED)

Most day-to-day tasks are one-liners; see `make help` for the full list:

```bash
make install          # sync deps for api/ui/ml/ingest
make db-up            # start Postgres/PostGIS
make migrate          # apply Alembic migrations
make dev-api          # http://localhost:8000
make dev-ui           # http://localhost:8501
# optional ingests
make ingest-firms     # NASA FIRMS fire detections
make ingest-weather   # NOAA GFS weather (pass ARGS="--run-time 2025-12-06T00:00Z")
make ingest-dem       # Copernicus DEM preprocessing (pass ARGS="--cog")
```

For platform specifics (.env template, Docker stack, troubleshooting), use [`docs/SETUP.md`](docs/SETUP.md).

---

## Quickstart (Python dev workflow)

We standardize on **Python 3.11.x** and use [`uv`](https://pypi.org/project/uv/) in each subproject. The short version:

- **API (FastAPI)**
  ```bash
  cd api
  uv sync
  uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
  # then hit http://localhost:8000/health and /version
  ```

- **UI (Streamlit)**
  ```bash
  cd ui
  uv sync
  uv run streamlit run app.py
  # then open the printed URL, usually http://localhost:8501/
  ```

For more on `uv`, venvs, and CI conventions, see `docs/dev-python-env.md`.

---

## Quickstart (Docker Compose stack)

If you want the whole stack (API + UI + Postgres/PostGIS + Redis) with one command:

```bash
docker compose up --build
```

Key endpoints:
- **API**: `http://localhost:8000/health`
- **UI**: `http://localhost:8501/`
- **Postgres**: `localhost:5432` (PostGIS enabled)
- **Redis**: `localhost:6379`

More details and environment variables are in `infra/README.md`.

---

## Architecture at a glance

High‑level flow:
- **UI (`ui/`)** – Streamlit app using Folium to render a map and collect user input (time window, layer toggles, clicks).
- **API (`api/`)** – FastAPI app that will expose fire, forecast, and risk endpoints; currently only internal health/version routes are implemented.
- **Data & storage** – Postgres + PostGIS for vector data; NetCDF on disk for weather forecasts.
- **ML & ingest (`ml/`, `ingest/`)** – ML experiments plus ingestion pipelines (FIRMS fire detections, GFS weather).
### Ingestion quickstart (FIRMS + weather)

```bash
cd ingest
uv sync
uv run -m ingest.firms_ingest --day-range 1 --area world
# or use the shortcut
make ingest-firms ARGS="--day-range 3 --sources VIIRS_SNPP_NRT"

# weather ingest (requires ecCodes for cfgrib)
make ingest-weather ARGS="--run-time 2025-12-06T00:00Z"
```

FIRMS hits NASA FIRMS and logs an `ingest_batches` row per source with deduped inserts into `fire_detections`. Weather ingest downloads NOAA GFS GRIB, writes NetCDF under `data/weather/...`, and tracks runs in `weather_runs`.

See `docs/ingest/data_quality.md` for the quick reference on ingest validation checks and how to read the tagged log messages.


For a short architecture + data‑flow walkthrough (including future ML pieces), see `docs/architecture.md`. For setup specifics and day-to-day commands, see `docs/SETUP.md`.

---

## Working on the repo

- **Issue organization** – Use simple labels by type (`type: feature`, `type: bug`, `type: docs`, etc.) and area (`area: api-backend`, `area: ui-map`, `area: ml-spread`, `area: ingest`, `area: infra-dev`, …) when filing work.
- **Docs** – Keep this `README.md` short. Put deeper explanations in `docs/` (see `docs/README.md` for the map; examples: `docs/architecture.md`, `docs/dev-python-env.md`, `docs/data/db-migrations.md`, `docs/WILDFIRE_NOWCAST_101.md`).
- **Testing & quality** – When adding non‑trivial code, add or update tests where it makes sense and update relevant docs.

If you’re unsure how a change fits into the bigger picture, start with `docs/architecture.md` and then check the open issues/epics.


