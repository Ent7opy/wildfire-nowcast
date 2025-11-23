## Project structure

**Wildfire Nowcast & Forecast** is an AI‑first web app for monitoring active wildfires and short‑term spread using open satellite, weather, and terrain data.

For a friendly overview and quickstart, see `README.md`. For deeper architecture and pipeline notes, see `docs/architecture.md` and `docs/WILDFIRE_NOWCAST_101.md`.

---

## Repository layout

- **`api/`** – FastAPI backend (currently health/version + DB wiring; future fire/forecast/risk endpoints).
- **`ui/`** – Streamlit UI with a Folium map, sidebar controls, and placeholder layers.
- **`ml/`** – Python project for models, training scripts, and experiments.
- **`ingest/`** – Planned data ingestion pipelines (FIRMS, weather, DEM, context layers).
- **`infra/`** – Docker Compose stack and infra docs (Postgres+PostGIS, Redis, API, UI).

Each area can grow its own README and configs as it matures.

---

## Development environment (short version)

- Python **3.11.x** (pinned in `.python-version`).
- Dependencies managed with `uv` in `api/`, `ui/`, and `ml/`.
- Canonical setup and examples live in `docs/dev-python-env.md`.

