## Architecture & Data Flow

This is the **high‑level mental model** for the project. It intentionally skips low‑level details and focuses on how the pieces fit together. See `docs/README.md` for the full doc map.

---

## 1. Big picture

- **Goal**: a web app that shows **current fires**, short‑term **spread forecasts (24–72h)**, and **risk** for new fires on a map.
- **Shape**:  
  UI (Streamlit) ⇄ API (FastAPI) ⇄ Database + ML outputs (Postgres/PostGIS + rasters/files).
- **Status today**:  
  - API has **health/version** endpoints and database wiring.  
  - UI has a **map with placeholder layers** and controls.  
  - Ingest pipelines exist for FIRMS fire detections, GFS 0.25° weather, and DEMs (terrain), but are not yet wired into the API or ML models.

---

## 2. Components

- **UI (`ui/`)**
  - Streamlit app (`app.py`) with:
    - Sidebar controls for **time window** and **layer toggles**.
    - Folium map (`components/map_view.py`) showing:
      - Placeholder fire markers.
      - Placeholder forecast and risk polygons.
    - Panels for click details and legends.
  - In the future, the UI will call the API to:
    - Fetch **fires**, **forecasts**, **risk maps**, and **summaries**.

- **API (`api/`)**
  - FastAPI app (`main.py`) with one router today: `routes/internal.py`.
  - Internal endpoints:
    - `GET /health` – basic liveness check.
    - `GET /version` – app name, version, git commit, environment.
  - Config (`config.py`):
    - Reads env via `AppSettings` (Pydantic Settings).
    - Builds a `database_url` from `POSTGRES_*` variables.
  - DB helper (`db.py`):
    - Creates a SQLAlchemy engine and `SessionLocal` for ORM work.
  - Migrations:
    - Alembic set up under `api/migrations/` with initial PostGIS schema.

- **Infra (`infra/`, `docker-compose.yml`)**
  - Docker Compose stack:
    - `api`: FastAPI + `uvicorn`, pointed at Postgres + Redis.
    - `ui`: Streamlit app talking to the API (`API_BASE_URL` + `API_PUBLIC_BASE_URL` for browser).
    - `db`: Postgres with PostGIS.
    - `redis`: cache / future worker backend.
  - `infra/README.md` explains ports, env vars, and migration commands.

- **ML & ingest (`ml/`, `ingest/`)**
  - `ml/` has its own `pyproject.toml` for model and experiment code.
  - `ingest/` now contains ingestion CLIs for FIRMS fire detections (`fire_detections` + `ingest_batches`), GFS 0.25° weather runs (`weather_runs` + NetCDF on disk), and DEM preprocessing (`terrain_metadata` + GeoTIFF/COG).

For the longer product + ML concept (pipeline, glossary, non‑goals), see `docs/WILDFIRE_NOWCAST_101.md`.

---

## 3. Data flow (conceptual)

This is the target flow; only pieces in **bold** exist in code today.

1. **Ingest**
   - **Planned**:
     - Download **FIRMS** fire detections.
     - Download short‑term **weather forecasts** (0–72h).
     - Load **terrain** (DEM) and derive slope/aspect.
   - Store:
     - Rasters (COG/Zarr/xarray) for grids.
     - Vectors (Postgres/PostGIS) for points, contours, AOIs.

2. **Validate & denoise**
   - **Implemented (optional)**:
     - Hotspot denoiser inference can run after FIRMS ingest (behind `DENOISER_ENABLED`) and writes
       `fire_detections.denoised_score` + `fire_detections.is_noise`.
   - **Planned ML**:
     - Basic QC rules to drop impossible points.

3. **Feature build**
   - **Planned**:
     - Reproject everything to a **common grid**.
     - Build local features per grid cell / fire cluster:
       - Recent detections, weather sequence, terrain, land cover, etc.

4. **Forecast & risk**
   - **Planned ML**:
     - Spread model to predict **probability of fire presence** for T+6/12/24/48/72h.
     - Risk model for **new‑fire probability** given fuels + weather.
     - Calibration and weather‑bias correction to clean up probabilities.
   - Outputs:
     - Probability rasters (per time step).
     - Vector contours / isochrones from rasters.

5. **Store & serve**
   - **Today**:
     - Database connection is ready (`api/config.py`, `api/db.py`).
   - **Planned**:
     - Store rasters as COG/Zarr for tile serving.
     - Store vectors in PostGIS (fires, contours, AOIs, history).
     - Add FastAPI routes for:
       - Fires in bbox/time window.
       - Forecast products for AOI / cluster.
       - Risk maps.
       - Historical metrics & exports.

6. **UI & summaries**
   - **Today**:
     - Streamlit map shows **placeholder** fires/forecast/risk.
     - Click interaction returns coordinates and shows simple details.
   - **Planned**:
     - UI fetches real data from the API instead of placeholders.
     - AOI selection (draw or pick) sends geometry to API.
     - An LLM (external API) turns numeric outputs into **short text summaries** with explicit uncertainty.

---

## 4. How to extend the system

When you add features, try to keep them inside this flow:

- **New data source?**
  - Put the raw pull + cleaning in `ingest/`.
  - Store long‑term copies in files/DB, not only in memory.

- **New model or metric?**
  - Implement in `ml/` and save outputs in a way that:
    - The API can read efficiently.
    - The UI can consume through a simple endpoint.

- **New API surface?**
  - Add a router module under `api/routes/` (e.g. `fires.py`, `forecast.py`).
  - Wire it in via `app.include_router(...)` in `api/main.py`.

- **New UI feature?**
  - Add a component in `ui/components/` and keep `app.py` as the glue.
  - Fetch data from the API, not directly from DB or files.

If you’re unsure where something should live, think:  
“Is this **ingest**, **ML/analytics**, **serving/API**, or **UI**?” and place it with similar code.


