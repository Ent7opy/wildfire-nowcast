## WILDFIRE_NOWCAST_101

> **Read this first** before working on any task in this repo (human or AI). See `docs/README.md` for navigation to the rest of the docs set.

This document is the **source-of-truth overview** for the Wildfire Nowcast & Forecast project.  
It defines what we are building, for whom, with what data and stack, and how work in this repo should be structured.  
It describes the **target system**; for what is implemented today, see `docs/architecture.md`.

---

## 1. Project in One Paragraph

**Wildfire Nowcast & Forecast** is an **AI-first web application** for monitoring active wildfires and predicting short-term spread (roughly **24–72 hours**) using **open satellite data + weather + terrain + ML**.

The app should let users:
- See **recent active fires** on a map.
- Inspect fires in context (time history, terrain, local weather).
- View **probabilistic spread forecasts** with clear uncertainty.
- Explore **risk maps** and simple history.
- Get **short natural-language summaries** for a selected area.

Target users: analysts, journalists, NGOs, and operational teams who need a **lightweight but serious** tool, not a toy demo.

The project is a **3-month side project** for a small team (≈3–4 people), so **simplicity and focus** matter.

---

## 2. Core User Capabilities

At a high level, the app should support:

1. **Explore the map**
   - Pan/zoom around a world map.
   - Filter fires by **time window** (e.g. last 6–48h).
   - See active fires as points or clusters.

2. **Inspect a specific fire / cluster**
   - Click to see:
     - First/last detection time.
     - Confidence / brightness / sensor.
     - Simple recent history.
   - Show **local context**: wind, humidity, temperature, terrain slope/elevation, maybe land cover.

3. **View AI forecast (24–72h)**
   - Toggle forecast overlays.
   - View spread as **probability rasters / contours** at multiple horizons (T+6/12/24/48/72h).
   - Show timestamps (when forecast was run, horizon) and an **uncertainty legend**.

4. **Add context**
   - Optional layers:
     - Population density.
     - Vegetation / land cover.
     - Roads / infrastructure.
   - At least rough visual impact context; simple metrics are a plus.

5. **Look at history and risk**
   - Historical fire patterns for a region (counts, simple charts, “season replay”).
   - **Fire-risk heatmap**: areas where new fires are more likely given **current weather and fuels**.

6. **Focused AI summary (LLM)**
   - User draws or selects an **Area of Interest (AOI)**.
   - System returns a **1–2 paragraph summary**:
     - Current situation.
     - Likely spread (24–72h).
     - Confidence level & key drivers (wind, slope, dryness).
     - Optionally population / infrastructure context.
   - Summary must be **tied to numeric outputs**, not hallucinated.

7. **Export & share**
   - Export:
     - **PNG** of current map view (with legend + timestamp).
     - **CSV** of fire points / metrics in current filter/bbox.
     - **GeoJSON** of forecast contours / AOIs.
   - Optional: shareable links or simple API integration.

---

## 3. Data Sources & Roles (What We Use, Why)

### 3.1 Core Data

- **Fire detections**
  - Source: **NASA FIRMS** (e.g. VIIRS, MODIS).
  - Role: ground truth for **where fires are now / recently**.
  - Fields: lat, lon, timestamp, sensor, confidence, brightness-like metrics.

- **Weather**
  - Short-term forecast (0–72h):
    - Wind speed/direction, relative humidity, temperature.
    - Optional: precipitation or dryness proxies.
  - Role: **main driver** for forward spread.
  - Additional reanalysis (e.g. ERA5):
    - For **training**, evaluating, and bias-correcting forecasts.

- **Terrain**
  - Source: **SRTM or similar DEM**.
  - Role:
    - Derive **slope** and **aspect** (slope direction).
    - Influence spread (fires move faster upslope, etc.).
  - Conventions: see `docs/grid_choice.md` and `docs/terrain_grid.md` (grid + terrain alignment contract).

### 3.2 Optional Context Layers

- Land cover / vegetation type.
- Population density.
- Roads / infrastructure (e.g. from OSM).

Role: **context for risk and impact**, not required for the core spread model.

---

## 4. AI / ML Components (What “AI-first” Means Here)

AI is not a garnish. It sits in the **core pipeline**.

Planned components:

1. **Hotspot denoiser (classification model)**
   - Input: FIRMS detections + local context (spectral fields, neighborhood density, time pattern, maybe terrain/weather).
   - Output: `P(real_fire)` vs `P(artefact/noise)`.
   - Use: filter or downweight noisy detections before spread modeling.

2. **Spread model (24–72h forecast)**
   - Inputs:
     - Clustered fire detections (local regions, e.g. 10×10 km).
     - Weather forecast sequence (wind, RH, temperature).
     - Terrain slope + aspect.
   - Output:
     - **Probability of fire presence** on a local grid for T+6/12/24/48/72h.
     - Uncertainty estimates (ensembles / Monte Carlo).
   - Details / assumptions / limitations: see `docs/spread_model_design.md`.

3. **Probability calibration**
   - Learned from historical data:
     - When the model says **30%**, events should occur ≈30% of the time.
   - Output: calibration functions that post-process model probabilities.

4. **Weather bias correction**
   - Compare forecast weather to reanalysis/ground truth historically.
   - Learn **local corrections** for wind, humidity, temperature, etc.
   - Apply corrections to live forecasts to reduce systematic biases.

5. **Fire-risk index (new-fire probability)**
   - Combine:
     - Static: vegetation, slope, historical fire frequency.
     - Dynamic: temperature, RH, wind, recent rain.
   - Output: normalized index / probability that **new fires** might start or spread more easily today.

6. **LLM-based summaries (application-level AI)**
   - Not the forecasting model itself.
   - Takes structured outputs and generates:
     - Human-readable summary of current fires + spread + risk.
     - Explanation of **drivers and confidence**, using only provided data.

---

## 5. System Flow (Conceptual Pipeline)

**Ingest → Validate/Denoise → Feature Build → Forecast → Calibrate → Store/Serve → UI & Summaries**

As of late 2025, only some parts of this pipeline are implemented in code; others are still design targets.  
For a current-status walkthrough (what exists vs planned), see `docs/architecture.md`. Roughly:
- **Implemented**: FIRMS fire detection ingest + dedupe (`fire_detections`, `ingest_batches`), GFS 0.25° weather ingest + `weather_runs`, DEM stitching + `terrain_metadata`, basic FastAPI API with DB wiring, and a Streamlit UI with placeholder layers.
- **Partially implemented**: spread model (heuristic v0 + learned v1 baseline). See `docs/spread_model_design.md` for what is implemented today and its limitations.
- **Not yet implemented**: risk models, probability calibration and weather bias correction, tile serving for UI consumption, LLM AOI summaries, and background workers/caching.

1. **Data ingest**
   - Periodically pull:
     - FIRMS detections.
     - Weather forecasts (0–72h).
     - Terrain (static DEM).
   - Store in internal formats:
     - Rasters: COG/Zarr/xarray.
     - Vectors: Postgres/PostGIS.

2. **Validation & denoising**
   - Run hotspot denoiser + basic QC.
   - Mark or drop suspicious points.

3. **Feature building**
   - Reproject to a **common grid**.
   - Derive slope/aspect and other terrain metrics.
   - Extract forecast weather features per time step.
   - Aggregate detections into **fire clusters/ignition areas**.

4. **AI forecasting**
   - For each active cluster / AOI:
     - Run spread model over 24–72h horizon.
     - Use bias-corrected weather fields.
     - Optionally generate ensembles to estimate uncertainty.

5. **Calibration & product generation**
   - Apply probability calibration.
   - Produce:
     - Probability rasters per time step.
     - Probability contours / iso-lines for chosen thresholds.

6. **Storage & tiling**
   - Store rasters as **COGs/Zarr** suitable for tile serving.
   - Store vectors (contours, AOIs, etc.) in **PostGIS**.

7. **Serving (API layer)**
   - Endpoints for:
     - Fires in bbox/time window.
     - Forecast products for AOI or cluster.
     - Risk maps, historical metrics.
     - Exports (PNG/CSV/GeoJSON).
     - AOI text summaries.

8. **UI (web app)**
   - Map-based interface (Streamlit MVP).
   - Tools: search, filters, layer toggles, click-to-inspect, history, exports.
   - Button/flow for **“Summarize this area”**.

---

## 6. Planned Tech Stack (Initial, Not Dogma)

> Agents: **do not change stack choices without an explicit issue / human approval.**  
> Prefer integrating with what already exists.

- **Language:** Python.
- **Backend API:** FastAPI.
- **UI (MVP):** Streamlit.
- **Database:** Postgres + PostGIS.
- **Caching & background work:** Redis + worker queue (RQ / Celery / similar).
- **ML/AI libraries:**
  - scikit-learn / simple gradient boosting, plus PyTorch-like if needed.
  - xarray + dask for gridded climate/forecast data.
  - External LLM API for AOI summaries.
- **Geospatial:**
  - Rasterio / GDAL for rasters and DEM handling.
  - GeoPandas + Shapely for vector data and AOIs.
- **Map serving:**
  - **Planned**: tile server like **TiTiler** for raster COG/Zarr tiles, vector layers served via API from PostGIS.
  - **Today**: weather grids are stored as NetCDF under `data/weather/`; map tiles are not yet served directly.
- **Packaging / deployment:**
  - Docker / Docker Compose.

Stack is **tentative**, but any deviation should be **intentional and discussed**, not ad-hoc.

---

## 7. Repository Structure

This repository is organized into top-level directories that separate concerns:

- **`api/`** – FastAPI backend application providing REST endpoints for fires, forecasts, risk maps, historical data, and AOI summaries.

- **`ui/`** – Streamlit web application providing the map-based interface, layer controls, filters, inspection tools, and summary generation UI.

- **`ml/`** – Machine learning models, training scripts, and experiments. Includes hotspot denoiser, spread forecasting model, probability calibration, weather bias correction, and fire-risk index components.

- **`ingest/`** – Data ingestion pipelines for pulling and processing FIRMS fire detections, GFS weather forecasts, terrain data (DEM), and optional context layers (land cover, population, infrastructure).

- **`infra/`** – Infrastructure configuration (alongside the root `docker-compose.yml`) including deployment notes, CI/CD stubs, and operational tooling.

For a quick overview, see `PROJECT.md`. Each directory may contain its own README, requirements, and configuration files as components are developed.

---

## 8. Work Structure: Epics & Labels

### 8.1 Epics (high-level)

Current high-level epics (each is an issue of `type: epic`):

1. Project Setup & Architecture  
2. Core Data Ingestion & Storage (FIRMS, Weather, Terrain)  
3. Terrain & Grid Preprocessing  
4. Hotspot Denoiser (ML v1)  
5. Spread Forecasting Model (24–72h v1)  
6. Probability Calibration & Weather Bias Correction  
7. Spatial Serving Layer (API + Tiles)  
8. Web App MVP: Fires Map & Inspection  
9. Forecast Visualization, Risk Maps & Exports  
10. Historical Explorer & Evaluation Views  
11. LLM AOI Summaries & Explainability  
12. Background Workers, Caching & Performance  
13. Quality, Evaluation & Documentation  

Each epic has a checklist that links to smaller issues.

### 8.2 Labels

We intentionally keep labels small and consistent.

**Type labels** (exactly one per issue):

- `type: epic`
- `type: feature`
- `type: task`
- `type: bug`
- `type: spike`
- `type: docs`

**Area labels** (one or more per issue):

- `area: infra-dev`      – repo, tooling, Docker, CI, general infra
- `area: ingest`         – FIRMS, weather, DEM, data pipelines
- `area: data-store`     – Postgres/PostGIS, raster storage
- `area: terrain-grid`   – DEM, slope/aspect, grid definition
- `area: ml-denoiser`    – hotspot classification
- `area: ml-spread`      – spread model, calibration, bias correction
- `area: risk-index`     – new-fire probability, risk maps
- `area: api-backend`    – FastAPI endpoints, schemas, routing
- `area: tiles`          – tile server, raster/vector tiles
- `area: ui-map`         – map UI, layers, filters, interactions
- `area: ui-summary`     – AOI summary UI + LLM integration

Agents should **apply or respect these labels** when suggesting or modifying issues.

---

## 9. Guidelines for AI Agents Working on This Repo

This section is specifically for any **AI agent** consuming this file.

1. **Read before generating code**
   - Check:
     - The issue description.
     - Linked epic.
     - This `WILDFIRE_NOWCAST_101.md`.
   - Make sure the change fits the overall architecture and stack.

2. **Stay within the stack**
   - Use the tools listed in §6.
   - Do not introduce new heavy dependencies (frameworks, databases, services) unless the issue explicitly asks for it.
   - Prefer minimal, composable utilities over big “magic” libraries.

3. **Don’t break the pipeline concept**
   - Keep the main flow as:
     - Ingest → Denoise → Features → Forecast → Calibrate → Store/Serve → UI/Summaries.
   - New code should slot into this flow, not work around it.

4. **Be explicit about assumptions**
   - If a requirement is ambiguous, **call it out in comments or PR description**.
   - Prefer simple, testable decisions over clever but opaque solutions.

5. **Testing and docs**
   - For any non-trivial change:
     - Add or update tests where reasonable.
     - Update relevant docs (README, architecture notes, or module docstrings).

6. **Uncertainty & honesty**
   - This project is about **clear uncertainty communication**:
     - Don’t claim deterministic accuracy where we only have probabilities.
     - When adding UI/summary logic, always surface uncertainty and assumptions.

7. **Performance and scale**
   - This is a **side project**, not a hyperscale product.
   - Prefer **clarity and maintainability** over premature optimization.
   - But avoid obviously pathological patterns (e.g. per-tile DB queries when batching is easy).

---

## 10. Non-goals / Scope Boundaries

To keep the project focused:

- We are **not** building a full incident management system (no tasking, SMS alerts, etc.).
- We are **not** solving long-term fire behavior (multi-week growth, full fuel models).
- We are **not** doing closed or proprietary data sources unless explicitly added.
- We are **not** promising operational-grade reliability; this is a serious **but experimental** tool.

---

## 11. Quick Glossary

- **Nowcast** – Best estimate of the current state (fires right now).
- **Forecast** – Predicted future fire spread (24–72h).
- **AOI (Area of Interest)** – User-selected region (drawn polygon or pre-defined box).
- **COG (Cloud-Optimized GeoTIFF)** – Raster format optimized for HTTP range requests and tiling.
- **Reanalysis** – Consistent historical climate dataset (e.g. ERA5) used for training/evaluation.
- **Calibration** – Adjusting predicted probabilities so they match observed frequencies.

---

If something you’re about to build or change **contradicts** this document,  
it probably needs a discussion and an explicit issue first.


