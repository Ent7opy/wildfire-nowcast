# Docs Hub & Navigation

Purpose: Give newcomers and contributors a single navigation point, reduce duplication, and clarify which doc to read for which task.

---

## Quick Start

**New to the project?** Start here:
1. [`GETTING_STARTED.md`](GETTING_STARTED.md) – Complete step-by-step guide to get the app running with real data
2. [`WILDFIRE_NOWCAST_101.md`](WILDFIRE_NOWCAST_101.md) – Product and scope overview
3. [`architecture.md`](architecture.md) – How the system fits together

---

## Documentation Structure

### Getting Started & Setup
- **[`GETTING_STARTED.md`](GETTING_STARTED.md)** – Complete workflow from setup to running with real data
- **[`SETUP.md`](SETUP.md)** – Platform prerequisites, `.env` configuration, Docker/WSL notes
- **[`dev-python-env.md`](dev-python-env.md)** – Python/uv environment details

### System Overview
- **[`WILDFIRE_NOWCAST_101.md`](WILDFIRE_NOWCAST_101.md)** – Product vision, goals, and scope
- **[`architecture.md`](architecture.md)** – High-level system architecture and data flow

### Data Ingestion
- **[`ingest/ingest_firms.md`](ingest/ingest_firms.md)** – NASA FIRMS fire detection ingestion
- **[`ingest/ingest_weather.md`](ingest/ingest_weather.md)** – NOAA GFS weather ingestion
- **[`ingest/ingest_dem.md`](ingest/ingest_dem.md)** – Copernicus DEM preprocessing
- **[`ingest/data_quality.md`](ingest/data_quality.md)** – Validation checks and data quality

### Machine Learning
- **[`ml/denoiser.md`](ml/denoiser.md)** – Hotspot denoiser: design, training, and workflow
- **[`ml/spread_hindcast_builder.md`](ml/spread_hindcast_builder.md)** – Building spread forecast hindcast datasets
- **[`ml/calibration_and_weather_bias_correction.md`](ml/calibration_and_weather_bias_correction.md)** – Model calibration and weather bias correction
- **[`spread_model_design.md`](spread_model_design.md)** – Spread forecasting models: contract, design, and limitations

### Data & Schema
- **[`data/data_schema_fires.md`](data/data_schema_fires.md)** – Fire detection database schema
- **[`data/data_formats.md`](data/data_formats.md)** – Dataset shapes and formats
- **[`data/db-migrations.md`](data/db-migrations.md)** – Alembic migration workflow

### Technical References
- **[`grid.md`](grid.md)** – Analysis grid contract: CRS, indexing, terrain alignment
- **[`spread_model_design.md`](spread_model_design.md)** – Spread model contract and implementation details

---

## Reading Order (New Contributor)

1. **`GETTING_STARTED.md`** – Get the app running
2. **`WILDFIRE_NOWCAST_101.md`** – Understand the product vision
3. **`architecture.md`** – See how components fit together
4. **`SETUP.md`** → **`dev-python-env.md`** – Environment details
5. Ingestion docs as needed (FIRMS, weather, DEM)
6. ML docs when working on models

---

## Contributing to Docs

- Start with a one-line "audience" and "when to use this doc" if adding new pages
- Prefer linking to existing docs instead of restating the same bullets
- Keep commands copy/pastable; note platform-specific flags when relevant
- Update this hub when adding a new doc so navigation stays accurate
