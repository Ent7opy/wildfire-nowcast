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

### API & JIT Forecasting
- **JIT Forecasting Workflow** – Generate forecasts for any global location on-demand (see below)

---

## JIT Forecasting

The JIT (Just-In-Time) forecast pipeline enables worldwide coverage without pre-provisioned data. Click any fire location to trigger automated terrain ingestion, weather ingestion, and forecast generation.

### Workflow

1. **Trigger**: POST to `/forecast/jit` with a bounding box
2. **Pipeline**: Background task ingests terrain → weather → generates forecast
3. **Poll**: GET `/forecast/jit/{job_id}` to check status
4. **Complete**: Results include forecast run_id, raster TileJSON URLs, and contour GeoJSON

### API Endpoints

#### POST /forecast/jit

Enqueue a JIT forecast for an arbitrary bbox.

**Request:**
```bash
curl -X POST http://localhost:8000/forecast/jit \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": [20.0, 40.0, 21.0, 41.0],
    "horizons_hours": [24, 48, 72]
  }'
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

#### GET /forecast/jit/{job_id}

Poll for job status.

**Request:**
```bash
curl http://localhost:8000/forecast/jit/550e8400-e29b-41d4-a716-446655440000
```

**Response (in progress):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "ingesting_weather",
  "progress_message": "Fetching weather data...",
  "created_at": "2026-01-19T12:00:00Z",
  "updated_at": "2026-01-19T12:00:15Z"
}
```

**Response (completed):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress_message": "Forecast complete!",
  "result": {
    "run_id": 123,
    "raster_urls": ["..."],
    "contour_geojson": {"type": "FeatureCollection", "features": [...]}
  }
}
```

### Status Flow

- `pending` → Job queued, waiting to start
- `ingesting_terrain` → Downloading Copernicus DEM tiles
- `ingesting_weather` → Fetching NOAA GFS weather data
- `running_forecast` → Generating spread forecast
- `completed` → Forecast ready (check `result` field)
- `failed` → Error occurred (check `error` field)

### Integration

The UI automatically triggers JIT forecasts when clicking fires in regions without pre-ingested data. The polling component (`ui/components/forecast_status.py`) checks status every 2 seconds and updates the map on completion.

For programmatic access, poll `/forecast/jit/{job_id}` until `status` is `completed` or `failed`.

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
