# Getting Started

**Complete step-by-step guide to get the Wildfire Nowcast & Forecast app running with real data.**

This guide walks you through the entire setup process from scratch. For detailed information about specific components, see the other docs in this directory.

---

## Prerequisites

Before starting, ensure you have:

| Tool | Windows | macOS | Linux |
| --- | --- | --- | --- |
| **Git** | [git-scm.com](https://git-scm.com/download/win) | `brew install git` | `sudo apt-get install git` |
| **Docker** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) | [Docker Desktop](https://www.docker.com/products/docker-desktop/) | `sudo apt-get install docker.io` |
| **GNU make** | `choco install make` or use WSL | `brew install make` | `sudo apt-get install build-essential` |

**Optional** (for manual setup or weather ingest):
- **Python 3.11** and **uv**: See [`SETUP.md`](SETUP.md) for details
- **ecCodes**: Windows (use WSL + `sudo apt-get install libeccodes0`), macOS (`brew install eccodes`), Linux (`sudo apt-get install libeccodes0`)

---

## Quick Start (Docker Compose - Recommended)

The easiest way to get started is using Docker Compose, which runs the entire stack (API, UI, Database, Redis, TiTiler) with one command.

### Step 1: Clone and Configure

```bash
git clone <repository-url>
cd wildfire-nowcast
```

Create a `.env` file in the repo root:

```env
# Required for FIRMS ingest
FIRMS_MAP_KEY=your_firms_api_key_here

# Database configuration (optional - defaults work)
POSTGRES_USER=wildfire
POSTGRES_PASSWORD=wildfire
POSTGRES_DB=wildfire
```

**Get your FIRMS API key**: Sign up at [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/) to get a free API key.

### Step 2: Start the Full Stack

Start all services (API, UI, Database, Redis, TiTiler):

```bash
docker compose up -d
```

This will:
- Build and start the API service on `http://localhost:8000`
- Build and start the UI service on `http://localhost:8501`
- Start PostgreSQL/PostGIS database on `localhost:5432`
- Start Redis on `localhost:6379`
- Start TiTiler (for raster tiles) on `http://localhost:8080`

Verify all services are running:
```bash
docker compose ps
```

### Step 3: Run Database Migrations

Set up the database schema:

```bash
docker compose exec api uv run alembic upgrade head
```

Or if you have `make` installed:
```bash
make migrate
```

### Step 4: Ingest Fire Detection Data

To see fires on the map, you need to ingest FIRMS data. Run the ingest command inside the API container:

```bash
# Option 1: Worldwide (last 3 days) - slower but comprehensive
docker compose exec api uv run --project ingest -m ingest.firms_ingest --day-range 3 --area world

# Option 2: Regional (faster) - example for Balkans
docker compose exec api uv run --project ingest -m ingest.firms_ingest --day-range 3 --area 13,40,23,49

# Option 3: Custom region (west,south,east,north)
docker compose exec api uv run --project ingest -m ingest.firms_ingest --day-range 3 --area -125,32,-114,43
```

**Or use make** (if installed locally):
```bash
make ingest-firms ARGS="--day-range 3 --area world"
```

**Notes:**
- `--day-range` can be 1-10 days (FIRMS API limit)
- NRT (Near Real-Time) sources typically only retain ~7 days of data
- For historical data, use backfill: `make ingest-firms-backfill ARGS="--start YYYY-MM-DD --end YYYY-MM-DD --area ..."`

See [`ingest/ingest_firms.md`](ingest/ingest_firms.md) for detailed options.

### Step 5: View the App

1. Open your browser to `http://localhost:8501`
2. You should see a map with fire detections (red markers/clusters)
3. Use the sidebar to:
   - Adjust time window (last 6/12/24/48 hours)
   - Filter by minimum confidence
   - Toggle denoiser filtering
   - Enable forecast/risk layers (if data is available)

### Step 6: Stop the Stack

When you're done:
```bash
docker compose down
```

To stop and remove volumes (clears database):
```bash
docker compose down -v
```

---

## Alternative: Manual Setup (For Development)

If you prefer to run services manually for development/debugging:

### Step 1: Install Dependencies

```bash
make install
```

This installs dependencies for `api/`, `ui/`, `ml/`, and `ingest/` subprojects.

### Step 2: Start Database Only

```bash
make db-up
```

### Step 3: Run Migrations

```bash
make migrate
```

### Step 4: Ingest Data

```bash
make ingest-firms ARGS="--day-range 3 --area world"
```

### Step 5: Start Services Manually

**Terminal 1 - API:**
```bash
make dev-api
```

**Terminal 2 - UI:**
```bash
make dev-ui
```

---

## Generating and Viewing Forecast Overlays

This section shows how to generate a spread forecast and see the overlay in the UI. The workflow involves ingesting weather data, generating a forecast via API, and viewing the result.

### Step 1: Ingest Weather Data

Weather forecasts require GFS 0.25° data. Ingest for a specific forecast run time:

```bash
# Using Docker Compose (recommended)
docker compose exec api uv run --project ingest -m ingest.weather_ingest --run-time <YYYY-MM-DD>T00:00Z

# Or using make
make ingest-weather ARGS="--run-time <YYYY-MM-DD>T00:00Z"
```

**Note:** Requires ecCodes installed (see Prerequisites). **Replace `<YYYY-MM-DD>` with today's date or a recent date** (e.g., `2026-01-18`). GFS forecast data is only available for recent runs (typically the last few days to weeks), so dates older than ~1 week will likely fail. The `--run-time` should be a valid GFS run time (00Z, 06Z, 12Z, or 18Z).

You can verify the weather data was ingested by checking the database:

```bash
docker compose exec db psql -U wildfire -d wildfire -c "SELECT id, model, run_time, status FROM weather_runs ORDER BY created_at DESC LIMIT 5;"
```

### Step 2: Ensure Fire Data Exists

Make sure you have recent fire detections (if you haven't already):

```bash
docker compose exec api uv run --project ingest -m ingest.firms_ingest --day-range 3 --area world
```

### Step 2.5: Ingest Terrain Data (Required for Forecast)

Forecast generation requires terrain data for the region grid specification:

```bash
# Using Docker Compose (use same bbox as forecast)
docker compose exec api uv run --project ingest -m ingest.dem_preprocess --region-name balkans --bbox 19.0 41.0 23.0 43.0 --cog

# Or using make
make ingest-dem ARGS="--region-name balkans --bbox 19.0 41.0 23.0 43.0 --cog"
```

**Note:** The region name and bounding box should match what you'll use in Step 3. For other regions, adjust accordingly.

You can verify terrain data was ingested:

```bash
docker compose exec db psql -U wildfire -d wildfire -c "SELECT id, region_name, dem_source FROM terrain_metadata ORDER BY created_at DESC LIMIT 5;"
```

### Step 3: Generate a Forecast via API

Call the forecast generation endpoint with a bounding box and region name:

```bash
curl -X POST http://localhost:8000/forecast/generate \
  -H "Content-Type: application/json" \
  -d '{
    "min_lon": 19.0,
    "min_lat": 41.0,
    "max_lon": 23.0,
    "max_lat": 43.0,
    "region_name": "balkans",
    "horizons_hours": [24, 48, 72]
  }'
```

**Expected response:** JSON with a non-null `run.id` field and contour GeoJSON.

Example response:
```json
{
  "run": {
    "id": 1,
    "model_name": "HeuristicSpreadModelV0",
    "model_version": "v0",
    "forecast_reference_time": "<YYYY-MM-DD>T00:00:00+00:00",
    "region_name": "balkans",
    "status": "completed"
  },
  "rasters": [...],
  "contours": {"type": "FeatureCollection", "features": [...]}
}
```

**Note the `run.id`** - this is used by the UI to fetch the forecast overlay tiles.

### Step 4: View Forecast Overlay in UI

1. Open the UI at `http://localhost:8501`
2. Navigate to the area you generated the forecast for (Balkans in the example above)
3. In the sidebar, enable the **"Show forecast overlay"** toggle
4. The forecast contours should appear on the map

Alternatively, if you click **"Generate Spread Forecast"** in the UI (in the click details panel after clicking on the map), the overlay will automatically appear after generation completes.

**Verification:**
- Open browser DevTools Network tab
- Filter requests to `/tiles/forecast_contours`
- Confirm requests include `run_id=` parameter matching the generated run

---

## Optional: Additional Terrain Data (For Other Regions)

If you want to work with additional regions beyond the Balkans example above:

```bash
# Example for a different region
docker compose exec api uv run --project ingest -m ingest.dem_preprocess --region-name <region> --bbox <min_lon> <min_lat> <max_lon> <max_lat> --cog
```

---

## Troubleshooting

### Services not starting
- Ensure Docker Desktop/Engine is running
- Check `docker compose ps` to see service status
- View logs: `docker compose logs api` or `docker compose logs ui`

### API returns 422 errors
- Ensure bounding box parameters are being sent
- Verify the database is running: `docker compose ps`
- Check API logs: `docker compose logs api`

### No fires showing on map
- Check that FIRMS ingest completed successfully
- Verify `FIRMS_MAP_KEY` is set in `.env`
- Try a wider time window or lower minimum confidence filter
- Check API logs: `docker compose logs api`

### Database connection errors
- Ensure database is running: `docker compose ps db`
- Verify `.env` has correct `POSTGRES_*` variables
- Check migrations ran: `docker compose exec api uv run alembic upgrade head`
- View database logs: `docker compose logs db`

### FIRMS ingest returns 0 rows
- NRT sources only retain ~7 days; try a smaller `--day-range`
- For older data, use backfill with archive sources
- Verify your API key is valid

### Port conflicts
- If ports 8000, 8501, 5432, 6379, or 8080 are already in use, stop the conflicting services or modify `docker-compose.yml`

---

## Next Steps

- **Explore the codebase**: See [`docs/architecture.md`](architecture.md) for system overview
- **Understand the data**: See [`data/`](data/) docs for schemas and formats
- **Work with ML models**: See [`ml/`](ml/) docs for denoiser and spread models
- **Customize ingestion**: See [`ingest/`](ingest/) docs for detailed ingest options

---

## Quick Reference

### Docker Compose (Recommended)

```bash
# Start everything
docker compose up -d

# Run migrations
docker compose exec api uv run alembic upgrade head

# Ingest data
docker compose exec api uv run --project ingest -m ingest.firms_ingest --day-range 3 --area world
docker compose exec api uv run --project ingest -m ingest.weather_ingest --run-time <TODAY-DATE>T00:00Z  # Use today's date, e.g., 2026-01-18

# Generate forecast (requires weather + fire data)
curl -X POST http://localhost:8000/forecast/generate \
  -H "Content-Type: application/json" \
  -d '{"min_lon": 19.0, "min_lat": 41.0, "max_lon": 23.0, "max_lat": 43.0, "region_name": "balkans", "horizons_hours": [24, 48, 72]}'

# View logs
docker compose logs -f api
docker compose logs -f ui

# Stop everything
docker compose down
```

### Manual Setup

```bash
# Setup (one-time)
make install
make db-up
make migrate

# Ingest data
make ingest-firms ARGS="--day-range 3 --area world"
make ingest-weather ARGS="--run-time <TODAY-DATE>T00:00Z"  # Use today's date, e.g., 2026-01-18

# Run services
make dev-api    # Terminal 1
make dev-ui     # Terminal 2

# Generate forecast (in another terminal, after services are running)
curl -X POST http://localhost:8000/forecast/generate \
  -H "Content-Type: application/json" \
  -d '{"min_lon": 19.0, "min_lat": 41.0, "max_lon": 23.0, "max_lat": 43.0, "region_name": "balkans", "horizons_hours": [24, 48, 72]}'

# Stop database
make db-down
```

For all available commands: `make help`
