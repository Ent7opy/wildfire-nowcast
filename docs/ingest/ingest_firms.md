# FIRMS Ingestion (NASA active fire detections)

## What it does
- Pulls the NASA FIRMS **area CSV API** for a configured bounding box and day range (1–10 days).
- Defaults to VIIRS NRT sources and accepts any FIRMS source key supported by the API.
- Validates/normalizes rows, inserts into Postgres `fire_detections` with dedupe on `(source, dedupe_hash)`, and records an `ingest_batches` entry per source.

## Inputs & validation
- Endpoint pattern: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{west,south,east,north}/{day_range}[/YYYY-MM-DD]`
- Required: `FIRMS_MAP_KEY` (API token from NASA FIRMS).
- Validation rules before insert:
  - Drop rows with missing, non-numeric, or out-of-range coordinates (lat ∈ [-90, 90], lon ∈ [-180, 180]).
  - Parse `acq_date` + `acq_time` into UTC timestamps.
  - Confidence: use numeric when present; otherwise map `confidence_text` (`high/nominal/low` or `h/n/l`) to 90/50/10; values outside 0–100 are nulled.
  - Brightness kept only when 200–500; otherwise nulled. Other numeric fields (`bright_t31`, `frp`, `scan`, `track`) kept when parseable.
  - Every raw row is preserved in `raw_properties`; validation counts are logged via `firms.validation_summary`.

## Configuration
- Environment variables (defaults in parentheses):
  - `FIRMS_MAP_KEY` – required.
  - `FIRMS_SOURCES` – comma list (default `VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT`).
  - `FIRMS_AREA` – `world` (→ `-180,-90,180,90`) or `west,south,east,north` bbox string.
  - `FIRMS_DAY_RANGE` – past days window, `1–10` (default `1`).
  - `FIRMS_REQUEST_TIMEOUT_SECONDS` – HTTP timeout (default `30`).
- CLI overrides (highest precedence):
  - `--day-range N`
  - `--area "w,s,e,n"` or `world`
  - `--sources "SRC1,SRC2"`
- Common FIRMS source keys: `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`, `MODIS_TERRA_NRT`, `MODIS_AQUA_NRT` (any FIRMS area API key is accepted).

## How to run
- Prereqs: Postgres/PostGIS running (API `.env` loaded for DB URL) and `.env` contains `FIRMS_MAP_KEY`.
- Recommended: `make ingest-firms ARGS="--day-range 3 --area -125,32,-114,43"`
- Direct: `uv run --project ingest -m ingest.firms_ingest --sources VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT`

## Outputs & persistence
- Inserts rows into `fire_detections` (geom Point 4326 + key FIRMS metrics) and keeps the full CSV row in `raw_properties`.
- Creates `ingest_batches` rows per source with `area`/`day_range` metadata and fetched/inserted/duplicate counters.
- Deduplication: `dedupe_hash = sha1(source|lat_rounded_4dp|lon_rounded_4dp|acq_time_utc)`; inserts use `ON CONFLICT (source, dedupe_hash) DO NOTHING` for idempotent re-runs.

## Notes
- BBox uses degrees (W,S,E,N); `world` ingests globally and is the largest download.
- Validation warnings are emitted as structured logs (`firms.validation*`); check logs when records look low.

## NRT vs historical data (important)
- The default sources (`*_NRT`) are **Near Real-Time** feeds. In practice they typically only retain the most recent **~7 days** of detections.
  - If you request a larger window, you may receive **0 rows** even though fires happened earlier.
- The FIRMS **area CSV API** day-range parameter is limited to **1–10 days**. For training workflows that need months/years of data:
  - Use **non-NRT archive sources** (if available for your AOI/use case), and/or
  - Use an **offline export** workflow (download historical data from FIRMS, then bulk load into `fire_detections`).

## Historical backfill (date-walk)

This repo includes a simple backfill tool that walks a date range backwards in <=10-day chunks using the
optional `/YYYY-MM-DD` "as-of date" suffix supported by the FIRMS area CSV endpoint.

- QA example (backfill 10 days):
  - `make ingest-firms-backfill ARGS="--start 2025-08-10 --end 2025-08-10 --area 13,40,23,49 --sources VIIRS_SNPP_SP,VIIRS_NOAA20_SP --chunk-days 1 --sleep-seconds 0.2 --max-chunks 1"`
- Full backfill example (months):
  - `make ingest-firms-backfill ARGS="--start 2025-01-01 --end 2025-12-31 --area 13,40,23,49 --sources VIIRS_SNPP_SP,VIIRS_NOAA20_SP --chunk-days 10 --sleep-seconds 0.2"`

Notes:
- You may need to choose **archive (non-NRT)** sources to access older dates; NRT sources often return 0 rows beyond ~7 days retention.
- For VIIRS, the archive sources that work well for historical backfill are typically `VIIRS_SNPP_SP` and `VIIRS_NOAA20_SP`.
- The backfill is idempotent thanks to dedupe on `(source, dedupe_hash)`.

