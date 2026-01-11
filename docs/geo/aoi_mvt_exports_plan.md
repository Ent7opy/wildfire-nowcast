## 1. Executive Summary

This repo is already set up for spatial serving: **FastAPI** (`api/`), **Postgres + PostGIS** via Alembic migrations (`api/migrations/`), and **raster tile serving** via **TiTiler** (`docker-compose.yml` + `api/routes/forecast.py` returning TiTiler TileJSON URLs). What’s missing for “end-to-end AOI + MVT + exports” is (a) a first-class AOI table + CRUD API, (b) a vector tile (MVT) strategy, and (c) export endpoints (sync + async).

**Recommended default architecture (MVP-first, scale-ready):**

- **AOIs stored in PostGIS** (EPSG:4326) with validation + limits at API boundaries.
- **Vector tiles served on-the-fly from PostGIS** using a dedicated tile server (**Martin** or **pg_tileserv**) as the primary approach, with **optional FastAPI proxy** for auth/rate-limits.
- **Exports**:
  - **Sync exports** for small AOIs (GeoJSON/CSV).
  - **Async export jobs** for large AOIs (Redis-backed queue + worker, results stored under `./data/exports/...` for MVP; optional S3/GCS later).

This plan is designed to integrate cleanly with existing patterns:

- Existing bbox/time API patterns (`api/routes/fires.py`, `api/fires/repo.py`)
- Existing “persist run metadata + files under `data/`” pattern (`ingest/spread_forecast.py`, `ingest/spread_repository.py`)
- Existing “public base URL vs internal service DNS” pattern for tile services (`docker-compose.yml`, `ui/components/map_view.py`)

**Sensible defaults (chosen where repo doesn’t constrain):**

- **Auth**: MVP assumes **no auth** (public AOIs). Schema includes nullable `owner_id` to enable future per-user AOIs without migrations churn.
- **Max AOI**: **50,000 km²** and **10,000 vertices** (hard limit). Above that: 413 or force async export mode + restrict tiles.
- **Tile zoom range**: z=**0–14** for points/polygons; allow overzoom in client if desired.
- **Simplification**: zoom-aware simplification in SQL (or via server config), targeting stable tile sizes (e.g., <= ~500KB per tile).
- **Export storage**: MVP uses **local filesystem** under `./data/exports/` (mirrors existing `./data/forecasts/...`), with a clean abstraction to later move to object storage.

---

## 2. Repo Reconnaissance (evidence-based)

### What was inspected (paths)

- **API entrypoints / routing**
  - `api/main.py` (router inclusion)
  - `api/routes/__init__.py`, `api/routes/internal.py`, `api/routes/fires.py`, `api/routes/forecast.py`
- **DB wiring / config patterns**
  - `api/config.py` (env vars; TiTiler base URL + mount mapping)
  - `api/db.py` (engine + sessions)
- **PostGIS + migrations strategy**
  - `api/migrations/versions/1ef68811334b_initial_schema_postgis.py` (enables PostGIS)
  - `api/migrations/versions/37962c109cd5_add_fire_detections_schema.py` (fire_detections geom POINT 4326, GiST index)
  - `api/migrations/versions/20251228_spread_forecasts.py` (spread_forecast_* tables; bbox POLYGON 4326; contours MULTIPOLYGON 4326)
  - `docs/data/db-migrations.md` (how migrations are done)
- **Existing spatial query patterns**
  - `api/fires/repo.py` (bbox + time queries; `geom && envelope` + `ST_Intersects`)
  - `api/forecast/repo.py` (bbox intersection queries; `ST_AsGeoJSON`)
- **Raster tile serving integration**
  - `docker-compose.yml` (TiTiler service; `./data:/data:ro` mount)
  - `api/routes/forecast.py` (TileJSON URL construction for TiTiler COG endpoint)
  - `ui/components/map_view.py` (resolves external TileJSON URL to internal service name in Docker)
- **File-based product persistence**
  - `ingest/spread_forecast.py` (writes COGs under `data/forecasts/...`)
  - `ingest/spread_repository.py` (DB inserts using SQL text + JSONB; SRID enforcement via `ST_SetSRID`)
- **Caching / queue presence**
  - `docker-compose.yml` has Redis, `docs/architecture.md` and `docs/WILDFIRE_NOWCAST_101.md` mention Redis for future caching/queue
  - No current backend usage of Redis was found (UI does some in-memory session caching in `ui/components/map_view.py`)
- **Exports**
  - No API export endpoints currently exist.
  - Some CLI export patterns exist (e.g., ML denoiser snapshot export to parquet; see `ml/denoiser/export_snapshot.py`), and spread forecast persistence to disk/DB (`ingest/spread_forecast.py`).

### Key repo constraints confirmed

- **Geometry SRID** is consistently **EPSG:4326** for stored vectors:
  - `fire_detections.geom` is `geometry(POINT, 4326)` (`api/migrations/versions/37962c109cd5_add_fire_detections_schema.py`)
  - `spread_forecast_runs.bbox` is `geometry(POLYGON, 4326)` and `spread_forecast_contours.geom` is `geometry(MULTIPOLYGON, 4326)` (`api/migrations/versions/20251228_spread_forecasts.py`)
  - Analysis grid helpers codify EPSG:4326 (`api/core/grid.py`)
- **Existing API filter style** uses bbox/time query params and returns JSON payloads:
  - `/fires` accepts bbox + `start_time`, `end_time`, optional `min_confidence`, `limit`, etc. (`api/routes/fires.py`)
  - `/forecast` accepts bbox + optional region; returns metadata + rasters + GeoJSON contours (`api/routes/forecast.py`)
- **PostGIS query style** is index-friendly:
  - `geom && ST_MakeEnvelope(...)` plus `ST_Intersects(...)` (`api/fires/repo.py`)
- **Rasters**: forecast products already written as **COG** and served via TiTiler:
  - TiTiler uses shared `./data` mount (`docker-compose.yml`)
  - API returns TiTiler TileJSON URL built from `storage_path` (`api/routes/forecast.py`)

---

## 3. Recommended Architecture (with diagrams described in text)

### 3.1 Architectural options & tradeoffs

#### MVT Serving Options

- **A) On-the-fly MVT from PostGIS, served by FastAPI**
  - **How**: FastAPI endpoint computes tile bbox for z/x/y, runs SQL using `ST_AsMVTGeom` + `ST_AsMVT`, returns `application/x-protobuf`.
  - **Pros**:
    - One service (API) to secure + rate limit
    - Full control of per-layer SQL and parameters
  - **Cons**:
    - Easy to get wrong (tile math, SRID transforms, performance footguns)
    - More Python app load; less specialized tooling (metadata endpoints, tilejson, inspection UI)

- **B) Pre-generated MBTiles (batch job), served via tile server or static hosting**
  - **How**: periodic job queries DB -> builds MBTiles (e.g., via Tippecanoe for vectors), serves from static host or tile server.
  - **Pros**:
    - Predictable performance; cheap serving (CDN-friendly)
    - Stable tiles for “snapshot” products
  - **Cons**:
    - Staleness + operational complexity (rebuild schedule, invalidation)
    - Poor fit for near-real-time layers (fires updating continuously)

- **C) Hybrid (recommended): on-the-fly + caching + optional warmup**
  - **How**: Serve on-the-fly from PostGIS (via dedicated tile server), add Redis/CDN caching, optionally pre-warm hot regions/zooms, and optionally pre-generate MBTiles for static layers.
  - **Pros**:
    - Works for both dynamic and static layers
    - Can incrementally scale (cache, CDN, warmup)
  - **Cons**:
    - More moving pieces (tile server + cache + API)

#### AOI Storage Options

- **PostGIS polygon (recommended)**
  - Store canonical geometry as `geometry(MULTIPOLYGON, 4326)` (or `POLYGON` with normalization to multi).
  - Store bbox separately (generated column) for fast pre-filter.
  - Validate and normalize on write.

- **Store bbox-only**
  - Cheaper but loses shape fidelity; not acceptable if AOIs need precise polygon filtering/exports.

- **Store both polygon + bbox**
  - Best of both; recommended for performance and simplicity in SQL.

#### Export Options

- **Sync export**
  - Best for small results; use streaming responses.
  - Must enforce limits to prevent runaway memory/time.

- **Async export jobs (recommended for large exports)**
  - Queue + worker; store results; status polling.
  - Enables expensive formats (MBTiles, Shapefile) and large AOIs safely.

### 3.2 Recommended default (for this repo)

**Default recommendation:** **C (Hybrid)** for MVT, **PostGIS** for AOI storage, and **sync+async** exports.

- Introduce a vector tile server (**Martin** or **pg_tileserv**) in Compose, pointing at the same PostGIS DB.
  - This aligns with how raster tiles are already served by TiTiler (separate service).
  - It also aligns with existing “public URL vs internal DNS” pattern in the UI for TiTiler (`ui/components/map_view.py`).
- Keep FastAPI as the “product API” (AOI CRUD, export orchestration, forecast run discovery).
- Add Redis usage for:
  - **Tile caching** (if proxying through FastAPI) and/or **export job queue**.

### 3.3 Diagrams (described in text)

**Diagram A (MVP request flow):**

- Client (UI/other tool)
  - calls `GET /aois`, `POST /aois` on **FastAPI**
  - requests vector tiles from **Vector Tile Server** (Martin/pg_tileserv): `GET /tiles/{layer}/{z}/{x}/{y}.pbf`
  - requests raster tiles from **TiTiler**: `GET /cog/.../tilejson.json?...`
- FastAPI talks to Postgres/PostGIS
- Tile servers talk to Postgres/PostGIS (vector) and `./data` (raster via TiTiler)

**Diagram B (Async export flow):**

- Client calls `POST /exports` (FastAPI) -> FastAPI writes `export_jobs` row and enqueues a job (Redis)
- Worker consumes job (Redis) -> queries PostGIS -> writes files under `./data/exports/<job_id>/...` -> updates DB row
- Client polls `GET /exports/{job_id}`; downloads via `GET /exports/{job_id}/download`

---

## 4. API Contracts (AOI CRUD, tiles, exports, status)

### 4.1 Conventions (aligned with existing API)

- **Base**: current API has unversioned routes (`/fires`, `/forecast`). For MVP we can keep this style. For longer-term, consider `/api/v1/...`.
- **Spatial inputs**:
  - bbox as `(min_lon, min_lat, max_lon, max_lat)` query params (as in `/fires`)
  - geometry as GeoJSON in request body for AOIs
- **Time**: ISO-8601 strings (FastAPI datetime parsing; UI uses `Z` suffix; see `ui/api_client.py::_isoformat`).
- **Errors**: standard FastAPI error payload, but we should standardize shape for clients.

### 4.2 AOI CRUD

#### `POST /aois`

- **Request (JSON)**:
  - `name: string`
  - `geometry: GeoJSON Polygon|MultiPolygon` (WGS84 lon/lat)
  - `tags?: object` (free-form JSON)
  - `description?: string`
  - `owner_id?: string|null` (MVP default null)

- **Validation**:
  - enforce SRID=4326
  - `ST_IsValid` or `ST_MakeValid` policy (see Gotchas)
  - limit by **max area** and **max vertices**

- **Response (JSON)**:
  - `id: string` (UUID)
  - `name`, `description`, `tags`
  - `geometry` (normalized GeoJSON)
  - `bbox` (GeoJSON Polygon or bbox tuple)
  - `area_km2: number`
  - `created_at`, `updated_at`

- **Errors**:
  - `400` invalid GeoJSON / self-intersection / invalid coordinates
  - `413` geometry too large (area/vertices)
  - `409` name conflict (if enforcing unique per owner)

#### `GET /aois`

- **Query params**:
  - `q?: string` (name search)
  - `bbox?: min_lon,min_lat,max_lon,max_lat` (optional filter)
  - `limit?: int` (default 50, max 200)
  - `cursor?: string` (optional; for pagination)
  - `owner_id?: string` (future)

- **Response**:
  - `{ "items": [AOI...], "next_cursor": string|null }`

#### `GET /aois/{aoi_id}`

- **Response**: AOI object (same shape as create)
- **Errors**: `404` if not found

#### `PATCH /aois/{aoi_id}`

- Update `name`, `description`, `tags`, and optionally `geometry` (re-validate if geometry changes).

#### `DELETE /aois/{aoi_id}`

- Soft delete is optional; MVP can hard delete.

### 4.3 Vector tile endpoints (MVT)

We should support a tile URL template compatible with common clients:

#### Option C recommended: vector tile server + optional FastAPI proxy

- **Vector tile server (public)**
  - `GET /mvt/{layer}/{z}/{x}/{y}.pbf`
  - Content-Type: `application/x-protobuf`
  - Optional: `?v=...` for cache-busting

- **FastAPI proxy (optional)**
  - `GET /tiles/{layer}/{z}/{x}/{y}.pbf`
  - Adds auth/rate limit, sets cache headers, optionally uses Redis cache

**Layers to expose (MVP)**:

- `fires` (from `fire_detections`)
- `forecast_contours` (from `spread_forecast_contours`, optionally filtered by `run_id` or `region_name`)
- `aois` (from `aois`)

**Tile query params (if supporting filtering at tile request time)**:

- `fires`:
  - `start_time`, `end_time` (optional; default “last 24h”)
  - `min_confidence` (optional)
  - `include_noise` (optional)
- `forecast_contours`:
  - `run_id` or (`region_name` + `forecast_reference_time` selection policy)
  - `horizon_hours` and `threshold` filters optional
- `aois`:
  - `owner_id` optional

### 4.4 Exports

#### Sync exports (small)

- `GET /aois/{aoi_id}/export?format=geojson`
  - Returns AOI geometry FeatureCollection (or a single Feature)

- `GET /fires/export?format=csv|geojson&min_lon...&start_time...`
  - Mirrors `/fires` bbox/time filters but returns file formats

- `GET /forecast/{run_id}/contours/export?format=geojson`

**Headers**:
- `Content-Disposition: attachment; filename=...`
- `Cache-Control` conservative (or `no-store` for auth’d contexts)

#### Async export jobs (large)

- `POST /exports`
  - Request:
    - `kind: "fires"|"forecast_contours"|"aoi_bundle"|...`
    - `aoi_id?: uuid` and/or `geometry?: GeoJSON` and/or bbox/time filters
    - `format: "geojson"|"csv"|"mbtiles"|"shp"|...`
  - Response:
    - `{ "job_id": uuid, "status": "queued" }`

- `GET /exports/{job_id}`
  - Response:
    - `{ "job_id", "status": "queued|running|succeeded|failed", "progress": ..., "result": { "download_url": ... } }`

- `GET /exports/{job_id}/download`
  - Streams the final artifact (or redirects to object storage if configured)

### 4.5 Auth assumptions

- MVP assumes **no auth**. Endpoints are public.
- Design should not block future auth:
  - include `owner_id` and plan for Postgres RLS or API-layer enforcement later
  - if using Martin/pg_tileserv, prefer function-based layers + DB role separation to avoid exposing all tables

---

## 5. Data Model / Schema changes (tables, indexes)

### 5.1 New table: `aois`

**Goal**: store user-defined polygons with metadata, SRID=4326.

Proposed columns:

- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `name TEXT NOT NULL`
- `description TEXT NULL`
- `tags JSONB NULL`
- `owner_id TEXT NULL` (future; MVP null)
- `geom geometry(MULTIPOLYGON, 4326) NOT NULL`
- `bbox geometry(POLYGON, 4326) NOT NULL` (stored or generated)
- `area_km2 DOUBLE PRECISION NOT NULL`
- `vertex_count INTEGER NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT now()`

Indexes:

- `GIST(geom)`
- `GIST(bbox)` (optional if bbox is stored; useful for quick map filtering)
- `BTREE(created_at)`
- `BTREE(owner_id, created_at)` (future)
- Optional uniqueness: `(owner_id, name)` unique where `owner_id IS NOT NULL`, and `(name)` unique where `owner_id IS NULL` (or skip for MVP).

Validation (enforced in API; optionally DB constraints/triggers):

- SRID is 4326
- `area_km2 <= MAX_AOI_AREA_KM2`
- `vertex_count <= MAX_AOI_VERTICES`
- `ST_IsValid(geom)` or normalize with `ST_MakeValid` (policy decision)

### 5.2 New table: `export_jobs` (for async exports)

Proposed columns:

- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `kind TEXT NOT NULL` (fires / forecast_contours / aoi_bundle / mbtiles / etc)
- `status TEXT NOT NULL` (queued/running/succeeded/failed)
- `request JSONB NOT NULL` (original request payload)
- `result JSONB NULL` (paths, sizes, checksums, download filenames)
- `error TEXT NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT now()`
- `started_at TIMESTAMPTZ NULL`
- `finished_at TIMESTAMPTZ NULL`

Indexes:

- `BTREE(status, created_at)`
- `BTREE(created_at)`

### 5.3 Optional: materialized views / helper tables (later)

- `fires_daily` partitions or BRIN indexes if `fire_detections` grows extremely large
- `forecast_contours_simplified(z)` precomputed geometry generalizations if contour complexity becomes a bottleneck

---

## 6. MVT Details (layer design, properties, zoom strategy, generalization)

### 6.1 Coordinate systems and transforms

- **Stored geometries** are EPSG:4326 (confirmed in migrations).
- **MVT tile coordinates** are Web Mercator (EPSG:3857) tile envelopes + a tile-local extent (typically 4096).
- For on-the-fly SQL, transform:
  - `geom_3857 = ST_Transform(geom, 3857)`
  - `tile_3857 = ST_TileEnvelope(z, x, y)` (if available) or app-side computed envelope in 3857

### 6.2 Layer design (MVP)

#### Layer: `fires`

- **Source**: `fire_detections`
- **Geometry**: points
- **Properties** (keep tight for tile size):
  - `id`, `acq_time`, `confidence`, `frp`, `sensor`, `source`, `denoised_score` (optional), `is_noise` (optional)
- **Zoom strategy**:
  - z 0–6: aggregate to reduce point explosion (cluster/grid aggregate)
  - z 7–14: serve raw points

Aggregation approach (SQL idea):
- compute a tile-grid cell id and return one feature per cell with `count`, `max_frp`, `latest_time`.

#### Layer: `forecast_contours`

- **Source**: `spread_forecast_contours` joined to `spread_forecast_runs` for filters
- **Geometry**: polygons/multipolygons
- **Properties**:
  - `run_id`, `horizon_hours`, `threshold`
- **Zoom strategy**:
  - z 0–8: simplified geometry
  - z 9–14: less simplification

#### Layer: `aois`

- **Source**: `aois`
- **Properties**:
  - `id`, `name` (maybe), `area_km2`
- **Zoom strategy**:
  - Always simplified at low zoom; show label from API (not tile) if needed.

### 6.3 SQL template for PostGIS MVT (for Option A, or for function-backed layers)

This is the canonical structure (pseudocode SQL):

```sql
WITH
tile AS (
  -- tile envelope in EPSG:3857
  SELECT ST_TileEnvelope(:z, :x, :y) AS tile_3857
),
src AS (
  SELECT
    id,
    confidence,
    frp,
    sensor,
    source,
    acq_time,
    -- convert geometry to 3857 and clip/simplify for tile
    ST_AsMVTGeom(
      ST_Transform(geom, 3857),
      (SELECT tile_3857 FROM tile),
      4096,  -- extent
      256,   -- buffer
      true   -- clip_geom
    ) AS geom
  FROM fire_detections
  WHERE acq_time BETWEEN :start_time AND :end_time
    AND ST_Intersects(ST_Transform(geom, 3857), (SELECT tile_3857 FROM tile))
    AND is_noise IS NOT TRUE  -- mirror existing default behavior
)
SELECT ST_AsMVT(src, 'fires', 4096, 'geom') FROM src;
```

Key knobs:

- **extent**: 4096 (default; good balance)
- **buffer**: 256 (reduces seam artifacts)
- **clip_geom**: true (keeps tile bounded)
- **simplification**:
  - can be controlled by pre-simplifying input geometry (e.g., `ST_SimplifyPreserveTopology`) based on zoom
  - or by increasing extent (reduces simplification, increases tile sizes)

### 6.4 Where to implement MVT in this repo

**Recommended**: add a dedicated vector tile server (Martin or pg_tileserv) and implement PostGIS **views/functions** to tightly control exposed layers.

**Alternative**: add a new FastAPI router `api/routes/tiles.py` and implement Option A directly in Python.

---

## 7. Export Pipeline (sync vs async, storage layout)

### 7.1 MVP sync exports

Support:

- **AOI GeoJSON**: quick, small.
- **Fires CSV**: common for analysts; points export.
- **Contours GeoJSON**: polygon export.

Implementation notes:

- Use streaming responses for CSV.
- For GeoJSON, avoid loading huge arrays into memory: paginate DB reads and stream if needed.
- Enforce:
  - max features (e.g., 200k points) for sync
  - max execution time (server-level)

### 7.2 Async export jobs (recommended for large/expensive)

When to use async:

- AOI area > `MAX_AOI_AREA_KM2`
- predicted features > `MAX_SYNC_FEATURES`
- formats like MBTiles/Shapefile that require packaging and tooling

Queue choice:

- **RQ** is a simple, Redis-backed Python queue (fits current stack: Redis already in Compose).
- Alternative: Celery (heavier).

Storage layout (MVP local):

- `data/exports/<job_id>/`
  - `request.json`
  - `result.<ext>` (e.g., `fires.csv`, `bundle.zip`, `tiles.mbtiles`)
  - `manifest.json` (counts, bbox, created_at)

Later migration:

- Abstract storage so `download_url` can be a signed URL from S3/GCS.

Formats:

- **Must**: GeoJSON, CSV
- **Optional**:
  - MBTiles: useful for offline map use; best generated async
  - Shapefile: for GIS compatibility; generate and zip
  - KML/GPX: optional; mostly for lightweight sharing

---

## 8. Gotchas & Performance Considerations

### 8.1 Geometry validity & normalization

- User-provided AOI polygons can be invalid (self-intersections, rings not closed, wrong winding).
- Recommended policy:
  - Reject obviously invalid geometry (`400`) unless you explicitly choose “auto-fix”.
  - If “auto-fix” is enabled: use `ST_MakeValid` and then normalize to MultiPolygon; return the normalized geometry to the client so behavior is visible.
- Always enforce:
  - lon/lat ranges
  - SRID=4326
  - maximum vertices and area limits

### 8.2 Dateline / antimeridian crossing

- EPSG:4326 polygons that cross ±180 can break bbox logic.
- MVP strategy: disallow dateline-crossing AOIs (return 400) OR normalize by splitting into MultiPolygon components that don’t cross.
- If supporting global AOIs later, invest in robust antimeridian handling early.

### 8.3 Index usage and query patterns

- Follow the repo’s existing pattern for bbox/time queries:
  - `geom && envelope` + `ST_Intersects` (GiST-friendly) (`api/fires/repo.py`)
- For AOI polygon intersections, prefer:
  - pre-filter by bbox: `geom && aoi_bbox`
  - then `ST_Intersects(geom, aoi_geom)`

### 8.4 MVT tile generation pitfalls

- **Tile math**: z/x/y is Web Mercator; stored data is 4326. Always transform properly.
- **Geometry simplification**:
  - polygons can explode tile size at low zoom; must simplify or pre-generalize
  - points can flood tiles; aggregate at low zoom
- **Buffer/seams**: set buffer to avoid “cut lines” at tile edges.
- **Huge tiles**: enforce max tile bytes and/or fallback to aggregated representations.

### 8.5 Caching

Cache layers:

- **HTTP caching**:
  - set `Cache-Control` for tiles (short TTL for fires; longer for contours/aois)
  - consider ETag on stable layers
- **Redis caching** (optional):
  - cache tile bytes for hot tiles
  - cache export job results/status lookups

### 8.6 Rate limits / abuse controls

- Especially important once AOI creation and exports exist:
  - limit AOI creations per minute
  - limit concurrent export jobs per client
  - cap tile request rate
- MVP can start with simple middleware-based rate limiting (IP-based).

### 8.7 Operational concerns

- Keep “public URL vs internal DNS” configurable:
  - TiTiler already needs this (UI rewrites `localhost:8080` -> `titiler:8000`)
  - do the same for vector tile server base URL
- Ensure DB connections are pooled and timeouts are set.

---

## 9. Implementation Task List (phases; ticket-style)

Each task includes **goal**, **steps**, **files affected**, and **acceptance criteria**. Tasks are sequenced to minimize cross-cutting churn and to be friendly for AI-agent execution.

### Phase 0 — Foundations (contracts, configs, wiring)

#### Task 0.1 — Add a `docs/geo/` planning hub
- **Goal**: establish a canonical place for geo-serving plans and contracts.
- **Steps**:
  - Ensure `docs/geo/` exists (this file lives here).
  - Add a short README later if needed.
- **Files affected**: `docs/geo/aoi_mvt_exports_plan.md`
- **Acceptance criteria**:
  - Plan doc is in repo and referenced by future PRs.

#### Task 0.2 — Add config settings for vector tile base URLs
- **Goal**: mirror existing TiTiler config pattern for vector tiles.
- **Steps**:
  - Extend `api/config.py` with:
    - `vector_tiles_public_base_url` (default `http://localhost:<port>`)
  - Extend UI runtime config (e.g., `ui/runtime_config.py`) if needed.
- **Files affected**: `api/config.py`, `ui/runtime_config.py` (and/or `ui/api_client.py`)
- **Acceptance criteria**:
  - App can build tile URLs without hardcoding hostnames.

### Phase 1 — AOI Storage + CRUD API

#### Task 1.1 — Create Alembic migration for `aois` table
- **Goal**: add PostGIS-backed AOI persistence.
- **Steps**:
  - Add migration under `api/migrations/versions/` creating `aois` table with SRID 4326.
  - Add GiST index on `geom` (and `bbox` if stored).
- **Files affected**: `api/migrations/versions/<new>_aois.py`
- **Acceptance criteria**:
  - `make migrate` creates table successfully.
  - AOI geometry stored as `geometry(MULTIPOLYGON, 4326)`.

#### Task 1.2 — Add AOI repo layer with safe SQL
- **Goal**: follow existing pattern (`api/fires/repo.py`, `api/forecast/repo.py`).
- **Steps**:
  - Create `api/aois/repo.py` with:
    - `create_aoi(...)`
    - `get_aoi(id)`
    - `list_aois(...)` (pagination)
    - `update_aoi(...)`
    - `delete_aoi(...)`
  - Use parameterized SQL with `sqlalchemy.text`.
  - Use PostGIS functions to compute:
    - bbox: `ST_Envelope(geom)`
    - area: `ST_Area(geom::geography) / 1e6`
    - vertex count: `ST_NPoints(geom)`
- **Files affected**: `api/aois/repo.py` (new), `api/db.py` (existing dependency)
- **Acceptance criteria**:
  - AOIs can be created/read/updated/deleted via repo functions with unit tests.

#### Task 1.3 — Add `api/routes/aois.py` with Pydantic models
- **Goal**: expose AOI CRUD endpoints consistent with existing route style.
- **Steps**:
  - Add router `aois_router = APIRouter(prefix="/aois", tags=["aois"])`.
  - Add request/response models (Pydantic) for AOI create/update.
  - Enforce limits:
    - max vertices
    - max area
    - optional “no dateline-crossing”
- **Files affected**: `api/routes/aois.py` (new), `api/routes/__init__.py`, `api/main.py`
- **Acceptance criteria**:
  - OpenAPI shows `/aois` endpoints.
  - Invalid geometry returns clear 400/413.

#### Task 1.4 — Add API tests for AOI CRUD
- **Goal**: ensure contracts and validation stay stable.
- **Steps**:
  - Add tests under `api/tests/` mirroring existing endpoint tests patterns.
- **Files affected**: `api/tests/test_aois_endpoint.py` (new)
- **Acceptance criteria**:
  - Tests pass in CI/local.

### Phase 2 — MVT Serving (choose one primary path)

#### Task 2.1 — Add a vector tile server service to Compose (recommended)
- **Goal**: serve `application/x-protobuf` MVT from PostGIS.
- **Steps**:
  - Add `martin` OR `pg_tileserv` service to `docker-compose.yml`.
  - Configure connection to `db` service.
  - Configure CORS and expose port (e.g., 7800).
- **Files affected**: `docker-compose.yml`, `infra/README.md`, `docs/GETTING_STARTED.md`
- **Acceptance criteria**:
  - `docker compose up` starts the tile server healthy.
  - `GET /.../{z}/{x}/{y}.pbf` returns bytes with correct content-type.

#### Task 2.2 — Define DB views/functions for published MVT layers
- **Goal**: avoid auto-exposing all tables; explicitly publish only intended layers.
- **Steps**:
  - Add SQL migration creating:
    - view/function for `fires` tiles with time filter default
    - view/function for `forecast_contours` tiles
    - view/function for `aois` tiles
  - Ensure functions accept z/x/y and optional filters.
- **Files affected**: `api/migrations/versions/<new>_mvt_functions.py`
- **Acceptance criteria**:
  - Tile server lists only the intended layers.
  - Tile payload sizes stay within a defined limit at low zoom (or are aggregated).

#### Task 2.3 — Optional FastAPI tile proxy (+ caching)
- **Goal**: unify auth/rate limiting/caching for tiles behind the API.
- **Steps**:
  - Add `api/routes/tiles.py` that proxies to the tile server.
  - Add simple Redis cache for tile bytes keyed by `{layer}:{z}:{x}:{y}:{filters_hash}`.
- **Files affected**: `api/routes/tiles.py`, `api/config.py`, `docker-compose.yml` (env wiring), `api/pyproject.toml` (deps)
- **Acceptance criteria**:
  - Client can hit `GET /tiles/...` and receive valid MVT.
  - Cache hit ratio measurable in logs (optional).

#### Task 2.4 — Add a minimal UI spike to validate MVT (optional, scoped)
- **Goal**: prove tiles render end-to-end.
- **Steps**:
  - Add a debug page in Streamlit that embeds a small MapLibre HTML snippet pointing at the MVT endpoint.
  - Keep Folium map unchanged for MVP if needed.
- **Files affected**: `ui/app.py` or `ui/components/map_view.py` (optional)
- **Acceptance criteria**:
  - A developer can see at least one vector layer render from MVT in the UI.

### Phase 3 — Sync Exports

#### Task 3.1 — Add `api/routes/exports.py` (sync endpoints)
- **Goal**: provide GeoJSON/CSV exports for bbox/time and AOIs.
- **Steps**:
  - Implement:
    - `GET /aois/{id}/export?format=geojson`
    - `GET /fires/export?format=csv|geojson` (bbox/time similar to `/fires`)
    - `GET /forecast/{run_id}/contours/export?format=geojson`
  - Ensure streaming for CSV.
  - Enforce `MAX_SYNC_FEATURES`.
- **Files affected**: `api/routes/exports.py` (new), `api/main.py`, `api/routes/__init__.py`
- **Acceptance criteria**:
  - Exports download successfully and match filters.
  - Large requests return 413 or instruct async flow.

#### Task 3.2 — Add export tests
- **Goal**: verify formats and bounds.
- **Steps**:
  - Add tests verifying CSV headers, row count limits, and GeoJSON validity.
- **Files affected**: `api/tests/test_exports_endpoint.py` (new)
- **Acceptance criteria**:
  - Tests pass; exported payloads parse correctly.

### Phase 4 — Async Exports (queue + worker)

#### Task 4.1 — Add `export_jobs` migration and repo
- **Goal**: durable tracking of export jobs.
- **Steps**:
  - Add migration creating `export_jobs`.
  - Add `api/exports/repo.py` to create/update job state.
- **Files affected**: `api/migrations/versions/<new>_export_jobs.py`, `api/exports/repo.py`
- **Acceptance criteria**:
  - Job lifecycle persisted correctly.

#### Task 4.2 — Add worker process (RQ) and queue wiring
- **Goal**: execute exports out-of-band.
- **Steps**:
  - Add deps to `api/pyproject.toml`: `redis` client + `rq`.
  - Add `worker` service in `docker-compose.yml` running `rq worker`.
  - Add `api/exports/worker.py` defining job functions.
- **Files affected**: `api/pyproject.toml`, `docker-compose.yml`, `api/exports/worker.py`, `infra/README.md`
- **Acceptance criteria**:
  - Jobs can be enqueued and processed locally with Compose.

#### Task 4.3 — Add async export endpoints
- **Goal**: `POST /exports`, `GET /exports/{id}`, `GET /exports/{id}/download`.
- **Steps**:
  - Implement endpoints and status responses.
  - Store results under `data/exports/<job_id>/...`.
- **Files affected**: `api/routes/exports.py` (extend), `api/config.py` (export storage root), `api/exports/repo.py`
- **Acceptance criteria**:
  - Large export request returns job id and completes successfully.

### Phase 5 — Hardening (rate limits, caching, observability)

#### Task 5.1 — Add consistent error responses + request validation
- **Goal**: predictable client behavior.
- **Steps**:
  - Add structured error model + exception handlers.
  - Ensure errors include `code`, `message`, `details`.
- **Files affected**: `api/main.py` (handlers), new `api/errors.py`
- **Acceptance criteria**:
  - Clients can rely on stable error payloads.

#### Task 5.2 — Add rate limiting (tiles + exports)
- **Goal**: protect the system from abuse and accidental overload.
- **Steps**:
  - Add middleware-based rate limiting (IP-based) using Redis.
- **Files affected**: `api/main.py`, `api/config.py`, deps
- **Acceptance criteria**:
  - Exceeding limits yields 429 with Retry-After.

---

## 10. QA Plan (tests + manual checks)

### 10.1 Automated tests (suggested)

- **AOI CRUD**
  - create with valid polygon
  - reject invalid geometry
  - enforce max vertices/area
- **Exports**
  - CSV schema and row count correctness
  - GeoJSON validity and intersection correctness
- **MVT**
  - smoke test endpoint returns non-empty bytes and correct content type
  - (optional) decode MVT with a test-only dependency to assert layer name + feature counts

### 10.2 Manual QA checklist

- Run stack:
  - `docker compose up --build`
  - `make migrate`
- AOI:
  - Create AOI via curl/Postman
  - List AOIs and fetch by id
  - Confirm bbox/area returned
- Tiles:
  - Request a tile at z/x/y where fires exist
  - Render with MapLibre in a simple HTML page (or the optional UI spike)
- Exports:
  - Download AOI GeoJSON
  - Export fires CSV for a small bbox/time window
  - Trigger async export for a large AOI and download when complete

---

## 11. Open Questions / Decisions Needed

1. **Vector tile server choice**: Martin vs pg_tileserv (both viable). Preference criteria:
   - deployment simplicity
   - function/view support
   - security model (DB roles/RLS)
2. **Dateline-crossing AOIs**: reject vs normalize/split (MVP recommendation: reject).
3. **Auth**: keep public vs add API keys now.
4. **Export storage**: local `./data` vs S3/GCS now (MVP recommendation: local).
5. **UI integration**: keep Folium + GeoJSON for now vs introduce MapLibre for MVT layers.

