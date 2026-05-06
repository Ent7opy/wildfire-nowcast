-- Stage 2 — PGlite-compatible variant of 0001_stage2.sql.
--
-- PGlite has no PostGIS, so geometry columns become TEXT (GeoJSON strings) and
-- GIST indexes are dropped. Spatial integration tests against a real
-- testcontainer (postgis/postgis:16-3.5) are the source of truth for ST_*
-- behaviour; this file only covers the non-spatial domain logic
-- (CSV parsing, dedupe-hash, route auth, job-runs bookkeeping).

CREATE TABLE IF NOT EXISTS "firms_detections" (
    "id" BIGSERIAL PRIMARY KEY,
    "source" TEXT NOT NULL,
    "detected_at" TIMESTAMPTZ NOT NULL,
    "geom" TEXT NOT NULL,            -- GeoJSON Point in PGlite
    "lat" DOUBLE PRECISION NOT NULL,
    "lon" DOUBLE PRECISION NOT NULL,
    "frp_mw" REAL,
    "confidence" TEXT,
    "daynight" TEXT,
    "acq_date" TEXT NOT NULL,
    "acq_time" TEXT NOT NULL,
    "bright_ti4" REAL,
    "bright_ti5" REAL,
    "scan" REAL,
    "track" REAL,
    "version" TEXT,
    "is_industrial_static" BOOLEAN,
    "bucket" TEXT NOT NULL,
    "inserted_at" TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS "firms_detections_dedupe"
    ON "firms_detections" ("source", "acq_date", "acq_time", "lat", "lon");

CREATE INDEX IF NOT EXISTS "firms_detections_bucket_detected"
    ON "firms_detections" ("bucket", "detected_at" DESC);

CREATE TABLE IF NOT EXISTS "aoi_events" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "aoi_id" UUID NOT NULL REFERENCES "aois"("id") ON DELETE CASCADE,
    "first_seen_at" TIMESTAMPTZ NOT NULL,
    "last_seen_at" TIMESTAMPTZ NOT NULL,
    "nearest_distance_km" REAL NOT NULL,
    "detection_count" INTEGER NOT NULL DEFAULT 1,
    "peak_frp_mw" REAL,
    "dedupe_hash" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'new',
    "closed_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS "aoi_events_aoi_dedupe_uniq"
    ON "aoi_events" ("aoi_id", "dedupe_hash");

CREATE INDEX IF NOT EXISTS "aoi_events_aoi_recent"
    ON "aoi_events" ("aoi_id", "last_seen_at" DESC);

CREATE TABLE IF NOT EXISTS "industrial_mask_static" (
    "id" SERIAL PRIMARY KEY,
    "kind" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "geom" TEXT NOT NULL,            -- GeoJSON Polygon in PGlite
    "source_url" TEXT,
    "loaded_at" TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS "industrial_mask_static_kind_name_uniq"
    ON "industrial_mask_static" ("kind", "name");

CREATE TABLE IF NOT EXISTS "job_runs" (
    "id" BIGSERIAL PRIMARY KEY,
    "job_name" TEXT NOT NULL,
    "bucket" TEXT,
    "started_at" TIMESTAMPTZ NOT NULL,
    "finished_at" TIMESTAMPTZ,
    "status" TEXT NOT NULL,
    "firms_request_count" INTEGER NOT NULL DEFAULT 0,
    "detections_inserted" INTEGER NOT NULL DEFAULT 0,
    "events_created" INTEGER NOT NULL DEFAULT 0,
    "error" TEXT
);

CREATE INDEX IF NOT EXISTS "job_runs_recent"
    ON "job_runs" ("job_name", "started_at" DESC);
