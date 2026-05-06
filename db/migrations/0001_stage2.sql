-- Stage 2 — A' pivot: FIRMS poll, AOI matcher, cron observability.
--
-- Authoritative source: docs/pivot-architecture.md §3 (tables 3.4 firms_detections,
-- 3.5 aoi_events, 3.8 industrial_mask_static, 3.9 job_runs).
--
-- Idempotent: every CREATE statement uses IF NOT EXISTS so the migrate script
-- can be re-applied safely. The static industrial mask seed runs as a separate
-- step (`scripts/db-seed-industrial-mask.ts`) so its data lives outside this
-- DDL and can evolve without a new migration.

-- 4. firms_detections -------------------------------------------------------
CREATE TABLE IF NOT EXISTS "firms_detections" (
    "id" BIGSERIAL PRIMARY KEY,
    "source" TEXT NOT NULL,
    "detected_at" TIMESTAMPTZ NOT NULL,
    "geom" geometry(Point, 4326) NOT NULL,
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

-- Idempotent dedupe across overlapping bucket fetches: same satellite pixel at
-- the same instant => same row. Lat/lon doubles are the natural key (geometry
-- equality on Points is brittle across drivers).
CREATE UNIQUE INDEX IF NOT EXISTS "firms_detections_dedupe"
    ON "firms_detections" ("source", "acq_date", "acq_time", "lat", "lon");

CREATE INDEX IF NOT EXISTS "firms_detections_geom_gix"
    ON "firms_detections" USING GIST ("geom");

CREATE INDEX IF NOT EXISTS "firms_detections_bucket_detected"
    ON "firms_detections" ("bucket", "detected_at" DESC);

CREATE INDEX IF NOT EXISTS "firms_detections_detected"
    ON "firms_detections" ("detected_at");

-- 5. aoi_events -------------------------------------------------------------
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

CREATE INDEX IF NOT EXISTS "aoi_events_status_recent"
    ON "aoi_events" ("status", "created_at" DESC);

-- 8. industrial_mask_static -------------------------------------------------
CREATE TABLE IF NOT EXISTS "industrial_mask_static" (
    "id" SERIAL PRIMARY KEY,
    "kind" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "geom" geometry(Polygon, 4326) NOT NULL,
    "source_url" TEXT,
    "loaded_at" TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS "industrial_mask_static_geom_gix"
    ON "industrial_mask_static" USING GIST ("geom");

CREATE UNIQUE INDEX IF NOT EXISTS "industrial_mask_static_kind_name_uniq"
    ON "industrial_mask_static" ("kind", "name");

-- 9. job_runs ---------------------------------------------------------------
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
