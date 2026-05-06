-- PGlite-compatible variant of 0000_init.sql for unit tests.
--
-- PGlite does not bundle PostGIS. We model `geometry(...)` columns as TEXT
-- holding GeoJSON strings, and skip the GIST indexes (PGlite supports btree
-- only). This is intentionally a *test substitute*, not a production schema —
-- the production migration in 0000_init.sql remains authoritative.
--
-- Repository code in db/repositories/aoi.ts uses two code paths gated by the
-- `usePostGIS` flag on the db client (see lib/db/client.ts):
--   * production (Neon)  → ST_GeomFromGeoJSON / ST_AsGeoJSON / ST_Area /
--     ST_Envelope / ST_Centroid in SQL
--   * test (PGlite)      → application-side computation in lib/geo/*.ts;
--     GeoJSON stored verbatim in the TEXT columns.
--
-- The repository contract is identical from the route handler's point of
-- view: GeoJSON in, GeoJSON out.

CREATE TABLE IF NOT EXISTS "users" (
    "id" TEXT PRIMARY KEY,
    "email" TEXT NOT NULL,
    "display_name" TEXT,
    "gemini_api_key_enc" BYTEA,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT now(),
    "deleted_at" TIMESTAMPTZ
);

INSERT INTO "users" ("id", "email", "display_name")
VALUES ('stub-user-1', 'stub@earthtools.local', 'Stub User')
ON CONFLICT ("id") DO NOTHING;

CREATE TABLE IF NOT EXISTS "aois" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "user_id" TEXT NOT NULL REFERENCES "users"("id") ON DELETE CASCADE,
    "name" TEXT NOT NULL,
    "polygon" TEXT NOT NULL,        -- GeoJSON MultiPolygon
    "bbox" TEXT NOT NULL,           -- GeoJSON Polygon (envelope)
    "centroid" TEXT NOT NULL,       -- GeoJSON Point
    "region_bucket" TEXT NOT NULL,
    "area_ha" REAL NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT now(),
    "archived_at" TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS "aois_region_bucket_active"
    ON "aois" ("region_bucket")
    WHERE "archived_at" IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS "aois_user_name_active_uniq"
    ON "aois" ("user_id", "name")
    WHERE "archived_at" IS NULL;

CREATE TABLE IF NOT EXISTS "aoi_rules" (
    "aoi_id" UUID PRIMARY KEY REFERENCES "aois"("id") ON DELETE CASCADE,
    "distance_buffer_km" REAL NOT NULL DEFAULT 25,
    "min_confidence" TEXT NOT NULL DEFAULT 'nominal',
    "min_frp_mw" REAL NOT NULL DEFAULT 5,
    "quiet_hours" JSONB,
    "paused_until" TIMESTAMPTZ,
    "notify_channels" JSONB NOT NULL DEFAULT '[]'::jsonb,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT now()
);
