-- Stage 3 — PGlite-compatible variant of 0002_stage3.sql.
--
-- Identical to the production DDL: `aoi_briefs` is a non-spatial table, so
-- there is nothing PostGIS-specific to strip. Kept as a separate file so the
-- PGlite harness can apply migrations in the same numbered order as Neon.

CREATE TABLE IF NOT EXISTS "aoi_briefs" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "aoi_id" UUID NOT NULL REFERENCES "aois"("id") ON DELETE CASCADE,
    "event_id" UUID NOT NULL REFERENCES "aoi_events"("id") ON DELETE CASCADE,
    "schema_version" INTEGER NOT NULL DEFAULT 1,
    "model" TEXT NOT NULL,
    "prompt_version" TEXT NOT NULL DEFAULT 'v1',
    "gate_reason" TEXT NOT NULL,
    "payload" JSONB NOT NULL,
    "rendered_markdown" TEXT NOT NULL,
    "cost_usd_est" NUMERIC(10,6),
    "latency_ms" INTEGER,
    "share_token" TEXT,
    "share_expires_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS "aoi_briefs_event_uniq"
    ON "aoi_briefs" ("event_id");

CREATE INDEX IF NOT EXISTS "aoi_briefs_aoi_recent"
    ON "aoi_briefs" ("aoi_id", "created_at" DESC);

CREATE UNIQUE INDEX IF NOT EXISTS "aoi_briefs_share_token_uniq"
    ON "aoi_briefs" ("share_token")
    WHERE "share_token" IS NOT NULL;

ALTER TABLE "aoi_events"
    ADD COLUMN IF NOT EXISTS "last_brief_at" TIMESTAMPTZ;

ALTER TABLE "job_runs"
    ADD COLUMN IF NOT EXISTS "briefs_generated" INTEGER NOT NULL DEFAULT 0;
