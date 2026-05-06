-- Stage 4 — PGlite-compatible variant of 0003_stage4.sql.
--
-- Identical to the production DDL: `notifications_log` is a non-spatial
-- table, so there is nothing PostGIS-specific to strip. Kept as a separate
-- file so the PGlite harness can apply migrations in the same numbered
-- order as Neon.

CREATE TABLE IF NOT EXISTS "notifications_log" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "aoi_id" UUID NOT NULL REFERENCES "aois"("id") ON DELETE CASCADE,
    "brief_id" UUID NOT NULL REFERENCES "aoi_briefs"("id") ON DELETE CASCADE,
    "channel" TEXT NOT NULL,
    "target" TEXT NOT NULL,
    "target_hash" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "provider_message_id" TEXT,
    "error" TEXT,
    "skip_reason" TEXT,
    "sent_at" TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS "notifications_log_brief_channel_target_uniq"
    ON "notifications_log" ("brief_id", "channel", "target_hash")
    WHERE "status" IN ('sent', 'skipped');

CREATE INDEX IF NOT EXISTS "notifications_log_aoi_target_recent"
    ON "notifications_log" ("aoi_id", "target_hash", "sent_at" DESC);

ALTER TABLE "aoi_briefs"
    ADD COLUMN IF NOT EXISTS "last_notified_at" TIMESTAMPTZ;

ALTER TABLE "job_runs"
    ADD COLUMN IF NOT EXISTS "notifications_sent" INTEGER NOT NULL DEFAULT 0;
