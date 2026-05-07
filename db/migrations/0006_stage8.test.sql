-- Stage 8 — PGlite-compatible variant of 0006_stage8.sql.
--
-- Identical to production: pure column additions + a partial index. PGlite
-- supports partial indexes (verified in the migration test); if a future
-- PGlite version regresses, drop the WHERE clause — the index is small.

ALTER TABLE "job_runs"
    ADD COLUMN IF NOT EXISTS "outcome" TEXT,
    ADD COLUMN IF NOT EXISTS "retry_pending" BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS "job_runs_bucket_started_at_idx"
    ON "job_runs" ("bucket", "started_at" DESC)
    WHERE "bucket" IS NOT NULL;
