-- Stage 9 — PGlite-compatible variant of 0007_stage9.sql.
--
-- Identical to production: pure column add + a NOT NULL drop + an index.

ALTER TABLE "notifications_log"
    ADD COLUMN IF NOT EXISTS "kind" TEXT NOT NULL DEFAULT 'brief';

ALTER TABLE "notifications_log"
    ALTER COLUMN "brief_id" DROP NOT NULL;

CREATE INDEX IF NOT EXISTS "notifications_log_kind_created_at_idx"
    ON "notifications_log" ("kind", "sent_at" DESC);
