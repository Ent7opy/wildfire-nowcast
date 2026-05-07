-- Stage 9 — A' pivot: watch-confirmed email + first-AOI backfill.
--
-- Adds:
--   * notifications_log.kind — discriminator so watch-confirmed rows can be
--     distinguished from brief-dispatch rows in launch-week queries. Defaults
--     to 'brief'; existing rows are backfilled by the column default.
--   * Drops NOT NULL on notifications_log.brief_id — a watch-confirmed row
--     has no brief, so brief_id is NULL for kind='watch_confirmed'.
--   * Index on (kind, created_at DESC) — supports filtering by row kind in
--     launch-readiness queries without scanning the whole log.
--
-- Authoritative source: pm/briefs/23-stage9-watch-confirmed-and-first-poll-backfill.md.

ALTER TABLE "notifications_log"
    ADD COLUMN IF NOT EXISTS "kind" TEXT NOT NULL DEFAULT 'brief';

ALTER TABLE "notifications_log"
    ALTER COLUMN "brief_id" DROP NOT NULL;

CREATE INDEX IF NOT EXISTS "notifications_log_kind_created_at_idx"
    ON "notifications_log" ("kind", "sent_at" DESC);
