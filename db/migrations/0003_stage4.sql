-- Stage 4 — A' pivot: notification dispatch (Resend).
--
-- Adds the `notifications_log` table (one row per send attempt, idempotent
-- per (brief, channel, target) when the prior outcome was `sent`/`skipped`),
-- plus `aoi_briefs.last_notified_at` and `job_runs.notifications_sent` for
-- observability.
--
-- Authoritative source: docs/pivot-architecture.md §3.7 `notifications_log`.
-- Spec: docs/SPEC-A-prime-v1.md §Data model `notifications` row shape.

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

-- Idempotency: re-running the dispatcher for the same (brief, channel, target)
-- after a successful send or a deliberate skip is a no-op. Failed rows are
-- excluded so the next poll naturally retries.
CREATE UNIQUE INDEX IF NOT EXISTS "notifications_log_brief_channel_target_uniq"
    ON "notifications_log" ("brief_id", "channel", "target_hash")
    WHERE "status" IN ('sent', 'skipped');

-- Rate-limit lookup index per spec §3.7.
CREATE INDEX IF NOT EXISTS "notifications_log_aoi_target_recent"
    ON "notifications_log" ("aoi_id", "target_hash", "sent_at" DESC);

ALTER TABLE "aoi_briefs"
    ADD COLUMN IF NOT EXISTS "last_notified_at" TIMESTAMPTZ;

ALTER TABLE "job_runs"
    ADD COLUMN IF NOT EXISTS "notifications_sent" INTEGER NOT NULL DEFAULT 0;
