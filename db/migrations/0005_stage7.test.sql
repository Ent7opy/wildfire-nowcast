-- Stage 7 — PGlite-compatible variant of 0005_stage7.sql.
--
-- Identical to the production DDL: all three additions are non-spatial.

CREATE TABLE IF NOT EXISTS "notify_action_tokens" (
    "token" TEXT PRIMARY KEY,
    "aoi_id" UUID NOT NULL REFERENCES "aois"("id") ON DELETE CASCADE,
    "brief_id" UUID REFERENCES "aoi_briefs"("id") ON DELETE SET NULL,
    "action" TEXT NOT NULL,
    "channel" TEXT NOT NULL,
    "target" TEXT NOT NULL,
    "expires_at" TIMESTAMPTZ NOT NULL,
    "redeemed_at" TIMESTAMPTZ,
    "redeemed_value" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS "notify_action_tokens_aoi_action"
    ON "notify_action_tokens" ("aoi_id", "action");

CREATE INDEX IF NOT EXISTS "notify_action_tokens_brief"
    ON "notify_action_tokens" ("brief_id");

CREATE TABLE IF NOT EXISTS "brief_feedback" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "brief_id" UUID NOT NULL REFERENCES "aoi_briefs"("id") ON DELETE CASCADE,
    "helpful" BOOLEAN NOT NULL,
    "recipient_token" TEXT NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS "brief_feedback_brief_token_uniq"
    ON "brief_feedback" ("brief_id", "recipient_token");

ALTER TABLE "job_runs"
    ADD COLUMN IF NOT EXISTS "detections_pruned" INTEGER;
