-- Stage 3 — A' pivot: LLM brief generation.
--
-- Adds the `aoi_briefs` table (one row per generated brief) plus a
-- `last_brief_at` column on `aoi_events` so the gate can cheaply reject
-- already-briefed events without joining to `aoi_briefs`.
--
-- Authoritative sources:
--   - docs/SPEC-A-prime-v1.md §LLM brief format (schema fields)
--   - docs/SPEC-A-prime-v1.md §Data model (aoi_briefs columns)
--   - docs/SPEC-A-prime-v1.md §Flow 6 (gate semantics)

CREATE TABLE IF NOT EXISTS "aoi_briefs" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "aoi_id" UUID NOT NULL REFERENCES "aois"("id") ON DELETE CASCADE,
    "event_id" UUID NOT NULL REFERENCES "aoi_events"("id") ON DELETE CASCADE,
    "schema_version" INTEGER NOT NULL DEFAULT 1,
    "model" TEXT NOT NULL,
    "gate_reason" TEXT NOT NULL,
    "payload" JSONB NOT NULL,
    "rendered_markdown" TEXT NOT NULL,
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
