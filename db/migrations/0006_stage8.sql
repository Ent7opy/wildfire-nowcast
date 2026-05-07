-- Stage 8 — A' pivot: authority-perimeter pre-fetch + per-bucket freshness.
--
-- Adds:
--   * job_runs.outcome — user-facing taxonomy (success | rate_limited |
--     network_error | timeout | partial). Operator-facing `status`
--     (ok|partial|error|running) stays for compatibility.
--   * job_runs.retry_pending — signal (not a promise) that the next cron
--     tick will retry this bucket. Surfaced in the freshness banner copy.
--   * Partial index on (bucket, started_at DESC) WHERE bucket IS NOT NULL —
--     supports the per-AOI freshness lookup without scanning parent runs.
--
-- Authoritative source: pm/briefs/22-stage8-authority-perimeter-and-freshness.md.

ALTER TABLE "job_runs"
    ADD COLUMN IF NOT EXISTS "outcome" TEXT,
    ADD COLUMN IF NOT EXISTS "retry_pending" BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS "job_runs_bucket_started_at_idx"
    ON "job_runs" ("bucket", "started_at" DESC)
    WHERE "bucket" IS NOT NULL;
