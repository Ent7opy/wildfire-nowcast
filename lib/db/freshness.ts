/**
 * Stage 8 — per-AOI freshness lookup for the dashboard.
 *
 * Joins `aois` → `job_runs` on bucket and returns the most recent completed
 * (i.e. not 'running') run for that bucket. The AOI page uses this to render
 * the freshness banner — "Last polled 47 minutes ago" or a yellow degraded
 * variant.
 *
 * Two-backend repository pattern: pure SQL, runs identically on Neon and
 * PGlite. Uses the partial index `job_runs_bucket_started_at_idx` for an
 * O(log n) lookup.
 */
import { sql } from "drizzle-orm";
import type { AppDb } from "./client";
import type { FreshnessOutcome } from "@/lib/firms/freshness";

export type AoiFreshness = {
  bucket: string;
  lastPolledAt: Date | null;
  outcome: FreshnessOutcome | null;
  retryPending: boolean;
  /** True when last successful poll is older than the staleness window. */
  isStale: boolean;
};

const STALE_AFTER_MS = 30 * 60 * 1000; // 30 min

export async function getAoiFreshness(
  db: AppDb,
  args: { aoiId: string; userId: string; now?: Date },
): Promise<AoiFreshness | null> {
  const result = (await db.execute(sql`
    SELECT
      a."region_bucket"     AS bucket,
      j."started_at"        AS started_at,
      j."finished_at"       AS finished_at,
      j."outcome"           AS outcome,
      j."retry_pending"     AS retry_pending
    FROM "aois" a
    LEFT JOIN LATERAL (
      SELECT "started_at", "finished_at", "outcome", "retry_pending"
      FROM "job_runs"
      WHERE "bucket" = a."region_bucket"
        AND "status" <> 'running'
      ORDER BY "started_at" DESC
      LIMIT 1
    ) j ON TRUE
    WHERE a."id" = ${args.aoiId}
      AND a."user_id" = ${args.userId}
    LIMIT 1
  `)) as unknown as {
    rows?: Array<{
      bucket: string;
      started_at: Date | string | null;
      finished_at: Date | string | null;
      outcome: string | null;
      retry_pending: boolean | null;
    }>;
  };
  const rows = (result.rows ??
    (result as unknown as Array<{
      bucket: string;
      started_at: Date | string | null;
      finished_at: Date | string | null;
      outcome: string | null;
      retry_pending: boolean | null;
    }>)) as Array<{
    bucket: string;
    started_at: Date | string | null;
    finished_at: Date | string | null;
    outcome: string | null;
    retry_pending: boolean | null;
  }>;
  const row = rows[0];
  if (!row) return null;

  const finished = toDate(row.finished_at) ?? toDate(row.started_at);
  const outcome = (row.outcome as FreshnessOutcome | null) ?? null;
  const retryPending = Boolean(row.retry_pending);
  const now = args.now ?? new Date();
  const isStale =
    outcome === "success" &&
    finished != null &&
    now.getTime() - finished.getTime() > STALE_AFTER_MS;
  return {
    bucket: row.bucket,
    lastPolledAt: finished,
    outcome,
    retryPending,
    isStale,
  };
}

function toDate(v: unknown): Date | null {
  if (v == null) return null;
  if (v instanceof Date) return v;
  if (typeof v === "string" || typeof v === "number") {
    const d = new Date(v);
    return Number.isFinite(d.getTime()) ? d : null;
  }
  return null;
}
