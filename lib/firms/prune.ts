/**
 * Stage 7 — 14-day retention sweep for `firms_detections`.
 *
 * `docs/pivot-architecture.md` §6 R5: free-tier Postgres has a hard storage
 * cap; FIRMS is the only growing table. The matcher and brief generator only
 * read detections from the current poll's window (matcher: `inserted_at >=
 * pollStart`; brief generator: `detected_at` between event first/last seen),
 * both of which are well inside 14 days, so dropping older rows is safe.
 */
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";
import { decodeRowCount } from "@/lib/db/decode-rows";

export type PruneArgs = {
  /** Override clock for tests. Defaults to `new Date()`. */
  now?: Date;
  /** Defaults to 14 per the architecture doc. */
  retentionDays?: number;
};

export async function pruneOldDetections(
  db: AppDb,
  args: PruneArgs = {},
): Promise<number> {
  const now = args.now ?? new Date();
  const days = args.retentionDays ?? 14;
  const cutoff = new Date(now.getTime() - days * 86400_000);
  const result = await db.execute(sql`
    DELETE FROM "firms_detections"
    WHERE "detected_at" < ${cutoff.toISOString()}::timestamptz
  `);
  return decodeRowCount(result);
}
