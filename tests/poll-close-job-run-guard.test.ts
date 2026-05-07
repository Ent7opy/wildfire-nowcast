/**
 * Reviewer-flagged hardening on Stage 8 (#411): closeJobRun must COALESCE-
 * protect retry_pending the same way it protects outcome, so a parent row's
 * no-arg close cannot clobber a child row's retry signal if the freshness
 * filter is ever loosened from `bucket = aoi.region_bucket`.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import { _setTestDb, type AppDb } from "@/lib/db/client";
import { _closeJobRunForTest } from "@/app/api/aoi/poll/route";
import type { PGlite } from "@electric-sql/pglite";

async function insertJobRun(
  db: AppDb,
  args: { bucket: string | null; retryPending: boolean; outcome: string | null },
): Promise<string> {
  const res = (await db.execute(sql`
    INSERT INTO "job_runs" ("job_name", "bucket", "started_at", "status", "outcome", "retry_pending")
    VALUES ('firms-poll', ${args.bucket}, NOW(), 'ok', ${args.outcome}, ${args.retryPending})
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const rows = (res.rows ?? (res as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  return rows[0].id;
}

async function readRun(
  db: AppDb,
  id: string,
): Promise<{ outcome: string | null; retry_pending: boolean; status: string }> {
  const res = (await db.execute(sql`
    SELECT status, outcome, retry_pending FROM job_runs WHERE id = ${id}
  `)) as unknown as {
    rows?: Array<{ status: string; outcome: string | null; retry_pending: boolean }>;
  };
  const rows = (res.rows ??
    (res as unknown as Array<{ status: string; outcome: string | null; retry_pending: boolean }>)) as Array<{
    status: string;
    outcome: string | null;
    retry_pending: boolean;
  }>;
  return rows[0];
}

describe("closeJobRun — retry_pending COALESCE guard", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    _setTestDb(db);
  });

  afterEach(async () => {
    _setTestDb(null);
    await pglite.close();
  });

  it("no-arg close on a row with retry_pending=true preserves it (does not clobber to false)", async () => {
    const id = await insertJobRun(db, {
      bucket: "5x5:W120_N35",
      retryPending: true,
      outcome: "rate_limited",
    });

    await _closeJobRunForTest(db, id, {
      status: "ok",
      error: null,
      finishedAt: new Date(),
    });

    const row = await readRun(db, id);
    expect(row.retry_pending).toBe(true);
    // outcome guard already exists; assert here to anchor the parallel.
    expect(row.outcome).toBe("rate_limited");
    expect(row.status).toBe("ok");
  });

  it("explicit retryPending: false on a row with retry_pending=true DOES set it to false", async () => {
    const id = await insertJobRun(db, {
      bucket: "5x5:W120_N35",
      retryPending: true,
      outcome: "rate_limited",
    });

    await _closeJobRunForTest(db, id, {
      status: "ok",
      error: null,
      finishedAt: new Date(),
      retryPending: false,
      outcome: "success",
    });

    const row = await readRun(db, id);
    expect(row.retry_pending).toBe(false);
    expect(row.outcome).toBe("success");
  });

  it("explicit retryPending: true on a row with retry_pending=false sets it to true", async () => {
    const id = await insertJobRun(db, {
      bucket: "5x5:W120_N35",
      retryPending: false,
      outcome: null,
    });

    await _closeJobRunForTest(db, id, {
      status: "partial",
      error: null,
      finishedAt: new Date(),
      retryPending: true,
      outcome: "network_error",
    });

    const row = await readRun(db, id);
    expect(row.retry_pending).toBe(true);
    expect(row.outcome).toBe("network_error");
  });
});
