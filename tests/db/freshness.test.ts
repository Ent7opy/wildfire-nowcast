/**
 * Stage 8 — `getAoiFreshness` PGlite tests.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import { getAoiFreshness } from "@/lib/db/freshness";
import type { PGlite } from "@electric-sql/pglite";

async function seedAoi(
  db: AppDb,
  args: { userId: string; bucket: string; name?: string },
): Promise<string> {
  const polygon = JSON.stringify({
    type: "Polygon",
    coordinates: [
      [
        [-122.7, 38.4],
        [-122.6, 38.4],
        [-122.6, 38.5],
        [-122.7, 38.5],
        [-122.7, 38.4],
      ],
    ],
  });
  await db.execute(sql`
    INSERT INTO "users" ("id", "email")
    VALUES (${args.userId}, ${args.userId + "@example.org"})
    ON CONFLICT ("id") DO NOTHING
  `);
  const res = (await db.execute(sql`
    INSERT INTO "aois" (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
    VALUES (
      ${args.userId},
      ${args.name ?? "Preserve"},
      ${polygon},
      ${polygon},
      ${JSON.stringify({ type: "Point", coordinates: [-122.65, 38.45] })},
      ${args.bucket},
      100
    )
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const rows = (res.rows ?? (res as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  return rows[0].id;
}

async function insertJobRun(
  db: AppDb,
  args: {
    bucket: string | null;
    startedAt: Date;
    finishedAt?: Date | null;
    status: string;
    outcome?: string | null;
    retryPending?: boolean;
  },
): Promise<void> {
  await db.execute(sql`
    INSERT INTO "job_runs" ("job_name", "bucket", "started_at", "finished_at", "status", "outcome", "retry_pending")
    VALUES (
      'firms-poll',
      ${args.bucket},
      ${args.startedAt.toISOString()},
      ${args.finishedAt ? args.finishedAt.toISOString() : null},
      ${args.status},
      ${args.outcome ?? null},
      ${args.retryPending ?? false}
    )
  `);
}

describe("getAoiFreshness", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("returns null when no completed run exists for the bucket", async () => {
    const aoiId = await seedAoi(db, { userId: "user_2fresh1", bucket: "5x5:W125_N40" });
    // A running row should be ignored.
    await insertJobRun(db, {
      bucket: "5x5:W125_N40",
      startedAt: new Date("2026-05-07T12:00:00Z"),
      status: "running",
    });
    const r = await getAoiFreshness(db, { aoiId, userId: "user_2fresh1" });
    expect(r).not.toBeNull();
    expect(r!.lastPolledAt).toBeNull();
  });

  it("picks the most recent completed run for the bucket", async () => {
    const aoiId = await seedAoi(db, { userId: "user_2fresh2", bucket: "5x5:W125_N40" });
    await insertJobRun(db, {
      bucket: "5x5:W125_N40",
      startedAt: new Date("2026-05-07T11:00:00Z"),
      finishedAt: new Date("2026-05-07T11:01:00Z"),
      status: "ok",
      outcome: "success",
    });
    await insertJobRun(db, {
      bucket: "5x5:W125_N40",
      startedAt: new Date("2026-05-07T12:00:00Z"),
      finishedAt: new Date("2026-05-07T12:01:00Z"),
      status: "error",
      outcome: "rate_limited",
      retryPending: true,
    });
    const r = await getAoiFreshness(db, {
      aoiId,
      userId: "user_2fresh2",
      now: new Date("2026-05-07T12:30:00Z"),
    });
    expect(r!.outcome).toBe("rate_limited");
    expect(r!.retryPending).toBe(true);
    expect(r!.isStale).toBe(false);
  });

  it("computes isStale when last success older than 30 min", async () => {
    const aoiId = await seedAoi(db, { userId: "user_2fresh3", bucket: "5x5:W125_N40" });
    await insertJobRun(db, {
      bucket: "5x5:W125_N40",
      startedAt: new Date("2026-05-07T11:00:00Z"),
      finishedAt: new Date("2026-05-07T11:00:30Z"),
      status: "ok",
      outcome: "success",
    });
    const r = await getAoiFreshness(db, {
      aoiId,
      userId: "user_2fresh3",
      // 31 minutes after finish
      now: new Date("2026-05-07T11:31:31Z"),
    });
    expect(r!.outcome).toBe("success");
    expect(r!.isStale).toBe(true);
  });

  it("does not leak other AOIs' bucket runs", async () => {
    const aoiA = await seedAoi(db, { userId: "user_2fresh4", bucket: "5x5:W125_N40", name: "A" });
    await seedAoi(db, { userId: "user_2fresh4", bucket: "5x5:W120_N40", name: "B" });
    await insertJobRun(db, {
      bucket: "5x5:W120_N40",
      startedAt: new Date("2026-05-07T12:00:00Z"),
      finishedAt: new Date("2026-05-07T12:01:00Z"),
      status: "error",
      outcome: "network_error",
      retryPending: true,
    });
    const ra = await getAoiFreshness(db, { aoiId: aoiA, userId: "user_2fresh4" });
    expect(ra!.bucket).toBe("5x5:W125_N40");
    expect(ra!.outcome).toBeNull();
    expect(ra!.lastPolledAt).toBeNull();
  });

  it("scopes to user — does not return another user's AOI", async () => {
    const aoiId = await seedAoi(db, { userId: "user_2fresh5", bucket: "5x5:W125_N40" });
    const r = await getAoiFreshness(db, { aoiId, userId: "user_2other" });
    expect(r).toBeNull();
  });
});
