/**
 * Stage 8 — `<FreshnessBanner>` smoke test. Renders happy + each degraded
 * variant by stubbing the freshness query directly via PGlite seed.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { renderToStaticMarkup } from "react-dom/server.node";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import { FreshnessBanner } from "@/app/dashboard/_components/freshness-banner";
import type { PGlite } from "@electric-sql/pglite";

async function seedAoi(db: AppDb, userId: string, bucket: string): Promise<string> {
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
    VALUES (${userId}, ${userId + "@example.org"})
    ON CONFLICT ("id") DO NOTHING
  `);
  const res = (await db.execute(sql`
    INSERT INTO "aois" (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
    VALUES (
      ${userId}, 'A',
      ${polygon}, ${polygon},
      ${JSON.stringify({ type: "Point", coordinates: [-122.65, 38.45] })},
      ${bucket}, 100
    ) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const rows = (res.rows ?? (res as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  return rows[0].id;
}

async function insertRun(
  db: AppDb,
  args: {
    bucket: string;
    startedAt: Date;
    finishedAt: Date;
    status: string;
    outcome: string | null;
    retryPending: boolean;
  },
): Promise<void> {
  await db.execute(sql`
    INSERT INTO "job_runs" ("job_name", "bucket", "started_at", "finished_at", "status", "outcome", "retry_pending")
    VALUES ('firms-poll', ${args.bucket}, ${args.startedAt.toISOString()}, ${args.finishedAt.toISOString()}, ${args.status}, ${args.outcome}, ${args.retryPending})
  `);
}

describe("<FreshnessBanner>", () => {
  let db: AppDb;
  let pglite: PGlite;
  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("renders 'First poll pending' when no completed run exists", async () => {
    const aoiId = await seedAoi(db, "user_ban1", "5x5:W125_N40");
    const node = await FreshnessBanner({
      db,
      aoiId,
      userId: "user_ban1",
      __testNow: new Date("2026-05-07T12:00:00Z"),
    });
    const html = renderToStaticMarkup(node);
    expect(html).toContain("First poll pending");
    expect(html).toContain("--warn");
  });

  it("renders happy 'Last polled N minutes ago' for a fresh success", async () => {
    const aoiId = await seedAoi(db, "user_ban2", "5x5:W125_N40");
    await insertRun(db, {
      bucket: "5x5:W125_N40",
      startedAt: new Date("2026-05-07T11:53:00Z"),
      finishedAt: new Date("2026-05-07T11:53:10Z"),
      status: "ok",
      outcome: "success",
      retryPending: false,
    });
    const node = await FreshnessBanner({
      db,
      aoiId,
      userId: "user_ban2",
      __testNow: new Date("2026-05-07T12:00:00Z"),
    });
    const html = renderToStaticMarkup(node);
    expect(html).toContain("Last polled");
    expect(html).toContain("minutes ago");
    expect(html).not.toContain("--warn");
  });

  it("renders rate-limited yellow banner with retry hint", async () => {
    const aoiId = await seedAoi(db, "user_ban3", "5x5:W125_N40");
    await insertRun(db, {
      bucket: "5x5:W125_N40",
      startedAt: new Date("2026-05-07T11:55:00Z"),
      finishedAt: new Date("2026-05-07T11:55:01Z"),
      status: "error",
      outcome: "rate_limited",
      retryPending: true,
    });
    const node = await FreshnessBanner({
      db,
      aoiId,
      userId: "user_ban3",
      __testNow: new Date("2026-05-07T12:00:00Z"),
    });
    const html = renderToStaticMarkup(node);
    expect(html).toContain("rate-limited");
    expect(html).toContain("retrying next tick");
    expect(html).toContain("--warn");
  });

  it("renders stale-success banner past the 30-min boundary", async () => {
    const aoiId = await seedAoi(db, "user_ban4", "5x5:W125_N40");
    await insertRun(db, {
      bucket: "5x5:W125_N40",
      startedAt: new Date("2026-05-07T11:00:00Z"),
      finishedAt: new Date("2026-05-07T11:00:10Z"),
      status: "ok",
      outcome: "success",
      retryPending: false,
    });
    const node = await FreshnessBanner({
      db,
      aoiId,
      userId: "user_ban4",
      __testNow: new Date("2026-05-07T11:31:00Z"),
    });
    const html = renderToStaticMarkup(node);
    expect(html).toContain("Polling delayed");
    expect(html).toContain("--warn");
  });
});
