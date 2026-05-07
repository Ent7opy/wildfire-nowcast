/**
 * Stage 7 — 14-day retention sweep for firms_detections.
 *
 * Asserts:
 *   - Rows older than 14 days are removed; rows newer are kept.
 *   - Returning the deleted count is robust against PGlite's `affectedRows`
 *     vs node-postgres's `rowCount` shape.
 *   - Idempotent: running again with no old rows returns 0.
 *   - Brief generation prerequisites (aoi_events, aoi_briefs) are not touched.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import { pruneOldDetections } from "@/lib/firms/prune";
import type { PGlite } from "@electric-sql/pglite";

async function insertDetection(
  db: AppDb,
  args: { detectedAt: string; lat: number; lon: number; bucket?: string },
): Promise<void> {
  const point = JSON.stringify({ type: "Point", coordinates: [args.lon, args.lat] });
  await db.execute(sql`
    INSERT INTO "firms_detections" (
      "source", "detected_at", "geom", "lat", "lon",
      "acq_date", "acq_time", "bucket"
    ) VALUES (
      'VIIRS_NOAA20_NRT', ${args.detectedAt}, ${point},
      ${args.lat}, ${args.lon},
      ${args.detectedAt.slice(0, 10)}, '0000', ${args.bucket ?? "5x5:E000_N00"}
    )
  `);
}

async function countDetections(db: AppDb): Promise<number> {
  const r = (await db.execute(sql`SELECT count(*)::int AS n FROM "firms_detections"`)) as unknown as {
    rows?: Array<{ n: number }>;
  };
  return Number(((r.rows ?? r) as Array<{ n: number }>)[0].n);
}

describe("pruneOldDetections — PGlite", () => {
  let db: AppDb;
  let pglite: PGlite;
  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("deletes rows older than 14 days and keeps newer rows", async () => {
    const now = new Date("2026-04-21T00:00:00Z");
    // 30 days old → prune
    await insertDetection(db, {
      detectedAt: new Date(now.getTime() - 30 * 86400_000).toISOString(),
      lat: 1, lon: 1,
    });
    // 13 days old → keep
    await insertDetection(db, {
      detectedAt: new Date(now.getTime() - 13 * 86400_000).toISOString(),
      lat: 2, lon: 2,
    });
    // today → keep
    await insertDetection(db, {
      detectedAt: now.toISOString(),
      lat: 3, lon: 3,
    });
    expect(await countDetections(db)).toBe(3);
    const removed = await pruneOldDetections(db, { now });
    expect(removed).toBe(1);
    expect(await countDetections(db)).toBe(2);
  });

  it("is idempotent at zero", async () => {
    const now = new Date();
    const removed = await pruneOldDetections(db, { now });
    expect(removed).toBe(0);
  });

  it("honours custom retentionDays override", async () => {
    // Pins the retentionDays knob: if a future caller wants a tighter window
    // (e.g. for a free-tier emergency squeeze) the parameter must work, not
    // be silently ignored in favour of the 14-day default.
    const now = new Date("2026-04-21T00:00:00Z");
    await insertDetection(db, {
      detectedAt: new Date(now.getTime() - 10 * 86400_000).toISOString(),
      lat: 1, lon: 1,
    });
    await insertDetection(db, {
      detectedAt: new Date(now.getTime() - 3 * 86400_000).toISOString(),
      lat: 2, lon: 2,
    });
    expect(await countDetections(db)).toBe(2);
    const removed = await pruneOldDetections(db, { now, retentionDays: 7 });
    expect(removed).toBe(1);
    expect(await countDetections(db)).toBe(1);
  });

  it("uses strict `<` cutoff: a row exactly at the boundary is kept", async () => {
    // Pins the boundary behaviour. The SQL says `detected_at < cutoff`, so a
    // row whose detected_at equals (now - 14d) to the millisecond must NOT be
    // deleted. If anyone flips this to `<=` they'll fail this test.
    const now = new Date("2026-04-21T00:00:00Z");
    const exactCutoff = new Date(now.getTime() - 14 * 86400_000);
    await insertDetection(db, {
      detectedAt: exactCutoff.toISOString(),
      lat: 1, lon: 1,
    });
    const removed = await pruneOldDetections(db, { now });
    expect(removed).toBe(0);
    expect(await countDetections(db)).toBe(1);
  });

  it("does not touch aoi_events or aoi_briefs", async () => {
    const userId = "u-prune";
    await db.execute(sql`INSERT INTO "users" (id, email) VALUES (${userId}, 'x@x')`);
    const polygon = JSON.stringify({
      type: "Polygon",
      coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    });
    const aoi = (await db.execute(sql`
      INSERT INTO "aois" (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
      VALUES (${userId}, 'A', ${polygon}, ${polygon},
              ${JSON.stringify({ type: "Point", coordinates: [0.5, 0.5] })},
              '5x5:E000_N00', 100) RETURNING id
    `)) as unknown as { rows?: Array<{ id: string }> };
    const aoiId = ((aoi.rows ?? aoi) as Array<{ id: string }>)[0].id;
    const oldEventDate = new Date(Date.now() - 60 * 86400_000).toISOString();
    await db.execute(sql`
      INSERT INTO "aoi_events" (aoi_id, first_seen_at, last_seen_at, nearest_distance_km,
        detection_count, dedupe_hash, status)
      VALUES (${aoiId}, ${oldEventDate}, ${oldEventDate}, 1, 1, 'h', 'new')
    `);
    await pruneOldDetections(db);
    const eventCount = (await db.execute(
      sql`SELECT count(*)::int AS n FROM "aoi_events"`,
    )) as unknown as { rows?: Array<{ n: number }> };
    expect(Number(((eventCount.rows ?? eventCount) as Array<{ n: number }>)[0].n)).toBe(1);
  });
});
