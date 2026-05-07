/**
 * Stage 9 — backfill unit tests on PGlite.
 *
 * PGlite has no PostGIS, so the matcher's spatial step returns []. These
 * tests cover:
 *   - config_missing skip path (FIRMS_MAP_KEY unset, no fetchImpl override)
 *   - fetch error path
 *   - happy path (job_runs row, 'aoi-backfill', detections inserted)
 *
 * Single-AOI scope and full integration end-to-end are exercised in
 * `tests/firms-matcher.integration.test.ts` (matcher) plus the AOI POST
 * after-response test that pipes through the real wiring.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import type { PGlite } from "@electric-sql/pglite";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import { backfillForNewAoi } from "@/lib/firms/backfill";
import type { FirmsFetchResult } from "@/lib/firms/client";

async function seedAoi(db: AppDb): Promise<{ aoiId: string; userId: string }> {
  const userId = `stub-user-${Math.random().toString(36).slice(2, 8)}`;
  await db.execute(sql`
    INSERT INTO "users" (id, email) VALUES (${userId}, 'owner@example.org')
  `);
  const polygon = JSON.stringify({
    type: "Polygon",
    coordinates: [
      [[-122.7, 38.4], [-122.6, 38.4], [-122.6, 38.5], [-122.7, 38.5], [-122.7, 38.4]],
    ],
  });
  const r = (await db.execute(sql`
    INSERT INTO "aois" (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
    VALUES (
      ${userId}, 'Test AOI', ${polygon}, ${polygon},
      ${JSON.stringify({ type: "Point", coordinates: [-122.65, 38.45] })},
      '5x5:W125_N35', 100
    ) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const rows = (r.rows ?? (r as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  await db.execute(sql`
    INSERT INTO "aoi_rules" (aoi_id) VALUES (${rows[0].id})
  `);
  return { aoiId: rows[0].id, userId };
}

async function readJobRuns(db: AppDb): Promise<Array<Record<string, unknown>>> {
  const r = (await db.execute(sql`
    SELECT * FROM "job_runs" WHERE "job_name" = 'aoi-backfill'
    ORDER BY "started_at" ASC
  `)) as unknown as { rows?: Array<Record<string, unknown>> };
  return (r.rows ?? (r as unknown as Array<Record<string, unknown>>)) as Array<Record<string, unknown>>;
}

describe("backfillForNewAoi — PGlite", () => {
  let db: AppDb;
  let pglite: PGlite;
  const originalKey = process.env.FIRMS_MAP_KEY;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });

  afterEach(async () => {
    await pglite.close();
    if (originalKey === undefined) delete process.env.FIRMS_MAP_KEY;
    else process.env.FIRMS_MAP_KEY = originalKey;
  });

  it("FIRMS_MAP_KEY unset → returns skipped/config_missing, no firms_detections written", async () => {
    delete process.env.FIRMS_MAP_KEY;
    const { aoiId, userId } = await seedAoi(db);
    const outcome = await backfillForNewAoi(db, {
      aoiId,
      userId,
      regionBucket: "5x5:W125_N35",
    });
    expect(outcome.status).toBe("skipped");
    expect(outcome.reason).toBe("config_missing");
    const detRows = (await db.execute(sql`SELECT COUNT(*)::int AS c FROM "firms_detections"`)) as unknown as {
      rows?: Array<{ c: number }>;
    };
    const c = (detRows.rows ?? (detRows as unknown as Array<{ c: number }>))[0].c;
    expect(Number(c)).toBe(0);
    const runs = await readJobRuns(db);
    expect(runs).toHaveLength(1);
    expect(runs[0].status).toBe("ok");
    expect(runs[0].bucket).toBe("5x5:W125_N35");
  });

  it("FIRMS fetch error → returns error/fetch_failed; job_runs row closed status=error", async () => {
    const { aoiId, userId } = await seedAoi(db);
    const fetchImpl = async (): Promise<FirmsFetchResult> => ({
      ok: false,
      code: "upstream_error",
      message: "FIRMS 502",
    });
    const outcome = await backfillForNewAoi(db, {
      aoiId,
      userId,
      regionBucket: "5x5:W125_N35",
      fetchImpl,
    });
    expect(outcome.status).toBe("error");
    expect(outcome.reason).toBe("fetch_failed");
    const runs = await readJobRuns(db);
    expect(runs[0].status).toBe("error");
    expect(String(runs[0].error)).toContain("upstream_error");
  });

  it("happy path — fetch succeeds, opens & closes aoi-backfill job_run", async () => {
    const { aoiId, userId } = await seedAoi(db);
    const fetchImpl = async (): Promise<FirmsFetchResult> => ({
      ok: true,
      source: "VIIRS_NOAA20_NRT",
      bbox: [-125, 35, -120, 40],
      dayRange: 1,
      detections: [
        {
          latitude: 38.45,
          longitude: -122.65,
          brightTi4: 320,
          brightTi5: 290,
          scan: 0.4,
          track: 0.4,
          acqDate: "2026-05-07",
          acqTime: "1234",
          satellite: "N20",
          instrument: "VIIRS",
          confidence: "n",
          version: "2.0",
          frp: 12,
          daynight: "D",
        },
      ],
      emptyArea: false,
    });
    const outcome = await backfillForNewAoi(db, {
      aoiId,
      userId,
      regionBucket: "5x5:W125_N35",
      fetchImpl,
    });
    expect(outcome.status).toBe("ok");
    expect(outcome.detectionsFetched).toBe(1);
    const runs = await readJobRuns(db);
    expect(runs).toHaveLength(1);
    expect(runs[0].status).toBe("ok");
    expect(runs[0].job_name).toBe("aoi-backfill");
    expect(runs[0].bucket).toBe("5x5:W125_N35");
  });
});
