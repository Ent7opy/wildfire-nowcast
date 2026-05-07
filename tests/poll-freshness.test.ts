/**
 * Stage 8 — poll route writes the new freshness columns.
 *
 * PGlite-backed (no PostGIS needed; the matcher tolerates the non-postgis
 * backend via its turf-style fallback). Stubs FIRMS to return:
 *   - one bucket: rate-limited error
 *   - one bucket: success (no detections)
 * Then asserts the per-bucket job_runs rows carry outcome / retry_pending,
 * and asserts getAoiFreshness reports the expected banner state for an AOI
 * in each bucket.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import { _setTestDb, type AppDb } from "@/lib/db/client";
import {
  _setTestFirmsFetch,
  _setTestBriefGen,
  _setTestNotifyDispatch,
  POST as pollPost,
} from "@/app/api/aoi/poll/route";
import { getAoiFreshness } from "@/lib/db/freshness";
import type { PGlite } from "@electric-sql/pglite";

async function seedAoi(
  db: AppDb,
  args: { userId: string; bucket: string; name: string },
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
      ${args.userId}, ${args.name},
      ${polygon}, ${polygon},
      ${JSON.stringify({ type: "Point", coordinates: [-122.65, 38.45] })},
      ${args.bucket}, 100
    ) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const rows = (res.rows ?? (res as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  await db.execute(sql`
    INSERT INTO "aoi_rules" (aoi_id, distance_buffer_km, min_confidence, min_frp_mw)
    VALUES (${rows[0].id}, 25, 'nominal', 5)
  `);
  return rows[0].id;
}

describe("poll route — Stage 8 freshness writes", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    _setTestDb(db);
    process.env.CRON_SECRET = "cron-secret";
    process.env.FIRMS_MAP_KEY = "firms-key";
    process.env.DATABASE_URL = "postgres://stub/stub";
    // Brief gen and notify dispatch are noops so this test stays focused on
    // freshness writes from the FIRMS path.
    _setTestBriefGen(async (_db, eventId) => ({
      status: "skipped",
      eventId,
      reason: "config_missing",
    }));
    _setTestNotifyDispatch(async () => ({ briefId: "x", attempts: [] }));
  });

  afterEach(async () => {
    _setTestDb(null);
    _setTestFirmsFetch(null);
    _setTestBriefGen(null);
    _setTestNotifyDispatch(null);
    await pglite.close();
  });

  it("rate_limited bucket → outcome=rate_limited + retry_pending=true; success bucket → outcome=success", async () => {
    const aoiRl = await seedAoi(db, {
      userId: "user_2pollFresh",
      bucket: "5x5:W125_N35",
      name: "RateLimitedAOI",
    });
    const aoiOk = await seedAoi(db, {
      userId: "user_2pollFresh",
      bucket: "5x5:W120_N35",
      name: "OkAOI",
    });

    _setTestFirmsFetch(async (args) => {
      if (args.bbox[0] >= -125 && args.bbox[2] <= -120) {
        // rate-limited bucket
        return {
          ok: false,
          code: "rate_limited",
          message: "FIRMS quota",
          status: 429,
        };
      }
      return {
        ok: true,
        source: args.source,
        bbox: args.bbox,
        dayRange: args.dayRange ?? 1,
        detections: [],
        emptyArea: true,
      };
    });

    const res = await pollPost(
      new Request("http://localhost/api/aoi/poll", {
        method: "POST",
        headers: {
          authorization: "Bearer cron-secret",
          "content-type": "application/json",
        },
        body: "",
      }) as Parameters<typeof pollPost>[0],
    );
    expect(res.status).toBe(200);

    const child = (await db.execute(sql`
      SELECT bucket, status, outcome, retry_pending FROM job_runs WHERE bucket IS NOT NULL ORDER BY bucket
    `)) as unknown as {
      rows?: Array<{ bucket: string; status: string; outcome: string | null; retry_pending: boolean }>;
    };
    const rows = (child.rows ??
      (child as unknown as Array<{ bucket: string; status: string; outcome: string | null; retry_pending: boolean }>)) as Array<{
      bucket: string;
      status: string;
      outcome: string | null;
      retry_pending: boolean;
    }>;
    expect(rows).toHaveLength(2);
    const rl = rows.find((r) => r.bucket === "5x5:W125_N35")!;
    const ok = rows.find((r) => r.bucket === "5x5:W120_N35")!;
    expect(rl.outcome).toBe("rate_limited");
    expect(rl.retry_pending).toBe(true);
    expect(ok.outcome).toBe("success");
    expect(ok.retry_pending).toBe(false);

    const fRl = await getAoiFreshness(db, { aoiId: aoiRl, userId: "user_2pollFresh" });
    expect(fRl!.outcome).toBe("rate_limited");
    expect(fRl!.retryPending).toBe(true);
    const fOk = await getAoiFreshness(db, { aoiId: aoiOk, userId: "user_2pollFresh" });
    expect(fOk!.outcome).toBe("success");
  });
});
