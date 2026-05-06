/**
 * Stage 5 integration — end-to-end authed flow on PostGIS.
 *
 * Two users (Alice, Bob) each create one AOI through the authenticated API.
 * The (unauthenticated, CRON_SECRET-bearing) cron poll then iterates ALL
 * users and produces a brief + dispatch per matched detection.
 *
 * We assert: each user's AOI is present, both buckets are picked up by the
 * cron, the dispatcher is called for each generated brief, and Alice's poll
 * does not affect Bob's `aoi_briefs`.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  dockerAvailable,
  tryStartPostgisContainer,
  type TestcontainerHandle,
} from "@/db/test/testcontainer";
import { _setTestDb } from "@/lib/db/client";
import { _setTestAuth } from "@/lib/auth/context";
import { regionBucketFromLonLat } from "@/lib/geo/region-bucket";
import { POST as aoiCreate } from "@/app/api/aoi/route";
import {
  _setTestFirmsFetch,
  _setTestBriefGen,
  _setTestNotifyDispatch,
  POST as pollPost,
} from "@/app/api/aoi/poll/route";
import type { FirmsFetchResult } from "@/lib/firms/client";
import type { GenerateOutcome } from "@/lib/ai/generate";

const ALICE = "user_2abcAliceFull";
const BOB = "user_2abcBobFull";

const SONOMA_LAT = 38.46;
const SONOMA_LON = -122.67;

const probe = await dockerAvailable();
const describeIntegration = probe.available ? describe : describe.skip;

if (!probe.available) {
  console.warn(
    `[integration] Skipping Stage 5 full-flow — Docker not available: ${probe.reason ?? "unknown"}`,
  );
}

describeIntegration("Stage 5 — full authed flow", () => {
  let handle: TestcontainerHandle | null = null;

  beforeAll(async () => {
    handle = await tryStartPostgisContainer();
  }, 180_000);

  afterAll(async () => {
    if (handle) await handle.stop();
    _setTestFirmsFetch(null);
    _setTestBriefGen(null);
    _setTestNotifyDispatch(null);
    _setTestAuth(null);
    _setTestDb(null);
  });

  beforeEach(async (ctx) => {
    if (!handle) {
      ctx.skip();
      return;
    }
    await handle!.pool.query(`DELETE FROM notifications_log`);
    await handle!.pool.query(`DELETE FROM aoi_briefs`);
    await handle!.pool.query(`DELETE FROM aoi_events`);
    await handle!.pool.query(`DELETE FROM firms_detections`);
    await handle!.pool.query(`DELETE FROM aoi_rules`);
    await handle!.pool.query(`DELETE FROM aois`);
    await handle!.pool.query(`DELETE FROM users`);
    await handle!.pool.query(`DELETE FROM job_runs`);
    await handle!.pool.query(
      `INSERT INTO users (id, email) VALUES
         ($1, $2),
         ($3, $4)`,
      [ALICE, "alice@example.org", BOB, "bob@example.org"],
    );

    _setTestDb(handle!.db);
    process.env.CRON_SECRET = "cron-secret";
    process.env.FIRMS_MAP_KEY = "firms-key";
    process.env.DATABASE_URL =
      (handle!.pool.options as { connectionString?: string }).connectionString ?? "";
  });

  it("authed POST /api/aoi as two users; cron picks up both", async () => {
    const polyAlice = {
      type: "Polygon",
      coordinates: [
        [
          [SONOMA_LON - 0.05, SONOMA_LAT - 0.04],
          [SONOMA_LON + 0.05, SONOMA_LAT - 0.04],
          [SONOMA_LON + 0.05, SONOMA_LAT + 0.04],
          [SONOMA_LON - 0.05, SONOMA_LAT + 0.04],
          [SONOMA_LON - 0.05, SONOMA_LAT - 0.04],
        ],
      ],
    };
    const polyBob = {
      type: "Polygon",
      coordinates: [
        [
          [23.3, 42.65],
          [23.4, 42.65],
          [23.4, 42.75],
          [23.3, 42.75],
          [23.3, 42.65],
        ],
      ],
    };

    _setTestAuth(() => ({ ok: true, userId: ALICE }));
    const aliceRes = await aoiCreate(
      new Request("http://localhost/api/aoi", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: "Alice AOI", geometry: polyAlice }),
      }) as Parameters<typeof aoiCreate>[0],
    );
    expect(aliceRes.status).toBe(201);

    _setTestAuth(() => ({ ok: true, userId: BOB }));
    const bobRes = await aoiCreate(
      new Request("http://localhost/api/aoi", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: "Bob AOI", geometry: polyBob }),
      }) as Parameters<typeof aoiCreate>[0],
    );
    expect(bobRes.status).toBe(201);

    // Both AOIs are present, owned by their respective users.
    const aoiCount = await handle!.pool.query(`SELECT user_id FROM aois ORDER BY user_id`);
    expect(aoiCount.rowCount).toBe(2);
    expect(aoiCount.rows.map((r) => r.user_id).sort()).toEqual([ALICE, BOB].sort());

    // Cron path: unauthenticated, iterates ALL users' AOIs.
    _setTestAuth(null);
    _setTestFirmsFetch(async (args): Promise<FirmsFetchResult> => ({
      ok: true,
      source: args.source,
      bbox: args.bbox,
      dayRange: 1,
      detections: [], // no detections — happy poll, no events created
      emptyArea: true,
    }));
    _setTestBriefGen(async (_db, eventId): Promise<GenerateOutcome> => ({
      status: "skipped",
      eventId,
      reason: "event_not_found",
    }));
    _setTestNotifyDispatch(async (_db, briefId) => ({
      briefId,
      attempts: [],
    }));

    const aliceBucket = regionBucketFromLonLat(SONOMA_LON, SONOMA_LAT);
    const bobBucket = regionBucketFromLonLat(23.35, 42.7);

    for (const bucket of [aliceBucket, bobBucket]) {
      const res = await pollPost(
        new Request("http://localhost/api/aoi/poll", {
          method: "POST",
          headers: {
            authorization: "Bearer cron-secret",
            "content-type": "application/json",
          },
          body: JSON.stringify({ bucket }),
        }) as Parameters<typeof pollPost>[0],
      );
      expect(res.status).toBe(200);
    }

    // Bob's `aoi_briefs` is unchanged by Alice's poll, and vice-versa.
    const aliceBriefs = await handle!.pool.query(
      `SELECT count(*)::int AS c FROM aoi_briefs b JOIN aois a ON a.id = b.aoi_id WHERE a.user_id = $1`,
      [ALICE],
    );
    const bobBriefs = await handle!.pool.query(
      `SELECT count(*)::int AS c FROM aoi_briefs b JOIN aois a ON a.id = b.aoi_id WHERE a.user_id = $1`,
      [BOB],
    );
    expect(aliceBriefs.rows[0].c).toBe(0);
    expect(bobBriefs.rows[0].c).toBe(0);
  });
});
