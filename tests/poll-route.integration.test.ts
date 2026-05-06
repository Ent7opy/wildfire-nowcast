/**
 * End-to-end poll route against a real PostGIS testcontainer.
 *
 * Uses `_setTestFirmsFetch` to stub the FIRMS client — we never hit the live
 * NASA endpoint from tests. Verifies:
 *   - happy path: detections flow through → events created → job_runs rows
 *     closed with status=ok
 *   - one bucket failing does not abort the whole poll; parent job_run ends
 *     as "partial"
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  dockerAvailable,
  tryStartPostgisContainer,
  type TestcontainerHandle,
} from "@/db/test/testcontainer";
import { _setTestDb } from "@/lib/db/client";
import { regionBucketFromLonLat } from "@/lib/geo/region-bucket";
import { _setTestFirmsFetch, POST as pollPost } from "@/app/api/aoi/poll/route";
import type { FirmsFetchResult } from "@/lib/firms/client";

const SONOMA_LAT = 38.46;
const SONOMA_LON = -122.67;
const SONOMA_BUCKET = regionBucketFromLonLat(SONOMA_LON, SONOMA_LAT);

const probe = await dockerAvailable();
const describeIntegration = probe.available ? describe : describe.skip;

if (!probe.available) {
  console.warn(
    `[integration] Skipping poll-route integration tests — Docker not available: ${probe.reason ?? "unknown"}`,
  );
}

describeIntegration("/api/aoi/poll — PostGIS integration", () => {
  let handle: TestcontainerHandle | null = null;

  beforeAll(async () => {
    handle = await tryStartPostgisContainer();
  }, 180_000);

  afterAll(async () => {
    if (handle) await handle.stop();
    _setTestFirmsFetch(null);
    _setTestDb(null);
  });

  beforeEach(async (ctx) => {
    if (!handle) {
      ctx.skip();
      return;
    }
    await handle!.pool.query(`DELETE FROM aoi_events`);
    await handle!.pool.query(`DELETE FROM firms_detections`);
    await handle!.pool.query(`DELETE FROM aoi_rules`);
    await handle!.pool.query(`DELETE FROM aois`);
    await handle!.pool.query(`DELETE FROM job_runs`);
    await handle!.pool.query(
      `INSERT INTO users (id, email) VALUES ('user_2pollRouteOwner', 'poll@example.org')
       ON CONFLICT (id) DO NOTHING`,
    );

    // Seed one AOI in the Sonoma bucket.
    await handle!.pool.query(
      `INSERT INTO aois (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
       VALUES (
         'user_2pollRouteOwner',
         'Spring Creek Preserve',
         ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)),
         ST_SetSRID(ST_Envelope(ST_GeomFromGeoJSON($1)), 4326),
         ST_SetSRID(ST_Centroid(ST_GeomFromGeoJSON($1)), 4326),
         $2,
         2040
       )`,
      [
        JSON.stringify({
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
        }),
        SONOMA_BUCKET,
      ],
    );
    await handle!.pool.query(
      `INSERT INTO aoi_rules (aoi_id, distance_buffer_km, min_confidence, min_frp_mw)
       SELECT id, 25, 'nominal', 5 FROM aois LIMIT 1`,
    );

    _setTestDb(handle!.db);
    process.env.CRON_SECRET = "cron-secret";
    process.env.FIRMS_MAP_KEY = "firms-key";
    process.env.DATABASE_URL =
      (handle!.pool.options as { connectionString?: string }).connectionString ?? "";
  });

  function stubFirms(result: FirmsFetchResult): void {
    _setTestFirmsFetch(async () => result);
  }

  it("happy path: detections become events and job_runs rows close ok", async () => {
    stubFirms({
      ok: true,
      source: "VIIRS_NOAA20_NRT",
      bbox: [-125, 35, -120, 40],
      dayRange: 1,
      detections: [
        {
          latitude: SONOMA_LAT + 0.08,
          longitude: SONOMA_LON + 0.01,
          brightTi4: 325,
          brightTi5: 289,
          scan: 0.4,
          track: 0.4,
          acqDate: "2026-04-21",
          acqTime: "0417",
          satellite: "1",
          instrument: "VIIRS",
          confidence: "n",
          version: "2.0NRT",
          frp: 11.2,
          daynight: "N",
        },
      ],
      emptyArea: false,
    });

    const res = await pollPost(
      new Request("http://localhost/api/aoi/poll", {
        method: "POST",
        headers: {
          authorization: "Bearer cron-secret",
          "content-type": "application/json",
        },
        body: JSON.stringify({ bucket: SONOMA_BUCKET }),
      }) as Parameters<typeof pollPost>[0],
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      runs: Array<{ status: string; detectionsInserted: number; eventsCreated: number }>;
    };
    expect(body.runs).toHaveLength(1);
    expect(body.runs[0].status).toBe("ok");
    expect(body.runs[0].detectionsInserted).toBe(1);
    expect(body.runs[0].eventsCreated).toBe(1);

    const events = await handle!.pool.query(`SELECT COUNT(*)::int AS c FROM aoi_events`);
    expect(events.rows[0].c).toBe(1);

    const jobRuns = await handle!.pool.query(
      `SELECT status, bucket, finished_at FROM job_runs ORDER BY id ASC`,
    );
    // parent + per-bucket child
    expect(jobRuns.rowCount).toBe(2);
    for (const row of jobRuns.rows) {
      expect(row.status).toBe("ok");
      expect(row.finished_at).not.toBeNull();
    }
  });

  it("marks the parent run 'partial' when a bucket fetch fails", async () => {
    stubFirms({
      ok: false,
      code: "upstream_error",
      message: "FIRMS flaky",
      status: 502,
    });

    const res = await pollPost(
      new Request("http://localhost/api/aoi/poll", {
        method: "POST",
        headers: {
          authorization: "Bearer cron-secret",
          "content-type": "application/json",
        },
        body: JSON.stringify({ bucket: SONOMA_BUCKET }),
      }) as Parameters<typeof pollPost>[0],
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      runs: Array<{ status: string; error?: string }>;
    };
    expect(body.runs[0].status).toBe("error");
    expect(body.runs[0].error).toContain("upstream_error");

    const parent = await handle!.pool.query(
      `SELECT status FROM job_runs WHERE bucket IS NULL`,
    );
    expect(parent.rows[0].status).toBe("partial");
    const child = await handle!.pool.query(
      `SELECT status, error FROM job_runs WHERE bucket IS NOT NULL`,
    );
    expect(child.rows[0].status).toBe("error");
    expect(child.rows[0].error).toContain("upstream_error");
  });
});
