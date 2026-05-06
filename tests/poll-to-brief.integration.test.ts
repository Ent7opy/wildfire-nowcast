/**
 * Stage 3 integration: poll route → matcher → brief generator on a real
 * PostGIS testcontainer. AI Gateway is stubbed via `_setTestBriefGen` so we
 * never hit the live LLM endpoint.
 *
 * Verifies the end-to-end Stage 2 + Stage 3 plumbing:
 *   - a synthetic FIRMS detection becomes an aoi_event
 *   - the brief generator is invoked with the new event id
 *   - aoi_briefs row is persisted with the rendered markdown
 *   - per-bucket outcome reports briefsGenerated=1 and no skip reasons
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  dockerAvailable,
  tryStartPostgisContainer,
  type TestcontainerHandle,
} from "@/db/test/testcontainer";
import { _setTestDb } from "@/lib/db/client";
import { regionBucketFromLonLat } from "@/lib/geo/region-bucket";
import {
  _setTestFirmsFetch,
  _setTestBriefGen,
  POST as pollPost,
} from "@/app/api/aoi/poll/route";
import type { FirmsFetchResult } from "@/lib/firms/client";
import type { GenerateOutcome } from "@/lib/ai/generate";
import type { Brief } from "@/lib/ai/schema";
import { SCHEMA_VERSION } from "@/lib/ai/schema";
import { renderBriefMarkdown } from "@/lib/ai/render";

const SONOMA_LAT = 38.46;
const SONOMA_LON = -122.67;
const SONOMA_BUCKET = regionBucketFromLonLat(SONOMA_LON, SONOMA_LAT);

const probe = await dockerAvailable();
const describeIntegration = probe.available ? describe : describe.skip;

if (!probe.available) {
  console.warn(
    `[integration] Skipping poll-to-brief integration test — Docker not available: ${probe.reason ?? "unknown"}`,
  );
}

describeIntegration("/api/aoi/poll → brief — PostGIS integration", () => {
  let handle: TestcontainerHandle | null = null;

  beforeAll(async () => {
    handle = await tryStartPostgisContainer();
  }, 180_000);

  afterAll(async () => {
    if (handle) await handle.stop();
    _setTestFirmsFetch(null);
    _setTestBriefGen(null);
    _setTestDb(null);
  });

  beforeEach(async (ctx) => {
    if (!handle) {
      ctx.skip();
      return;
    }
    await handle!.pool.query(`DELETE FROM aoi_briefs`);
    await handle!.pool.query(`DELETE FROM aoi_events`);
    await handle!.pool.query(`DELETE FROM firms_detections`);
    await handle!.pool.query(`DELETE FROM aoi_rules`);
    await handle!.pool.query(`DELETE FROM aois`);
    await handle!.pool.query(`DELETE FROM job_runs`);
    await handle!.pool.query(
      `INSERT INTO users (id, email) VALUES ('user_2pollToBriefOwner', 'pollbrief@example.org')
       ON CONFLICT (id) DO NOTHING`,
    );

    await handle!.pool.query(
      `INSERT INTO aois (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
       VALUES (
         'user_2pollToBriefOwner',
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

  it("generates a brief end-to-end for a synthetic detection", async () => {
    _setTestFirmsFetch(async (): Promise<FirmsFetchResult> => ({
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
        {
          latitude: SONOMA_LAT + 0.082,
          longitude: SONOMA_LON + 0.011,
          brightTi4: 326,
          brightTi5: 290,
          scan: 0.4,
          track: 0.4,
          acqDate: "2026-04-21",
          acqTime: "0418",
          satellite: "1",
          instrument: "VIIRS",
          confidence: "n",
          version: "2.0NRT",
          frp: 9.8,
          daynight: "N",
        },
      ],
      emptyArea: false,
    }));

    _setTestBriefGen(async (_db, eventId): Promise<GenerateOutcome> => {
      const stub: Brief = {
        schema_version: SCHEMA_VERSION,
        aoi: {
          id: "00000000-0000-4000-8000-000000000099",
          name: "Spring Creek Preserve",
          area_ha: 2040,
        },
        summary: "Stubbed integration brief (fixture).",
        key_facts: {
          nearest_detection_km: 8,
          bearing_from_aoi_deg: 0,
          wind_dir_deg: null,
          wind_speed_kmh: null,
          wind_toward_aoi: null,
          detection_count_in_window: 2,
          max_frp_mw: 11,
          satellites: ["VIIRS_NOAA20_NRT"],
          window_hours: 24,
        },
        context: {
          weather_note: null,
          authority_perimeter: { source: null, posted_ts: null, contains_detection: null },
          prior_events: [],
        },
        recommended_watch_items: ["fixture watch item"],
        uncertainty: "fixture",
        next_brief_hint: { when: "next tick", trigger: "fixture" },
      };
      const md = renderBriefMarkdown(stub);
      // Persist via the actual DB pool (mirrors what the real generator does,
      // minus the AI call) so the integration assertion can hit aoi_briefs.
      const inserted = await handle!.pool.query(
        `INSERT INTO aoi_briefs (aoi_id, event_id, model, gate_reason, payload, rendered_markdown)
         SELECT aoi_id, id, 'test/stub', 'multi_pixel', $1::jsonb, $2 FROM aoi_events WHERE id = $3
         RETURNING id`,
        [JSON.stringify(stub), md, eventId],
      );
      await handle!.pool.query(
        `UPDATE aoi_events SET last_brief_at = now() WHERE id = $1`,
        [eventId],
      );
      return {
        status: "generated",
        eventId,
        briefId: String(inserted.rows[0].id),
        modelId: "test/stub",
        gateReason: "multi_pixel",
      };
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
      runs: Array<{
        status: string;
        eventsCreated: number;
        briefsGenerated: number;
        briefSkipReason: Record<string, string>;
      }>;
      totalBriefsGenerated: number;
    };
    expect(body.runs).toHaveLength(1);
    expect(body.runs[0].status).toBe("ok");
    expect(body.runs[0].eventsCreated).toBe(1);
    expect(body.runs[0].briefsGenerated).toBe(1);
    expect(Object.keys(body.runs[0].briefSkipReason)).toHaveLength(0);
    expect(body.totalBriefsGenerated).toBe(1);

    const briefs = await handle!.pool.query(`SELECT COUNT(*)::int AS c FROM aoi_briefs`);
    expect(briefs.rows[0].c).toBe(1);
  });
});
