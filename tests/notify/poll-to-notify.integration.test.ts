/**
 * Stage 4 integration: poll route → matcher → brief → notify on a real
 * PostGIS testcontainer. AI Gateway and Resend are stubbed.
 *
 * Verifies:
 *   - the dispatcher is called for the generated brief
 *   - notifications_log gets one row with status='sent'
 *   - aoi_briefs.last_notified_at is populated
 *   - job_runs.notifications_sent rolls up to 1
 *   - re-running the poll does not produce a second send (idempotency)
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
    `[integration] Skipping poll-to-notify integration test — Docker not available: ${probe.reason ?? "unknown"}`,
  );
}

describeIntegration("/api/aoi/poll → notify — PostGIS integration", () => {
  let handle: TestcontainerHandle | null = null;
  let sendCalls: Array<{ to: string; subject: string }> = [];
  let lastBriefId: string | null = null;

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
    sendCalls = [];
    lastBriefId = null;
    await handle!.pool.query(`DELETE FROM notifications_log`);
    await handle!.pool.query(`DELETE FROM aoi_briefs`);
    await handle!.pool.query(`DELETE FROM aoi_events`);
    await handle!.pool.query(`DELETE FROM firms_detections`);
    await handle!.pool.query(`DELETE FROM aoi_rules`);
    await handle!.pool.query(`DELETE FROM aois`);
    await handle!.pool.query(`DELETE FROM users WHERE id <> 'stub-user-1'`);
    await handle!.pool.query(`DELETE FROM job_runs`);

    await handle!.pool.query(
      `INSERT INTO users (id, email)
       VALUES ('integ-user-1', 'integration@example.org')
       ON CONFLICT (id) DO UPDATE SET email = EXCLUDED.email`,
    );

    await handle!.pool.query(
      `INSERT INTO aois (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
       VALUES (
         'integ-user-1',
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
      `INSERT INTO aoi_rules (aoi_id, distance_buffer_km, min_confidence, min_frp_mw, notify_channels)
       SELECT id, 25, 'nominal', 5, '[{"type":"email","target":"alerts@example.org"}]'::jsonb FROM aois LIMIT 1`,
    );

    _setTestDb(handle!.db);
    process.env.CRON_SECRET = "cron-secret";
    process.env.FIRMS_MAP_KEY = "firms-key";
    process.env.RESEND_API_KEY = "re_test_dummy";
    process.env.DATABASE_URL =
      (handle!.pool.options as { connectionString?: string }).connectionString ?? "";
  });

  it("dispatches one email per generated brief and is idempotent on second poll", async () => {
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
        summary: "Stubbed integration brief.",
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
      const inserted = await handle!.pool.query(
        `INSERT INTO aoi_briefs (aoi_id, event_id, model, gate_reason, payload, rendered_markdown)
         SELECT aoi_id, id, 'test/stub', 'multi_pixel', $1::jsonb, $2 FROM aoi_events WHERE id = $3
         ON CONFLICT (event_id) DO NOTHING
         RETURNING id`,
        [JSON.stringify(stub), md, eventId],
      );
      const briefId = inserted.rows[0]?.id;
      if (briefId) lastBriefId = String(briefId);
      await handle!.pool.query(
        `UPDATE aoi_events SET last_brief_at = now() WHERE id = $1`,
        [eventId],
      );
      return briefId
        ? {
            status: "generated",
            eventId,
            briefId: String(briefId),
            modelId: "test/stub",
            gateReason: "multi_pixel",
          }
        : { status: "skipped", eventId, reason: "already_briefed" };
    });

    // Stub the dispatcher to record sends + write notifications_log directly
    // against the testcontainer pool. We exercise the *real* dispatcher in
    // the unit tests; here we focus on the route-level rollups + idempotency.
    const { _setTestNotifyDispatch } = await import("@/app/api/aoi/poll/route");
    const { dispatchBrief } = await import("@/lib/notify/dispatch");

    _setTestNotifyDispatch(async (db, briefId) => {
      return dispatchBrief(db, briefId, {
        send: async (a) => {
          sendCalls.push({ to: a.to, subject: a.subject });
          return { ok: true, providerMessageId: `int-${sendCalls.length}`, latencyMs: 1 };
        },
      });
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
        briefsGenerated: number;
        notificationsSent: number;
      }>;
      totalNotificationsSent: number;
    };
    expect(body.runs[0].briefsGenerated).toBe(1);
    expect(body.runs[0].notificationsSent).toBe(1);
    expect(body.totalNotificationsSent).toBe(1);
    expect(sendCalls).toHaveLength(1);
    expect(sendCalls[0].to).toBe("alerts@example.org");

    const logs = await handle!.pool.query(
      `SELECT status, target FROM notifications_log WHERE brief_id = $1`,
      [lastBriefId],
    );
    expect(logs.rows).toHaveLength(1);
    expect(logs.rows[0].status).toBe("sent");

    const briefRow = await handle!.pool.query(
      `SELECT last_notified_at FROM aoi_briefs WHERE id = $1`,
      [lastBriefId],
    );
    expect(briefRow.rows[0].last_notified_at).not.toBeNull();

    const parentJob = await handle!.pool.query(
      `SELECT notifications_sent FROM job_runs WHERE bucket IS NULL ORDER BY id DESC LIMIT 1`,
    );
    expect(parentJob.rows[0].notifications_sent).toBe(1);

    // Second poll — same bucket. The matcher will UPDATE the existing event
    // (no new event), so generated briefs = 0 and no second send occurs.
    sendCalls = [];
    const res2 = await pollPost(
      new Request("http://localhost/api/aoi/poll", {
        method: "POST",
        headers: {
          authorization: "Bearer cron-secret",
          "content-type": "application/json",
        },
        body: JSON.stringify({ bucket: SONOMA_BUCKET }),
      }) as Parameters<typeof pollPost>[0],
    );
    expect(res2.status).toBe(200);
    expect(sendCalls).toHaveLength(0);
    const logs2 = await handle!.pool.query(
      `SELECT COUNT(*)::int AS c FROM notifications_log WHERE brief_id = $1`,
      [lastBriefId],
    );
    expect(logs2.rows[0].c).toBe(1);

    _setTestNotifyDispatch(null);
  });
});
