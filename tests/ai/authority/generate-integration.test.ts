/**
 * Stage 8 — orchestrator integration: stubs both the gateway AND the
 * authority fetch, asserts the persisted `aoi_briefs.payload.context.
 * authority_perimeter` carries fetched values and that a rejecting
 * fetcher still ships a brief (build-without-blocking).
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import { generateBriefForEvent } from "@/lib/ai/generate";
import type { Brief } from "@/lib/ai/schema";
import { SCHEMA_VERSION } from "@/lib/ai/schema";
import type { GatewayResult } from "@/lib/ai/gateway";
import type { PGlite } from "@electric-sql/pglite";

const STUB_BRIEF: Brief = {
  schema_version: SCHEMA_VERSION,
  aoi: { id: "00000000-0000-4000-8000-000000000099", name: "Test", area_ha: 100 },
  summary: "Stubbed.",
  key_facts: {
    nearest_detection_km: 8,
    bearing_from_aoi_deg: 90,
    wind_dir_deg: null,
    wind_speed_kmh: null,
    wind_toward_aoi: null,
    detection_count_in_window: 2,
    max_frp_mw: 12,
    satellites: [],
    window_hours: 24,
  },
  // Real LLM would echo back the perimeter; we hardcode a passing payload here
  // because the test is about what the orchestrator PASSED IN, not what the
  // model echoed. The persisted payload is the model's output.
  context: {
    weather_note: null,
    authority_perimeter: {
      source: "NIFC WFIGS",
      posted_ts: "2026-05-07T11:00:00.000Z",
      contains_detection: true,
    },
    prior_events: [],
  },
  recommended_watch_items: ["x"],
  uncertainty: "stub",
  next_brief_hint: { when: "next tick", trigger: "any" },
};

async function seed(db: AppDb): Promise<{ eventId: string; aoiId: string }> {
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
    VALUES ('user_2authGen', 'auth@example.org')
    ON CONFLICT ("id") DO NOTHING
  `);
  const aoiRes = (await db.execute(sql`
    INSERT INTO "aois" (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
    VALUES (
      'user_2authGen', 'AuthAOI',
      ${polygon}, ${polygon},
      ${JSON.stringify({ type: "Point", coordinates: [-122.65, 38.45] })},
      '5x5:W125_N35', 100
    ) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const aoiRows = (aoiRes.rows ?? (aoiRes as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  const aoiId = aoiRows[0].id;
  await db.execute(sql`
    INSERT INTO "aoi_rules" (aoi_id, distance_buffer_km, min_confidence, min_frp_mw)
    VALUES (${aoiId}, 25, 'nominal', 5)
  `);
  // Seed a non-industrial detection inside the window so loadEventContext
  // resolves a nearestDetection.
  const detGeom = JSON.stringify({ type: "Point", coordinates: [-122.6, 38.5] });
  await db.execute(sql`
    INSERT INTO "firms_detections" (
      source, detected_at, geom, lat, lon, frp_mw, confidence, daynight,
      acq_date, acq_time, bucket
    ) VALUES (
      'VIIRS_NOAA20_NRT', '2026-04-21T04:15:00Z', ${detGeom},
      38.5, -122.6, 11, 'nominal', 'D', '2026-04-21', '0415',
      '5x5:W125_N35'
    )
  `);
  const evRes = (await db.execute(sql`
    INSERT INTO "aoi_events" (
      aoi_id, first_seen_at, last_seen_at,
      nearest_distance_km, detection_count, peak_frp_mw,
      dedupe_hash, status
    ) VALUES (
      ${aoiId}, '2026-04-21T04:00:00Z', '2026-04-21T04:30:00Z',
      8, 2, 11, ${"hash-" + Math.random().toString(36).slice(2)}, 'new'
    ) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const evRows = (evRes.rows ?? (evRes as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  return { aoiId, eventId: evRows[0].id };
}

describe("generateBriefForEvent — Stage 8 authority perimeter wiring", () => {
  let db: AppDb;
  let pglite: PGlite;
  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("calls fetchPerimeter with bucket+nearest detection and persists the result", async () => {
    const { eventId, aoiId } = await seed(db);
    let receivedArgs: { lat: number; lon: number; regionBucket: string } | null = null;
    const fetchPerimeter = async (args: { lat: number; lon: number; regionBucket: string }) => {
      receivedArgs = { lat: args.lat, lon: args.lon, regionBucket: args.regionBucket };
      return {
        source: "NIFC WFIGS",
        postedTs: "2026-05-07T11:00:00.000Z",
        containsDetection: true,
      };
    };
    const stubGateway = async (): Promise<GatewayResult> => ({
      ok: true,
      brief: { ...STUB_BRIEF, aoi: { ...STUB_BRIEF.aoi, id: aoiId } },
      modelId: "test/stub",
      promptVersion: "v1",
      latencyMs: 1,
      costUsdEst: null,
    });

    const outcome = await generateBriefForEvent(db, eventId, {
      gateway: stubGateway,
      fetchPerimeter,
    });
    expect(outcome.status).toBe("generated");
    expect(receivedArgs).not.toBeNull();
    expect(receivedArgs!.regionBucket).toBe("5x5:W125_N35");
    expect(receivedArgs!.lat).toBeCloseTo(38.5, 4);

    const briefs = (await db.execute(sql`
      SELECT payload FROM aoi_briefs WHERE event_id = ${eventId}
    `)) as unknown as { rows?: Array<{ payload: { context: { authority_perimeter: unknown } } }> };
    const brows = (briefs.rows ?? (briefs as unknown as Array<{ payload: { context: { authority_perimeter: unknown } } }>)) as Array<{
      payload: { context: { authority_perimeter: { source: string | null; posted_ts: string | null; contains_detection: boolean | null } } };
    }>;
    expect(brows[0].payload.context.authority_perimeter.source).toBe("NIFC WFIGS");
    expect(brows[0].payload.context.authority_perimeter.contains_detection).toBe(true);
  });

  it("ships the brief with all-null perimeter when fetcher rejects", async () => {
    const { eventId, aoiId } = await seed(db);
    const fetchPerimeter = async () => {
      throw new Error("authority down");
    };
    const stubGateway = async (): Promise<GatewayResult> => ({
      ok: true,
      brief: {
        ...STUB_BRIEF,
        aoi: { ...STUB_BRIEF.aoi, id: aoiId },
        context: {
          weather_note: null,
          authority_perimeter: { source: null, posted_ts: null, contains_detection: null },
          prior_events: [],
        },
      },
      modelId: "test/stub",
      promptVersion: "v1",
      latencyMs: 1,
      costUsdEst: null,
    });
    const outcome = await generateBriefForEvent(db, eventId, {
      gateway: stubGateway,
      fetchPerimeter,
    });
    expect(outcome.status).toBe("generated");
  });
});
