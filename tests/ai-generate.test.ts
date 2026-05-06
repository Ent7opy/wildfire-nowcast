/**
 * `generateBriefForEvent` PGlite pipeline test.
 *
 * Seeds an AOI + rules + a "new" aoi_event row directly via SQL (we don't need
 * the PostGIS spatial path here — only the generator's load → gate → persist
 * pipeline). Stubs the AI Gateway so the test never hits the network.
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
  aoi: {
    id: "00000000-0000-4000-8000-000000000099",
    name: "Test Preserve",
    area_ha: 100,
  },
  summary: "Stubbed brief for pipeline test (fixture).",
  key_facts: {
    nearest_detection_km: 8,
    bearing_from_aoi_deg: 90,
    wind_dir_deg: null,
    wind_speed_kmh: null,
    wind_toward_aoi: null,
    detection_count_in_window: 2,
    max_frp_mw: 12,
    satellites: ["VIIRS_NOAA20_NRT"],
    window_hours: 24,
  },
  context: {
    weather_note: null,
    authority_perimeter: { source: null, posted_ts: null, contains_detection: null },
    prior_events: [],
  },
  recommended_watch_items: ["Re-check at next cron tick."],
  uncertainty: "Stubbed; not a real reading.",
  next_brief_hint: { when: "next tick", trigger: "any new detection" },
};

async function seedAoiAndEvent(
  db: AppDb,
  opts: { detectionCount?: number; nearestKm?: number; peakFrp?: number | null; lastBriefAt?: Date | null } = {},
): Promise<{ aoiId: string; eventId: string }> {
  const aoiPolygon = JSON.stringify({
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
  const aoiRes = (await db.execute(sql`
    INSERT INTO "aois" (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
    VALUES (
      'stub-user-1',
      'Test Preserve',
      ${aoiPolygon},
      ${aoiPolygon},
      ${JSON.stringify({ type: "Point", coordinates: [-122.65, 38.45] })},
      '5x5:W125_N35',
      100
    )
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const aoiRows = (aoiRes.rows ?? (aoiRes as unknown as Array<{ id: string }>)) as Array<{
    id: string;
  }>;
  const aoiId = aoiRows[0].id;

  await db.execute(sql`
    INSERT INTO "aoi_rules" (aoi_id, distance_buffer_km, min_confidence, min_frp_mw)
    VALUES (${aoiId}, 25, 'nominal', 5)
  `);

  const evRes = (await db.execute(sql`
    INSERT INTO "aoi_events" (
      aoi_id, first_seen_at, last_seen_at,
      nearest_distance_km, detection_count, peak_frp_mw,
      dedupe_hash, status, last_brief_at
    ) VALUES (
      ${aoiId},
      '2026-04-21T04:00:00Z',
      '2026-04-21T04:30:00Z',
      ${opts.nearestKm ?? 8},
      ${opts.detectionCount ?? 2},
      ${opts.peakFrp ?? 11},
      ${"hash-" + Math.random().toString(36).slice(2, 10)},
      'new',
      ${opts.lastBriefAt ? opts.lastBriefAt.toISOString() : null}
    )
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const evRows = (evRes.rows ?? (evRes as unknown as Array<{ id: string }>)) as Array<{
    id: string;
  }>;
  return { aoiId, eventId: evRows[0].id };
}

describe("generateBriefForEvent — PGlite pipeline", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });

  afterEach(async () => {
    await pglite.close();
  });

  it("generates a brief, persists it, and stamps last_brief_at", async () => {
    const { aoiId, eventId } = await seedAoiAndEvent(db, {
      detectionCount: 2,
      nearestKm: 8,
      peakFrp: 11,
    });

    const stubGateway = async (): Promise<GatewayResult> => ({
      ok: true,
      brief: { ...STUB_BRIEF, aoi: { ...STUB_BRIEF.aoi, id: aoiId } },
      modelId: "test/stub-model",
      promptVersion: "v1",
      latencyMs: 12,
      costUsdEst: null,
    });

    const outcome = await generateBriefForEvent(db, eventId, {
      gateway: stubGateway,
    });
    expect(outcome.status).toBe("generated");
    if (outcome.status !== "generated") return;
    expect(outcome.gateReason).toBe("prior_absence");
    expect(outcome.modelId).toBe("test/stub-model");

    const briefs = (await db.execute(sql`
      SELECT id, payload, rendered_markdown, gate_reason FROM aoi_briefs WHERE event_id = ${eventId}
    `)) as unknown as { rows?: Array<{ id: string; payload: unknown; rendered_markdown: string; gate_reason: string }> };
    const brows = (briefs.rows ?? (briefs as unknown as Array<{ id: string; payload: unknown; rendered_markdown: string; gate_reason: string }>)) as Array<{
      id: string;
      payload: unknown;
      rendered_markdown: string;
      gate_reason: string;
    }>;
    expect(brows).toHaveLength(1);
    expect(brows[0].rendered_markdown).toContain("Test Preserve");
    expect(brows[0].gate_reason).toBe("prior_absence");

    const ev = (await db.execute(sql`
      SELECT last_brief_at FROM aoi_events WHERE id = ${eventId}
    `)) as unknown as { rows?: Array<{ last_brief_at: Date | string | null }> };
    const evRows = (ev.rows ?? (ev as unknown as Array<{ last_brief_at: Date | string | null }>)) as Array<{
      last_brief_at: Date | string | null;
    }>;
    expect(evRows[0].last_brief_at).not.toBeNull();
  });

  it("skips with reason=already_briefed when last_brief_at is set", async () => {
    const { eventId } = await seedAoiAndEvent(db, {
      lastBriefAt: new Date("2026-04-21T04:31:00Z"),
    });
    let called = false;
    const stub = async (): Promise<GatewayResult> => {
      called = true;
      return { ok: false, code: "upstream_error", message: "should not run" };
    };
    const outcome = await generateBriefForEvent(db, eventId, { gateway: stub });
    expect(outcome.status).toBe("skipped");
    if (outcome.status === "skipped") {
      expect(outcome.reason).toBe("already_briefed");
    }
    expect(called).toBe(false);
  });

  it("skips with reason=config_missing when AI_GATEWAY_API_KEY is unset", async () => {
    const { eventId } = await seedAoiAndEvent(db);
    const stub = async (): Promise<GatewayResult> => ({
      ok: false,
      code: "config_missing",
      message: "no key",
    });
    const outcome = await generateBriefForEvent(db, eventId, { gateway: stub });
    expect(outcome.status).toBe("skipped");
    if (outcome.status === "skipped") {
      expect(outcome.reason).toBe("config_missing");
    }
  });

  it("returns error when the gateway response is schema-invalid", async () => {
    const { eventId } = await seedAoiAndEvent(db);
    const stub = async (): Promise<GatewayResult> => ({
      ok: true,
      // @ts-expect-error -- intentionally malformed payload to test re-validation
      brief: { schema_version: 1, aoi: { id: "not-a-uuid" } },
      modelId: "test/stub",
      promptVersion: "v1",
      latencyMs: 5,
      costUsdEst: null,
    });
    const outcome = await generateBriefForEvent(db, eventId, { gateway: stub });
    expect(outcome.status).toBe("error");
  });

  it("returns skipped/event_not_found for an unknown event id", async () => {
    const outcome = await generateBriefForEvent(
      db,
      "00000000-0000-4000-8000-000000000000",
    );
    expect(outcome.status).toBe("skipped");
    if (outcome.status === "skipped") {
      expect(outcome.reason).toBe("event_not_found");
    }
  });
});
