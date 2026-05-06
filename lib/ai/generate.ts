/**
 * Stage 3 brief-generation orchestrator.
 *
 * Single entry point: `generateBriefForEvent(db, eventId)`.
 *
 * Steps:
 *   1. Load AOI + event + rules + last-brief timestamp.
 *   2. Run the pure-TS gate (`evaluateGate`).
 *   3. If pass: build prompt, call AI Gateway via `generateBriefViaGateway`.
 *   4. Re-validate the returned object against `BriefSchema` (defence in depth).
 *   5. Render markdown.
 *   6. Transactional INSERT into `aoi_briefs`, UPDATE `aoi_events.last_brief_at`.
 *
 * Returns a discriminated outcome the caller (`/api/aoi/poll`) can record on
 * the per-bucket job_run row.
 *
 * Two-backend repository pattern: the SQL touched here (`aoi_events`,
 * `aoi_briefs`, `aoi_rules`, `aois`) is non-spatial and runs identically on
 * Neon+PostGIS and PGlite.
 */
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";
import { evaluateGate, type GateInputs, type GateReason } from "./gate";
import { buildUserPrompt, SYSTEM_PROMPT, type BriefContext } from "./prompt";
import {
  generateBriefViaGateway,
  DEFAULT_MODEL_ID,
  type GatewayResult,
} from "./gateway";
import { renderBriefMarkdown } from "./render";
import { BriefSchema, type Brief } from "./schema";

export type GenerateOutcome =
  | { status: "generated"; eventId: string; briefId: string; modelId: string; gateReason: GateReason }
  | { status: "skipped"; eventId: string; reason: GateReason | "config_missing" | "event_not_found" }
  | { status: "error"; eventId: string; reason: string };

type GeneratorDeps = {
  /**
   * Override the gateway call for tests/integration. Production injects null
   * and the orchestrator calls the real `generateBriefViaGateway`.
   */
  gateway?: (args: {
    systemPrompt: string;
    userPrompt: string;
    modelId?: string;
  }) => Promise<GatewayResult>;
  /** Override clock for deterministic testing of the gate window. */
  now?: Date;
};

const WINDOW_HOURS_DEFAULT = 24;

export async function generateBriefForEvent(
  db: AppDb,
  eventId: string,
  deps: GeneratorDeps = {},
): Promise<GenerateOutcome> {
  const loaded = await loadEventContext(db, eventId);
  if (!loaded) {
    return { status: "skipped", eventId, reason: "event_not_found" };
  }

  const gate = evaluateGate({
    pausedUntil: loaded.pausedUntil,
    lastBriefAt: loaded.lastBriefAt,
    lastAoiEventBriefedAt: loaded.lastAoiEventBriefedAt,
    detectionCountInEvent: loaded.detectionCount,
    peakFrpMw: loaded.peakFrpMw,
    nearestDistanceKm: loaded.nearestDistanceKm,
    alertDistanceKm: loaded.alertDistanceKm,
    minFrpMw: loaded.minFrpMw,
    now: deps.now,
  } satisfies GateInputs);

  if (!gate.pass) {
    return { status: "skipped", eventId, reason: gate.reason };
  }

  const ctx: BriefContext = {
    aoi: { id: loaded.aoiId, name: loaded.aoiName, areaHa: loaded.aoiAreaHa },
    event: {
      nearestDistanceKm: loaded.nearestDistanceKm,
      bearingFromAoiDeg: loaded.bearingFromAoiDeg ?? 0,
      detectionCount: loaded.detectionCount,
      peakFrpMw: loaded.peakFrpMw,
      windowHours: WINDOW_HOURS_DEFAULT,
      satellites: loaded.satellites,
      firstSeenAt: loaded.firstSeenAt.toISOString(),
      lastSeenAt: loaded.lastSeenAt.toISOString(),
    },
    weather: null,
    authorityPerimeter: null,
    priorEvents: loaded.priorEvents,
  };

  const userPrompt = buildUserPrompt(ctx);
  const gw = (deps.gateway ?? generateBriefViaGateway);
  const result = await gw({
    systemPrompt: SYSTEM_PROMPT,
    userPrompt,
    modelId: DEFAULT_MODEL_ID,
  });

  if (!result.ok) {
    if (result.code === "config_missing") {
      return { status: "skipped", eventId, reason: "config_missing" };
    }
    return { status: "error", eventId, reason: `${result.code}: ${result.message}` };
  }

  const reparsed = BriefSchema.safeParse(result.brief);
  if (!reparsed.success) {
    return {
      status: "error",
      eventId,
      reason: `schema_invalid: ${reparsed.error.issues
        .map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`)
        .slice(0, 3)
        .join("; ")}`,
    };
  }
  const brief: Brief = reparsed.data;
  const markdown = renderBriefMarkdown(brief);

  const briefId = await persistBrief(db, {
    aoiId: loaded.aoiId,
    eventId,
    model: result.modelId,
    gateReason: gate.reason,
    payload: brief,
    rendered: markdown,
  });

  return {
    status: "generated",
    eventId,
    briefId,
    modelId: result.modelId,
    gateReason: gate.reason,
  };
}

// ---------------------------------------------------------------------------
// DB load / persist

type LoadedContext = {
  aoiId: string;
  aoiName: string;
  aoiAreaHa: number;
  pausedUntil: Date | null;
  alertDistanceKm: number;
  minFrpMw: number;
  detectionCount: number;
  peakFrpMw: number | null;
  nearestDistanceKm: number;
  bearingFromAoiDeg: number | null;
  firstSeenAt: Date;
  lastSeenAt: Date;
  lastBriefAt: Date | null;
  lastAoiEventBriefedAt: Date | null;
  satellites: string[];
  priorEvents: Array<{ date: string; description: string; outcome: string | null }>;
};

async function loadEventContext(
  db: AppDb,
  eventId: string,
): Promise<LoadedContext | null> {
  const result = (await db.execute(sql`
    SELECT
      e."id"                    AS event_id,
      e."aoi_id"                AS aoi_id,
      e."detection_count"       AS detection_count,
      e."peak_frp_mw"           AS peak_frp_mw,
      e."nearest_distance_km"   AS nearest_distance_km,
      e."first_seen_at"         AS first_seen_at,
      e."last_seen_at"          AS last_seen_at,
      e."last_brief_at"         AS last_brief_at,
      a."name"                  AS aoi_name,
      a."area_ha"               AS aoi_area_ha,
      r."distance_buffer_km"    AS alert_distance_km,
      r."min_frp_mw"            AS min_frp_mw,
      r."paused_until"          AS paused_until
    FROM "aoi_events" e
    JOIN "aois" a       ON a."id" = e."aoi_id"
    LEFT JOIN "aoi_rules" r ON r."aoi_id" = e."aoi_id"
    WHERE e."id" = ${eventId}
    LIMIT 1
  `)) as unknown as {
    rows?: Array<Record<string, unknown>>;
  };
  const rows = (result.rows ?? (result as unknown as Array<Record<string, unknown>>)) as Array<Record<string, unknown>>;
  const row = rows[0];
  if (!row) return null;

  const lastAoiBrief = await fetchLastAoiBriefedAt(db, String(row.aoi_id), eventId);
  const satellites = await fetchEventSatellites(db, String(row.aoi_id));
  const priorEvents = await fetchPriorEvents(db, String(row.aoi_id), eventId);

  return {
    aoiId: String(row.aoi_id),
    aoiName: String(row.aoi_name),
    aoiAreaHa: toNumber(row.aoi_area_ha) ?? 0,
    pausedUntil: toDate(row.paused_until),
    alertDistanceKm: toNumber(row.alert_distance_km) ?? 25,
    minFrpMw: toNumber(row.min_frp_mw) ?? 5,
    detectionCount: Number(row.detection_count ?? 0),
    peakFrpMw: toNumber(row.peak_frp_mw),
    nearestDistanceKm: toNumber(row.nearest_distance_km) ?? 0,
    bearingFromAoiDeg: null,
    firstSeenAt: toDate(row.first_seen_at) ?? new Date(0),
    lastSeenAt: toDate(row.last_seen_at) ?? new Date(0),
    lastBriefAt: toDate(row.last_brief_at),
    lastAoiEventBriefedAt: lastAoiBrief,
    satellites,
    priorEvents,
  };
}

async function fetchLastAoiBriefedAt(
  db: AppDb,
  aoiId: string,
  excludingEventId: string,
): Promise<Date | null> {
  const result = (await db.execute(sql`
    SELECT MAX(e."last_brief_at") AS t
    FROM "aoi_events" e
    WHERE e."aoi_id" = ${aoiId}
      AND e."id" <> ${excludingEventId}
      AND e."last_brief_at" IS NOT NULL
  `)) as unknown as { rows?: Array<{ t: Date | string | null }> };
  const rows = (result.rows ?? (result as unknown as Array<{ t: Date | string | null }>)) as Array<{
    t: Date | string | null;
  }>;
  const v = rows[0]?.t ?? null;
  return toDate(v);
}

async function fetchEventSatellites(db: AppDb, aoiId: string): Promise<string[]> {
  // Distinct satellite sources observed in this AOI's recent (24h) detections.
  // Used as the `satellites` list on the brief. We pick from the AOI's region
  // bucket so the query stays cheap.
  const result = (await db.execute(sql`
    SELECT DISTINCT d."source" AS source
    FROM "firms_detections" d
    JOIN "aois" a ON a."region_bucket" = d."bucket"
    WHERE a."id" = ${aoiId}
      AND d."detected_at" >= now() - INTERVAL '24 hours'
    ORDER BY 1
  `)) as unknown as { rows?: Array<{ source: string }> };
  const rows = (result.rows ?? (result as unknown as Array<{ source: string }>)) as Array<{
    source: string;
  }>;
  const out = rows.map((r) => r.source).filter(Boolean);
  return out.length > 0 ? out : ["VIIRS_NOAA20_NRT"];
}

async function fetchPriorEvents(
  db: AppDb,
  aoiId: string,
  excludingEventId: string,
): Promise<Array<{ date: string; description: string; outcome: string | null }>> {
  const result = (await db.execute(sql`
    SELECT
      e."first_seen_at" AS first_seen_at,
      e."detection_count" AS detection_count,
      e."peak_frp_mw" AS peak_frp_mw,
      e."nearest_distance_km" AS nearest_distance_km
    FROM "aoi_events" e
    WHERE e."aoi_id" = ${aoiId}
      AND e."id" <> ${excludingEventId}
    ORDER BY e."first_seen_at" DESC
    LIMIT 5
  `)) as unknown as {
    rows?: Array<{
      first_seen_at: Date | string;
      detection_count: number;
      peak_frp_mw: number | null;
      nearest_distance_km: number;
    }>;
  };
  const rows = (result.rows ?? (result as unknown as Array<{
    first_seen_at: Date | string;
    detection_count: number;
    peak_frp_mw: number | null;
    nearest_distance_km: number;
  }>)) as Array<{
    first_seen_at: Date | string;
    detection_count: number;
    peak_frp_mw: number | null;
    nearest_distance_km: number;
  }>;
  return rows.map((r) => {
    const d = r.first_seen_at instanceof Date ? r.first_seen_at : new Date(r.first_seen_at);
    const date = d.toISOString().slice(0, 10);
    const dist = Number(r.nearest_distance_km).toFixed(1);
    const peak = r.peak_frp_mw == null ? "—" : `${Number(r.peak_frp_mw).toFixed(1)} MW`;
    return {
      date,
      description: `${r.detection_count} detection(s) at ${dist} km, peak FRP ${peak}.`,
      outcome: null,
    };
  });
}

type PersistArgs = {
  aoiId: string;
  eventId: string;
  model: string;
  gateReason: GateReason;
  payload: Brief;
  rendered: string;
};

async function persistBrief(db: AppDb, args: PersistArgs): Promise<string> {
  // Single statement INSERT with conflict guard, then UPDATE the parent
  // event's `last_brief_at`. The unique index on `aoi_briefs.event_id` makes
  // the INSERT idempotent if the same eventId is processed twice.
  const inserted = (await db.execute(sql`
    INSERT INTO "aoi_briefs" (
      "aoi_id", "event_id", "schema_version", "model", "gate_reason",
      "payload", "rendered_markdown"
    ) VALUES (
      ${args.aoiId},
      ${args.eventId},
      ${args.payload.schema_version},
      ${args.model},
      ${args.gateReason},
      ${JSON.stringify(args.payload)}::jsonb,
      ${args.rendered}
    )
    ON CONFLICT ("event_id") DO NOTHING
    RETURNING "id"
  `)) as unknown as { rows?: Array<{ id: string }> };
  const rows = (inserted.rows ?? (inserted as unknown as Array<{ id: string }>)) as Array<{
    id: string;
  }>;
  const id = rows[0]?.id;
  if (!id) {
    // Already inserted by a parallel run — fetch it.
    const existing = (await db.execute(sql`
      SELECT "id" FROM "aoi_briefs" WHERE "event_id" = ${args.eventId} LIMIT 1
    `)) as unknown as { rows?: Array<{ id: string }> };
    const erows = (existing.rows ?? (existing as unknown as Array<{ id: string }>)) as Array<{
      id: string;
    }>;
    return erows[0]?.id ?? "";
  }

  await db.execute(sql`
    UPDATE "aoi_events"
    SET "last_brief_at" = COALESCE("last_brief_at", now())
    WHERE "id" = ${args.eventId}
  `);
  return id;
}

// ---------------------------------------------------------------------------
// Helpers

function toDate(v: unknown): Date | null {
  if (v == null) return null;
  if (v instanceof Date) return v;
  if (typeof v === "string" || typeof v === "number") {
    const d = new Date(v);
    return Number.isFinite(d.getTime()) ? d : null;
  }
  return null;
}

function toNumber(v: unknown): number | null {
  if (v == null) return null;
  if (typeof v === "number") return v;
  if (typeof v === "string") {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}
