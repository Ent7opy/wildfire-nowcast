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
import {
  fetchAuthorityPerimeter,
  type AuthorityPerimeter,
  type FetchPerimeterArgs,
} from "./authority/fetch";

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
  /**
   * Override the authority-perimeter fetch (Stage 8). Production injects null
   * and the orchestrator calls the real `fetchAuthorityPerimeter`. Tests can
   * stub a happy or rejecting impl; the brief still ships either way.
   */
  fetchPerimeter?: (args: FetchPerimeterArgs) => Promise<AuthorityPerimeter | null>;
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
      bearingFromAoiDeg: loaded.bearingFromAoiDeg,
      detectionCount: loaded.detectionCount,
      peakFrpMw: loaded.peakFrpMw,
      windowHours: WINDOW_HOURS_DEFAULT,
      satellites: loaded.satellites,
      firstSeenAt: loaded.firstSeenAt.toISOString(),
      lastSeenAt: loaded.lastSeenAt.toISOString(),
    },
    weather: null,
    authorityPerimeter: await gatherAuthorityPerimeter(loaded, deps),
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
    promptVersion: result.promptVersion,
    gateReason: gate.reason,
    payload: brief,
    rendered: markdown,
    latencyMs: result.latencyMs,
    costUsdEst: result.costUsdEst,
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
// Stage 8 — authority perimeter gather (Path A: orchestrator pre-fetch).

async function gatherAuthorityPerimeter(
  loaded: LoadedContext,
  deps: GeneratorDeps,
): Promise<{
  source: string | null;
  postedTs: string | null;
  containsDetection: boolean | null;
} | null> {
  if (!loaded.nearestDetection || !loaded.regionBucket) return null;
  const fetcher = deps.fetchPerimeter ?? fetchAuthorityPerimeter;
  let r: AuthorityPerimeter | null;
  try {
    r = await fetcher({
      lat: loaded.nearestDetection.lat,
      lon: loaded.nearestDetection.lon,
      regionBucket: loaded.regionBucket,
    });
  } catch (err) {
    // Build-without-blocking — never let an authority fetch failure abort
    // brief generation. Logged; the brief ships with all-null perimeter.
    console.warn(`[authority] fetcher threw: ${err instanceof Error ? err.message : String(err)}`);
    return null;
  }
  if (!r) return null;
  return {
    source: r.source,
    postedTs: r.postedTs,
    containsDetection: r.containsDetection,
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
  regionBucket: string;
  /** Lat/lon of the nearest non-industrial detection in the event window, when known. */
  nearestDetection: { lat: number; lon: number } | null;
};

async function loadEventContext(
  db: AppDb,
  eventId: string,
): Promise<LoadedContext | null> {
  // Centroid is `geometry(Point,4326)` on Neon and a GeoJSON TEXT column on
  // PGlite — branch on the backend flag the same way `lib/firms/matcher.ts`
  // does so the bearing math has lon/lat in both worlds.
  const centroidExpr = db.usePostGIS
    ? sql`ST_X(a."centroid"::geometry) AS centroid_lon, ST_Y(a."centroid"::geometry) AS centroid_lat`
    : sql`a."centroid" AS centroid_geojson`;
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
      a."region_bucket"         AS region_bucket,
      ${centroidExpr},
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

  const aoiId = String(row.aoi_id);
  const regionBucket = String(row.region_bucket ?? "");
  const firstSeenAt = toDate(row.first_seen_at) ?? new Date(0);
  const lastSeenAt = toDate(row.last_seen_at) ?? new Date(0);

  let centroidLon: number | null = null;
  let centroidLat: number | null = null;
  if (db.usePostGIS) {
    centroidLon = toNumber(row.centroid_lon);
    centroidLat = toNumber(row.centroid_lat);
  } else {
    const raw = row.centroid_geojson;
    if (typeof raw === "string") {
      try {
        const parsed = JSON.parse(raw) as { type: string; coordinates: [number, number] };
        if (parsed?.type === "Point" && Array.isArray(parsed.coordinates)) {
          centroidLon = Number(parsed.coordinates[0]);
          centroidLat = Number(parsed.coordinates[1]);
          if (!Number.isFinite(centroidLon) || !Number.isFinite(centroidLat)) {
            centroidLon = null;
            centroidLat = null;
          }
        }
      } catch {
        // Centroid unparseable — leave null; bearing stays null (no fabrication).
      }
    }
  }

  const lastAoiBrief = await fetchLastAoiBriefedAt(db, aoiId, eventId);
  const nearestDetection = await fetchNearestDetection(db, {
    regionBucket,
    firstSeenAt,
    lastSeenAt,
    centroidLon,
    centroidLat,
  });
  const satellites = await fetchEventSatellites(db, {
    regionBucket,
    firstSeenAt,
    lastSeenAt,
  });
  const priorEvents = await fetchPriorEvents(db, aoiId, eventId);

  const bearing =
    centroidLon != null && centroidLat != null && nearestDetection
      ? bearingDeg(centroidLat, centroidLon, nearestDetection.lat, nearestDetection.lon)
      : null;

  return {
    aoiId,
    aoiName: String(row.aoi_name),
    aoiAreaHa: toNumber(row.aoi_area_ha) ?? 0,
    pausedUntil: toDate(row.paused_until),
    alertDistanceKm: toNumber(row.alert_distance_km) ?? 25,
    minFrpMw: toNumber(row.min_frp_mw) ?? 5,
    detectionCount: Number(row.detection_count ?? 0),
    peakFrpMw: toNumber(row.peak_frp_mw),
    nearestDistanceKm: toNumber(row.nearest_distance_km) ?? 0,
    bearingFromAoiDeg: bearing,
    firstSeenAt,
    lastSeenAt,
    lastBriefAt: toDate(row.last_brief_at),
    lastAoiEventBriefedAt: lastAoiBrief,
    satellites,
    priorEvents,
    regionBucket,
    nearestDetection,
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

async function fetchEventSatellites(
  db: AppDb,
  args: { regionBucket: string; firstSeenAt: Date; lastSeenAt: Date },
): Promise<string[]> {
  // Distinct satellite sources for detections that fall inside this event's
  // own time window in the AOI's region bucket. Detections are not directly
  // joined to events, so window+bucket is the tightest scope we have without
  // a spatial filter (the AOI bucket is 5°×5°; same bucket is the matcher's
  // unit of work). No 24h fallback — we never invent satellites.
  const result = (await db.execute(sql`
    SELECT DISTINCT d."source" AS source
    FROM "firms_detections" d
    WHERE d."bucket" = ${args.regionBucket}
      AND d."detected_at" >= ${args.firstSeenAt.toISOString()}::timestamptz
      AND d."detected_at" <= ${args.lastSeenAt.toISOString()}::timestamptz
    ORDER BY 1
  `)) as unknown as { rows?: Array<{ source: string }> };
  const rows = (result.rows ?? (result as unknown as Array<{ source: string }>)) as Array<{
    source: string;
  }>;
  return rows.map((r) => r.source).filter(Boolean);
}

async function fetchNearestDetection(
  db: AppDb,
  args: {
    regionBucket: string;
    firstSeenAt: Date;
    lastSeenAt: Date;
    centroidLon: number | null;
    centroidLat: number | null;
  },
): Promise<{ lat: number; lon: number } | null> {
  if (args.centroidLon == null || args.centroidLat == null) return null;
  if (!args.regionBucket) return null;
  const result = (await db.execute(sql`
    SELECT d."lat" AS lat, d."lon" AS lon
    FROM "firms_detections" d
    WHERE d."bucket" = ${args.regionBucket}
      AND d."detected_at" >= ${args.firstSeenAt.toISOString()}::timestamptz
      AND d."detected_at" <= ${args.lastSeenAt.toISOString()}::timestamptz
      AND (d."is_industrial_static" IS NULL OR d."is_industrial_static" = FALSE)
  `)) as unknown as { rows?: Array<{ lat: number | string; lon: number | string }> };
  const rows = (result.rows ?? (result as unknown as Array<{ lat: number | string; lon: number | string }>)) as Array<{
    lat: number | string;
    lon: number | string;
  }>;
  let best: { lat: number; lon: number; d: number } | null = null;
  for (const r of rows) {
    const lat = Number(r.lat);
    const lon = Number(r.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const d = haversineKm(args.centroidLat, args.centroidLon, lat, lon);
    if (!best || d < best.d) best = { lat, lon, d };
  }
  return best ? { lat: best.lat, lon: best.lon } : null;
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
    LIMIT 3
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
  promptVersion: string;
  gateReason: GateReason;
  payload: Brief;
  rendered: string;
  latencyMs: number | null;
  costUsdEst: number | null;
};

async function persistBrief(db: AppDb, args: PersistArgs): Promise<string> {
  // Brief 17 §"Brief-generation pipeline" step 7: INSERT into aoi_briefs +
  // UPDATE aoi_events.last_brief_at must be atomic. Both Drizzle backends
  // (node-postgres and PGlite) expose `db.transaction(tx => ...)` with the
  // same shape, so the same code runs on Neon and the unit-test PGlite.
  // The unique index on aoi_briefs.event_id keeps INSERT idempotent across
  // parallel runs.
  return await db.transaction(async (tx) => {
    const inserted = (await tx.execute(sql`
      INSERT INTO "aoi_briefs" (
        "aoi_id", "event_id", "schema_version", "model", "prompt_version",
        "gate_reason", "payload", "rendered_markdown",
        "cost_usd_est", "latency_ms"
      ) VALUES (
        ${args.aoiId},
        ${args.eventId},
        ${args.payload.schema_version},
        ${args.model},
        ${args.promptVersion},
        ${args.gateReason},
        ${JSON.stringify(args.payload)}::jsonb,
        ${args.rendered},
        ${args.costUsdEst},
        ${args.latencyMs}
      )
      ON CONFLICT ("event_id") DO NOTHING
      RETURNING "id"
    `)) as unknown as { rows?: Array<{ id: string }> };
    const rows = (inserted.rows ?? (inserted as unknown as Array<{ id: string }>)) as Array<{
      id: string;
    }>;
    const id = rows[0]?.id;
    if (!id) {
      const existing = (await tx.execute(sql`
        SELECT "id" FROM "aoi_briefs" WHERE "event_id" = ${args.eventId} LIMIT 1
      `)) as unknown as { rows?: Array<{ id: string }> };
      const erows = (existing.rows ?? (existing as unknown as Array<{ id: string }>)) as Array<{
        id: string;
      }>;
      return erows[0]?.id ?? "";
    }

    await tx.execute(sql`
      UPDATE "aoi_events"
      SET "last_brief_at" = COALESCE("last_brief_at", now())
      WHERE "id" = ${args.eventId}
    `);
    return id;
  });
}

// ---------------------------------------------------------------------------
// Geometry helpers — kept local to avoid pulling turf for two formulas.

function toRad(deg: number): number {
  return (deg * Math.PI) / 180;
}

function toDeg(rad: number): number {
  return (rad * 180) / Math.PI;
}

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371; // mean Earth radius in km
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
}

function bearingDeg(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const phi1 = toRad(lat1);
  const phi2 = toRad(lat2);
  const dLambda = toRad(lon2 - lon1);
  const y = Math.sin(dLambda) * Math.cos(phi2);
  const x =
    Math.cos(phi1) * Math.sin(phi2) -
    Math.sin(phi1) * Math.cos(phi2) * Math.cos(dLambda);
  const theta = Math.atan2(y, x);
  return (toDeg(theta) + 360) % 360;
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
