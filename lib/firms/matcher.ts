/**
 * FIRMS detection → AOI matcher.
 *
 * Inputs:
 *   - `bucket` — the 5°×5° tile we just polled
 *   - `detections` — raw FIRMS rows from `fetchAreaCsv`
 *
 * Steps per call:
 *   1. Load the per-bucket AOI list (id, rules) once.
 *   2. Insert detections into `firms_detections` with ON CONFLICT DO NOTHING;
 *      `is_industrial_static` computed inline via ST_Intersects (PostGIS) or
 *      a point-in-bbox scan over the seed table (PGlite). The PGlite path is
 *      exact for the seed (axis-aligned boxes); see `pgliteIndustrialHit`.
 *   3. For each (AOI × non-industrial detection) within `distance_buffer_km`:
 *      compute the dedupe hash, UPSERT the event row — extend `last_seen_at`,
 *      bump `detection_count`, update `peak_frp_mw` if higher.
 *
 * Returns the counts the caller (`/api/aoi/poll`) records in `job_runs`.
 *
 * PostGIS path is the production path. The PGlite path exists solely for the
 * non-spatial unit tests that exercise the dedupe/update logic without
 * needing ST_DWithin. Real spatial coverage lives in
 * `tests/firms-matcher.integration.test.ts` against the testcontainer.
 */
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";
import { computeDedupeHash } from "./dedupe";
import type { FirmsDetection, FirmsSource } from "./client";
import { decodeRows } from "@/lib/db/decode-rows";

const MIN_CONFIDENCE_RANK: Record<string, number> = {
  low: 0,
  l: 0,
  nominal: 1,
  n: 1,
  high: 2,
  h: 2,
};

/**
 * Confidence gate for FIRMS detections.
 *
 * Today FIRMS confidence is one of two vocabularies: VIIRS letters
 * ("l"/"n"/"h") or MODIS integers (0..100). Anything outside that dichotomy
 * hits the unknown-token branch below.
 *
 * **Intentional fail-open contract**: unknown tokens are treated as
 * "nominal", not dropped. Reasoning — for a stewardship product, silently
 * dropping detections because NASA shipped a new vocabulary string is worse
 * than admitting a marginal row past the `nominal` gate. The user can see
 * and dismiss a borderline event; they cannot see one we never told them
 * about.
 *
 * **Test contract**: `tests/firms-matcher-internals.test.ts` pins this
 * behavior. If FIRMS introduces a new vocabulary that breaks the
 * digit/letter dichotomy (e.g. mixed alphanumeric codes, new categorical
 * labels), update BOTH this function AND that test together — extending
 * the parser is the correct fix; making the test pass against the existing
 * fail-open path is not.
 */
function confidencePassesGate(
  raw: string | null,
  aoiMin: string,
): boolean {
  if (raw == null) return aoiMin === "low";
  const norm = raw.trim().toLowerCase();
  // VIIRS confidence: "l" / "n" / "h". MODIS confidence: integer 0..100.
  if (/^\d+$/.test(norm)) {
    const pct = Number(norm);
    const level = pct >= 80 ? "high" : pct >= 30 ? "nominal" : "low";
    return (MIN_CONFIDENCE_RANK[level] ?? 0) >= (MIN_CONFIDENCE_RANK[aoiMin] ?? 0);
  }
  const rank = MIN_CONFIDENCE_RANK[norm];
  if (rank == null) {
    // Unknown token — fail open (count as "nominal"). See JSDoc above:
    // changing this also requires updating firms-matcher-internals.test.ts.
    return (MIN_CONFIDENCE_RANK["nominal"] ?? 1) >= (MIN_CONFIDENCE_RANK[aoiMin] ?? 0);
  }
  return rank >= (MIN_CONFIDENCE_RANK[aoiMin] ?? 0);
}

export type MatchArgs = {
  bucket: string;
  source: FirmsSource;
  detections: FirmsDetection[];
  /**
   * Stage 9: when set, restrict matching to ONLY these AOIs in the bucket.
   * Used by the first-AOI backfill to avoid creating events for other AOIs
   * that happen to share the bucket. When undefined/null, matches against
   * all active AOIs in the bucket (cron-poll behaviour, unchanged).
   */
  aoiIds?: string[];
};

export type MatchResult = {
  detectionsInserted: number;
  detectionsSkippedIndustrial: number;
  eventsCreated: number;
  eventsUpdated: number;
  /** Event IDs created in this match call — Stage 3 brief generator picks these up. */
  createdEventIds: string[];
};

export async function matchDetectionsToAois(
  db: AppDb,
  args: MatchArgs,
): Promise<MatchResult> {
  const result: MatchResult = {
    detectionsInserted: 0,
    detectionsSkippedIndustrial: 0,
    eventsCreated: 0,
    eventsUpdated: 0,
    createdEventIds: [],
  };
  if (args.detections.length === 0) return result;

  // Mark a poll-start timestamp so the match query considers only detections
  // freshly inserted in *this* poll. Without this, repeated polls with the
  // same FIRMS data would re-extend the same event row endlessly.
  // Use the DB's clock — the test container, the Vercel function, and Neon
  // can all disagree by seconds; the canonical clock for inserted_at is the
  // DB itself. We subtract 1 ms to guard against same-millisecond inserts.
  const pollStart = await dbNow(db);

  // 1. Insert detections with industrial flag. ON CONFLICT DO NOTHING means
  //    a re-poll with identical rows inserts zero rows — so the match step
  //    naturally finds zero new detections to consider.
  const inserted = await insertDetections(db, args);
  result.detectionsInserted = inserted.inserted;
  result.detectionsSkippedIndustrial = inserted.industrial;

  // 2. For each AOI in the bucket, find matching non-industrial detections
  //    *from this poll* and UPSERT events.
  const matches = await findAoiMatches(db, args.bucket, pollStart, args.aoiIds);
  for (const match of matches) {
    const outcome = await upsertEvent(db, args.bucket, args.source, match);
    if (outcome.kind === "created") {
      result.eventsCreated += 1;
      if (outcome.eventId) result.createdEventIds.push(outcome.eventId);
    } else if (outcome.kind === "updated") {
      result.eventsUpdated += 1;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Step 1: insert detections

type InsertedSummary = { inserted: number; industrial: number };

async function insertDetections(
  db: AppDb,
  args: MatchArgs,
): Promise<InsertedSummary> {
  let inserted = 0;
  let industrial = 0;
  for (const d of args.detections) {
    if (!Number.isFinite(d.latitude) || !Number.isFinite(d.longitude)) continue;
    const detectedAt = detectionTimestamp(d);
    if (!detectedAt) continue;

    const geomJson = {
      type: "Point" as const,
      coordinates: [d.longitude, d.latitude],
    };
    let isIndustrialExpr = db.usePostGIS
      ? sql`EXISTS (
          SELECT 1 FROM "industrial_mask_static" m
          WHERE ST_Intersects(m."geom", ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(geomJson)}), 4326))
        )`
      : sql`false`;
    if (!db.usePostGIS) {
      const hit = await pgliteIndustrialHit(db, d.longitude, d.latitude);
      isIndustrialExpr = sql`${hit}`;
    }

    const geomLit = db.usePostGIS
      ? sql`ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(geomJson)}), 4326)`
      : sql`${JSON.stringify(geomJson)}`;

    const insertResult = await db.execute(sql`
      INSERT INTO "firms_detections" (
        "source", "detected_at", "geom", "lat", "lon",
        "frp_mw", "confidence", "daynight",
        "acq_date", "acq_time",
        "bright_ti4", "bright_ti5", "scan", "track", "version",
        "is_industrial_static", "bucket"
      ) VALUES (
        ${args.source}, ${detectedAt.toISOString()}, ${geomLit}, ${d.latitude}, ${d.longitude},
        ${d.frp}, ${d.confidence}, ${d.daynight},
        ${d.acqDate}, ${d.acqTime},
        ${d.brightTi4}, ${d.brightTi5}, ${d.scan}, ${d.track}, ${d.version},
        ${isIndustrialExpr}, ${args.bucket}
      )
      ON CONFLICT ("source", "acq_date", "acq_time", "lat", "lon") DO NOTHING
      RETURNING "is_industrial_static"
    `);
    const rows = decodeRows<{ is_industrial_static: boolean | null }>(insertResult);
    if (rows.length > 0) {
      inserted += 1;
      if (rows[0].is_industrial_static === true) industrial += 1;
    }
  }
  return { inserted, industrial };
}

async function pgliteIndustrialHit(
  db: AppDb,
  lon: number,
  lat: number,
): Promise<boolean> {
  // PGlite fallback: iterate the (small) mask table and do point-in-bbox.
  // The seed polygons are axis-aligned boxes, so a bbox test is exact for
  // them; for the production path, ST_Intersects handles arbitrary polygons.
  const result = await db.execute(sql`
    SELECT "geom" FROM "industrial_mask_static"
  `);
  const rows = decodeRows<{ geom: string }>(result);
  for (const r of rows) {
    try {
      const g = JSON.parse(r.geom) as {
        type: string;
        coordinates: number[][][];
      };
      if (g.type !== "Polygon" || !g.coordinates[0]) continue;
      const ring = g.coordinates[0];
      let minLon = Infinity;
      let maxLon = -Infinity;
      let minLat = Infinity;
      let maxLat = -Infinity;
      for (const [rLon, rLat] of ring) {
        if (rLon < minLon) minLon = rLon;
        if (rLon > maxLon) maxLon = rLon;
        if (rLat < minLat) minLat = rLat;
        if (rLat > maxLat) maxLat = rLat;
      }
      if (lon >= minLon && lon <= maxLon && lat >= minLat && lat <= maxLat) {
        return true;
      }
    } catch {
      // Skip malformed row.
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Step 2: find matches and upsert events

type PerAoiMatch = {
  aoiId: string;
  nearestDistanceKm: number;
  nearestLat: number;
  nearestLon: number;
  peakFrpMw: number | null;
  firstSeenAt: Date;
  lastSeenAt: Date;
  confidence: string | null;
  minConfidence: string;
};

async function findAoiMatches(
  db: AppDb,
  bucket: string,
  pollStart: Date,
  aoiIds?: string[],
): Promise<PerAoiMatch[]> {
  if (!db.usePostGIS) {
    // PGlite does not expose ST_DWithin; the non-spatial tests cover the
    // UPSERT/dedupe logic directly by stubbing `findAoiMatches` via the
    // exported internals below. We return [] here to keep the production-
    // shaped API; spatial correctness is covered by the testcontainer tests.
    return [];
  }

  // Build an OR-of-equals filter rather than ANY($1::uuid[]) — node-postgres
  // serializes a JS array as a comma-joined string by default, which Postgres
  // rejects as a malformed array literal. The OR form is parameterized cleanly
  // for any reasonable aoiIds.length (single-AOI backfill is the only caller).
  const aoiFilter =
    aoiIds && aoiIds.length > 0
      ? sql` AND a."id" IN (${sql.join(
          aoiIds.map((id) => sql`${id}`),
          sql`, `,
        )})`
      : sql``;

  const result = await db.execute(sql`
    WITH active_detections AS (
      SELECT d."id", d."lat", d."lon", d."detected_at", d."frp_mw", d."confidence", d."geom"
      FROM "firms_detections" d
      WHERE d."bucket" = ${bucket}
        AND (d."is_industrial_static" IS NULL OR d."is_industrial_static" = FALSE)
        -- Only freshly-inserted rows from this poll are considered. ON
        -- CONFLICT DO NOTHING upstream means a repeat poll with the same
        -- FIRMS rows produces zero new inserts, so this filter rejects them.
        AND d."inserted_at" >= ${pollStart.toISOString()}::timestamptz
    ),
    active_aois AS (
      SELECT a."id" AS aoi_id, a."polygon", r."distance_buffer_km", r."min_confidence"
      FROM "aois" a
      LEFT JOIN "aoi_rules" r ON r."aoi_id" = a."id"
      WHERE a."archived_at" IS NULL
        AND a."region_bucket" = ${bucket}
        ${aoiFilter}
    ),
    matches AS (
      SELECT
        a.aoi_id,
        a.distance_buffer_km,
        a.min_confidence,
        d."id" AS det_id,
        d."lat" AS det_lat,
        d."lon" AS det_lon,
        d."detected_at",
        d."frp_mw",
        d."confidence",
        ST_Distance(a."polygon"::geography, d."geom"::geography) AS dist_m
      FROM active_aois a
      JOIN active_detections d
        ON ST_DWithin(
             a."polygon"::geography,
             d."geom"::geography,
             COALESCE(a."distance_buffer_km", 25) * 1000
           )
    ),
    ranked AS (
      SELECT
        aoi_id,
        min_confidence,
        det_lat,
        det_lon,
        detected_at,
        frp_mw,
        confidence,
        dist_m,
        ROW_NUMBER() OVER (PARTITION BY aoi_id ORDER BY dist_m ASC) AS rn
      FROM matches
    )
    SELECT
      aoi_id,
      min_confidence,
      (SELECT MIN(dist_m)  FROM matches m2 WHERE m2.aoi_id = r.aoi_id) AS nearest_m,
      (SELECT MAX(frp_mw)  FROM matches m2 WHERE m2.aoi_id = r.aoi_id) AS peak_frp,
      (SELECT MIN(detected_at) FROM matches m2 WHERE m2.aoi_id = r.aoi_id) AS first_seen,
      (SELECT MAX(detected_at) FROM matches m2 WHERE m2.aoi_id = r.aoi_id) AS last_seen,
      det_lat,
      det_lon,
      confidence
    FROM ranked r
    WHERE rn = 1
  `);
  const rows = decodeRows<{
    aoi_id: string;
    min_confidence: string | null;
    nearest_m: number | string | null;
    peak_frp: number | string | null;
    first_seen: string | Date;
    last_seen: string | Date;
    det_lat: number | string;
    det_lon: number | string;
    confidence: string | null;
  }>(result);

  const out: PerAoiMatch[] = [];
  for (const r of rows) {
    const minConfidence = (r.min_confidence ?? "nominal").toLowerCase();
    if (!confidencePassesGate(r.confidence, minConfidence)) continue;
    out.push({
      aoiId: r.aoi_id,
      nearestDistanceKm: Number(r.nearest_m ?? 0) / 1000,
      peakFrpMw: r.peak_frp == null ? null : Number(r.peak_frp),
      firstSeenAt: r.first_seen instanceof Date ? r.first_seen : new Date(r.first_seen),
      lastSeenAt: r.last_seen instanceof Date ? r.last_seen : new Date(r.last_seen),
      nearestLat: Number(r.det_lat),
      nearestLon: Number(r.det_lon),
      confidence: r.confidence,
      minConfidence,
    });
  }
  return out;
}

async function upsertEvent(
  db: AppDb,
  bucket: string,
  source: FirmsSource,
  match: PerAoiMatch,
): Promise<{ kind: "created" | "updated"; eventId?: string }> {
  const hash = computeDedupeHash({
    aoiId: match.aoiId,
    bucket,
    centroidLat: match.nearestLat,
    centroidLon: match.nearestLon,
    source,
    detectedAt: match.lastSeenAt,
  });

  // Try UPDATE first — if a "new"/"open" event with this hash exists, extend
  // it. Otherwise INSERT a new "new" event. Doing it as two statements keeps
  // the logic readable and avoids ON CONFLICT surprises between drivers.
  const existing = await db.execute(sql`
    SELECT "id", "detection_count", "peak_frp_mw", "first_seen_at", "last_seen_at"
    FROM "aoi_events"
    WHERE "aoi_id" = ${match.aoiId} AND "dedupe_hash" = ${hash}
    LIMIT 1
  `);
  const existingRows = decodeRows<{
    id: string;
    detection_count: number;
    peak_frp_mw: number | null;
    first_seen_at: Date | string;
    last_seen_at: Date | string;
  }>(existing);

  if (existingRows.length > 0) {
    const row = existingRows[0];
    const newPeak = Math.max(
      row.peak_frp_mw ?? 0,
      match.peakFrpMw ?? 0,
    );
    await db.execute(sql`
      UPDATE "aoi_events"
      SET
        "last_seen_at" = GREATEST("last_seen_at", ${match.lastSeenAt.toISOString()}::timestamptz),
        "first_seen_at" = LEAST("first_seen_at", ${match.firstSeenAt.toISOString()}::timestamptz),
        "detection_count" = "detection_count" + 1,
        "peak_frp_mw" = ${newPeak},
        "nearest_distance_km" = LEAST("nearest_distance_km", ${match.nearestDistanceKm}),
        "status" = CASE WHEN "status" = 'closed' THEN 'closed' ELSE "status" END
      WHERE "id" = ${row.id}
    `);
    return { kind: "updated" };
  }

  const insertResult = await db.execute(sql`
    INSERT INTO "aoi_events" (
      "aoi_id", "first_seen_at", "last_seen_at",
      "nearest_distance_km", "detection_count", "peak_frp_mw",
      "dedupe_hash", "status"
    ) VALUES (
      ${match.aoiId},
      ${match.firstSeenAt.toISOString()},
      ${match.lastSeenAt.toISOString()},
      ${match.nearestDistanceKm},
      1,
      ${match.peakFrpMw},
      ${hash},
      'new'
    )
    ON CONFLICT ("aoi_id", "dedupe_hash") DO NOTHING
    RETURNING "id"
  `);
  const insertRows = decodeRows<{ id: string }>(insertResult);
  return { kind: "created", eventId: insertRows[0]?.id };
}

// ---------------------------------------------------------------------------
// Helpers

function detectionTimestamp(d: FirmsDetection): Date | null {
  if (!d.acqDate || !d.acqTime) return null;
  // FIRMS `acq_time` is UTC "HHMM"; pad 3-digit "HMM" if NASA drops a leading
  // zero.
  const padded = d.acqTime.padStart(4, "0");
  const hh = padded.slice(0, 2);
  const mm = padded.slice(2, 4);
  const iso = `${d.acqDate}T${hh}:${mm}:00Z`;
  const t = new Date(iso);
  return Number.isFinite(t.getTime()) ? t : null;
}

async function dbNow(db: AppDb): Promise<Date> {
  const result = await db.execute(sql`SELECT now() AS now`);
  const rows = decodeRows<{ now: Date | string }>(result);
  const raw = rows[0]?.now;
  if (!raw) return new Date();
  const t = raw instanceof Date ? raw : new Date(raw);
  // 1 ms back to guard against same-millisecond inserts immediately after.
  return new Date(t.getTime() - 1);
}

// Exported for tests.
export const _internal = {
  confidencePassesGate,
  detectionTimestamp,
};
