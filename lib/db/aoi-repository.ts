/**
 * AOI repository — single boundary between API routes and the database.
 *
 * Holds the geometry-translation logic so the routes are oblivious to whether
 * the underlying engine is Neon+PostGIS or PGlite+TEXT. Both code paths
 * accept and return GeoJSON; the contract is identical.
 *
 * Spec references:
 *   - docs/SPEC-A-prime-v1.md §Data model, §API surface
 *   - docs/pivot-architecture.md §3
 */
import { and, eq, isNull, sql } from "drizzle-orm";
import type { AppDb } from "./client";
import { aois, aoiRules } from "@/db/schema";
import {
  areaHaOfMultiPolygon,
  bboxOfMultiPolygon,
  centroidOfBbox,
  toMultiPolygon,
  type GeoJSONMultiPolygon,
  type GeoJSONPoint,
  type GeoJSONPolygon,
} from "@/lib/geo/polygon";
import { regionBucketFromLonLat } from "@/lib/geo/region-bucket";
import type { PolygonalGeom } from "@/lib/validators/geojson";
import type { RulesUpsert } from "@/lib/validators/aoi";

const MAX_AREA_HA = 100_000;

export class AoiAreaTooLargeError extends Error {
  constructor(public readonly areaHa: number) {
    super(`AOI area ${areaHa.toFixed(1)} ha exceeds the v1 limit of ${MAX_AREA_HA} ha`);
    this.name = "AoiAreaTooLargeError";
  }
}

export class AoiNotFoundError extends Error {
  constructor(public readonly id: string) {
    super(`AOI ${id} not found`);
    this.name = "AoiNotFoundError";
  }
}

export class AoiNameConflictError extends Error {
  constructor(public readonly name: string) {
    super(`AOI named "${name}" already exists for this user`);
    this.name = "AoiNameConflictError";
  }
}

type DerivedGeometry = {
  polygon: GeoJSONMultiPolygon;
  bbox: GeoJSONPolygon;
  centroid: GeoJSONPoint;
  areaHa: number;
  regionBucket: string;
};

export type AoiRow = {
  id: string;
  userId: string;
  name: string;
  polygon: GeoJSONMultiPolygon;
  bbox: GeoJSONPolygon;
  centroid: GeoJSONPoint;
  regionBucket: string;
  areaHa: number;
  createdAt: Date;
  archivedAt: Date | null;
};

export type AoiRulesRow = {
  aoiId: string;
  distanceBufferKm: number;
  minConfidence: "low" | "nominal" | "high";
  minFrpMw: number;
  quietHours: { tz: string; startHour: number; endHour: number } | null;
  pausedUntil: Date | null;
  notifyChannels: Array<
    { type: "email"; target: string } | { type: "webhook"; target: string }
  >;
  updatedAt: Date;
};

function deriveGeometry(geom: PolygonalGeom): DerivedGeometry {
  const polygon = toMultiPolygon(geom);
  const bbox = bboxOfMultiPolygon(polygon);
  const centroid = centroidOfBbox(bbox);
  const areaHa = areaHaOfMultiPolygon(polygon);
  if (areaHa > MAX_AREA_HA) throw new AoiAreaTooLargeError(areaHa);
  const [lon, lat] = centroid.coordinates;
  const regionBucket = regionBucketFromLonLat(lon, lat);
  return { polygon, bbox, centroid, areaHa, regionBucket };
}

/**
 * Encode a GeoJSON object for the storage column.
 *
 * On Neon+PostGIS we wrap with `ST_GeomFromGeoJSON(<json>::text)::geometry`
 * inline at the SQL site. On PGlite we just store the JSON string.
 */
function geomLiteralForStorage(
  db: AppDb,
  geom: GeoJSONMultiPolygon | GeoJSONPolygon | GeoJSONPoint,
) {
  const json = JSON.stringify(geom);
  if (db.usePostGIS) {
    return sql`ST_SetSRID(ST_GeomFromGeoJSON(${json}), 4326)`;
  }
  return sql`${json}`;
}

/**
 * Decode a storage value into GeoJSON.
 *
 * Neon+PostGIS rows already get GeoJSON via the SELECT projection (see
 * `geomColumnExpression`). PGlite stores TEXT — parse it here.
 */
function decodeGeom<T>(db: AppDb, value: unknown): T {
  if (typeof value === "string") return JSON.parse(value) as T;
  if (value && typeof value === "object") return value as T;
  // PostGIS rows projected via ST_AsGeoJSON arrive as strings on node-postgres
  // by default; the conditional above covers them. Defensive fallback:
  if (db.usePostGIS && value === null) {
    throw new Error("decodeGeom: NULL geometry from PostGIS");
  }
  throw new Error(`decodeGeom: unsupported value type ${typeof value}`);
}

/**
 * SQL projection for a geometry column in SELECTs.
 *   PostGIS: ST_AsGeoJSON(col)::text
 *   PGlite : the raw TEXT column
 */
function projGeom(db: AppDb, column: string): ReturnType<typeof sql.raw> {
  return db.usePostGIS
    ? sql.raw(`ST_AsGeoJSON("${column}")::text AS "${column}"`)
    : sql.raw(`"${column}"`);
}

export async function createAoi(
  db: AppDb,
  args: { userId: string; name: string; geometry: PolygonalGeom },
): Promise<{ aoi: AoiRow; rules: AoiRulesRow }> {
  const derived = deriveGeometry(args.geometry);

  // Check name conflict explicitly so we can return a typed error instead of
  // bubbling a Postgres unique-constraint error.
  const existing = await db
    .select({ id: aois.id })
    .from(aois)
    .where(
      and(
        eq(aois.userId, args.userId),
        eq(aois.name, args.name),
        isNull(aois.archivedAt),
      ),
    )
    .limit(1);
  if (existing.length > 0) throw new AoiNameConflictError(args.name);

  const insertSql = sql`
    INSERT INTO ${aois} (
      "user_id", "name", "polygon", "bbox", "centroid", "region_bucket", "area_ha"
    ) VALUES (
      ${args.userId},
      ${args.name},
      ${geomLiteralForStorage(db, derived.polygon)},
      ${geomLiteralForStorage(db, derived.bbox)},
      ${geomLiteralForStorage(db, derived.centroid)},
      ${derived.regionBucket},
      ${derived.areaHa}
    )
    RETURNING "id"
  `;
  const insertResult = (await db.execute(insertSql)) as unknown as {
    rows?: Array<{ id: string }>;
  };
  const rows = (insertResult.rows ?? (insertResult as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  const newId = rows[0].id;

  // Default rules row.
  await db.insert(aoiRules).values({ aoiId: newId });

  const aoi = await getAoiById(db, args.userId, newId);
  if (!aoi) throw new Error("createAoi: row vanished after insert");
  const rules = await getRulesByAoiId(db, newId);
  if (!rules) throw new Error("createAoi: rules row vanished after insert");
  return { aoi, rules };
}

export async function listAois(
  db: AppDb,
  userId: string,
): Promise<AoiRow[]> {
  const result = (await db.execute(sql`
    SELECT
      "id", "user_id", "name",
      ${projGeom(db, "polygon")},
      ${projGeom(db, "bbox")},
      ${projGeom(db, "centroid")},
      "region_bucket", "area_ha", "created_at", "archived_at"
    FROM ${aois}
    WHERE "user_id" = ${userId} AND "archived_at" IS NULL
    ORDER BY "created_at" DESC
  `)) as unknown as { rows?: RawAoiRow[] };
  const rows = (result.rows ?? (result as unknown as RawAoiRow[])) as RawAoiRow[];
  return rows.map((r) => mapAoiRow(db, r));
}

export async function getAoiById(
  db: AppDb,
  userId: string,
  aoiId: string,
): Promise<AoiRow | null> {
  const result = (await db.execute(sql`
    SELECT
      "id", "user_id", "name",
      ${projGeom(db, "polygon")},
      ${projGeom(db, "bbox")},
      ${projGeom(db, "centroid")},
      "region_bucket", "area_ha", "created_at", "archived_at"
    FROM ${aois}
    WHERE "id" = ${aoiId} AND "user_id" = ${userId} AND "archived_at" IS NULL
    LIMIT 1
  `)) as unknown as { rows?: RawAoiRow[] };
  const rows = (result.rows ?? (result as unknown as RawAoiRow[])) as RawAoiRow[];
  if (rows.length === 0) return null;
  return mapAoiRow(db, rows[0]);
}

export async function updateAoi(
  db: AppDb,
  args: {
    userId: string;
    aoiId: string;
    patch: { name?: string; geometry?: PolygonalGeom };
  },
): Promise<AoiRow> {
  const current = await getAoiById(db, args.userId, args.aoiId);
  if (!current) throw new AoiNotFoundError(args.aoiId);

  const fragments: ReturnType<typeof sql>[] = [];

  if (args.patch.name !== undefined && args.patch.name !== current.name) {
    // Same conflict check as create.
    const conflict = await db
      .select({ id: aois.id })
      .from(aois)
      .where(
        and(
          eq(aois.userId, args.userId),
          eq(aois.name, args.patch.name),
          isNull(aois.archivedAt),
        ),
      )
      .limit(1);
    if (conflict.length > 0) throw new AoiNameConflictError(args.patch.name);
    fragments.push(sql`"name" = ${args.patch.name}`);
  }

  if (args.patch.geometry) {
    const derived = deriveGeometry(args.patch.geometry);
    fragments.push(sql`"polygon" = ${geomLiteralForStorage(db, derived.polygon)}`);
    fragments.push(sql`"bbox" = ${geomLiteralForStorage(db, derived.bbox)}`);
    fragments.push(sql`"centroid" = ${geomLiteralForStorage(db, derived.centroid)}`);
    fragments.push(sql`"region_bucket" = ${derived.regionBucket}`);
    fragments.push(sql`"area_ha" = ${derived.areaHa}`);
  }

  if (fragments.length === 0) {
    return current; // no-op patch (e.g. only same-name set)
  }

  const setClause = sql.join(fragments, sql.raw(", "));
  await db.execute(sql`
    UPDATE ${aois}
    SET ${setClause}
    WHERE "id" = ${args.aoiId} AND "user_id" = ${args.userId}
  `);

  const updated = await getAoiById(db, args.userId, args.aoiId);
  if (!updated) throw new AoiNotFoundError(args.aoiId);
  return updated;
}

export async function archiveAoi(
  db: AppDb,
  args: { userId: string; aoiId: string },
): Promise<void> {
  const current = await getAoiById(db, args.userId, args.aoiId);
  if (!current) throw new AoiNotFoundError(args.aoiId);
  await db
    .update(aois)
    .set({ archivedAt: new Date() })
    .where(and(eq(aois.id, args.aoiId), eq(aois.userId, args.userId)));
}

export async function getRulesByAoiId(
  db: AppDb,
  aoiId: string,
): Promise<AoiRulesRow | null> {
  const rows = await db
    .select()
    .from(aoiRules)
    .where(eq(aoiRules.aoiId, aoiId))
    .limit(1);
  if (rows.length === 0) return null;
  const r = rows[0];
  return {
    aoiId: r.aoiId,
    distanceBufferKm: r.distanceBufferKm,
    minConfidence: r.minConfidence as "low" | "nominal" | "high",
    minFrpMw: r.minFrpMw,
    quietHours: r.quietHours,
    pausedUntil: r.pausedUntil,
    notifyChannels: r.notifyChannels,
    updatedAt: r.updatedAt,
  };
}

export async function upsertRules(
  db: AppDb,
  args: { userId: string; aoiId: string; rules: RulesUpsert },
): Promise<AoiRulesRow> {
  // Authorisation: the AOI must exist and belong to this user.
  const aoi = await getAoiById(db, args.userId, args.aoiId);
  if (!aoi) throw new AoiNotFoundError(args.aoiId);

  const pausedUntil =
    args.rules.pausedUntil === null ? null : new Date(args.rules.pausedUntil);

  await db
    .insert(aoiRules)
    .values({
      aoiId: args.aoiId,
      distanceBufferKm: args.rules.distanceBufferKm,
      minConfidence: args.rules.minConfidence,
      minFrpMw: args.rules.minFrpMw,
      quietHours: args.rules.quietHours,
      pausedUntil,
      notifyChannels: args.rules.notifyChannels,
      updatedAt: new Date(),
    })
    .onConflictDoUpdate({
      target: aoiRules.aoiId,
      set: {
        distanceBufferKm: args.rules.distanceBufferKm,
        minConfidence: args.rules.minConfidence,
        minFrpMw: args.rules.minFrpMw,
        quietHours: args.rules.quietHours,
        pausedUntil,
        notifyChannels: args.rules.notifyChannels,
        updatedAt: new Date(),
      },
    });

  const out = await getRulesByAoiId(db, args.aoiId);
  if (!out) throw new Error("upsertRules: row vanished after upsert");
  return out;
}

// ---------------------------------------------------------------------------
// Internal: row mapper

type RawAoiRow = {
  id: string;
  user_id: string;
  name: string;
  polygon: unknown;
  bbox: unknown;
  centroid: unknown;
  region_bucket: string;
  area_ha: number;
  created_at: Date | string;
  archived_at: Date | string | null;
};

function mapAoiRow(db: AppDb, r: RawAoiRow): AoiRow {
  return {
    id: r.id,
    userId: r.user_id,
    name: r.name,
    polygon: decodeGeom<GeoJSONMultiPolygon>(db, r.polygon),
    bbox: decodeGeom<GeoJSONPolygon>(db, r.bbox),
    centroid: decodeGeom<GeoJSONPoint>(db, r.centroid),
    regionBucket: r.region_bucket,
    areaHa: r.area_ha,
    createdAt: r.created_at instanceof Date ? r.created_at : new Date(r.created_at),
    archivedAt:
      r.archived_at == null
        ? null
        : r.archived_at instanceof Date
          ? r.archived_at
          : new Date(r.archived_at),
  };
}
