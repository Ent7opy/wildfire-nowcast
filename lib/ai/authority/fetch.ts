/**
 * Stage 8 — fetch the most recent authority-published fire perimeter near
 * a detection. Used by the brief orchestrator to populate
 * `BriefContext.authorityPerimeter` (replaces the Stage 3 hardcoded null).
 *
 * Build-without-blocking discipline: any failure (no covering source, HTTP
 * error, timeout, parse error, no features in radius) returns null. The
 * brief then ships with all-null perimeter fields exactly as it did in
 * Stages 3–7. The next 15-min cron tick is the retry — no backoff loop.
 */
import { selectSourceForBucket, type AuthoritySource } from "./sources";

export type AuthorityPerimeter = {
  source: string;
  postedTs: string;
  containsDetection: boolean;
  rawFeatureId?: string;
};

export type FetchPerimeterArgs = {
  lat: number;
  lon: number;
  radiusKm?: number;
  regionBucket: string;
  now?: Date;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
};

const DEFAULT_RADIUS_KM = 25;
const DEFAULT_TIMEOUT_MS = 10_000;

type GeoJsonFeature = {
  type: "Feature";
  id?: unknown;
  geometry: GeoJsonGeometry | null;
  properties?: Record<string, unknown> | null;
};

type GeoJsonGeometry =
  | { type: "Polygon"; coordinates: number[][][] }
  | { type: "MultiPolygon"; coordinates: number[][][][] };

type GeoJsonCollection = {
  type: "FeatureCollection";
  features?: GeoJsonFeature[];
};

export async function fetchAuthorityPerimeter(
  args: FetchPerimeterArgs,
): Promise<AuthorityPerimeter | null> {
  const source = selectSourceForBucket(args.regionBucket);
  if (!source) return null;

  const radiusKm = args.radiusKm ?? DEFAULT_RADIUS_KM;
  const timeoutMs = args.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const fetchImpl = args.fetchImpl ?? fetch;

  let res: Response;
  try {
    res = await fetchImpl(source.url, {
      signal: AbortSignal.timeout(timeoutMs),
      headers: { accept: "application/geo+json, application/json" },
    });
  } catch (err) {
    console.warn(`[authority] ${source.id}: fetch failed: ${describeErr(err)}`);
    return null;
  }
  if (!res.ok) {
    console.warn(`[authority] ${source.id}: HTTP ${res.status}`);
    return null;
  }

  let body: unknown;
  try {
    body = await res.json();
  } catch (err) {
    console.warn(`[authority] ${source.id}: parse error: ${describeErr(err)}`);
    return null;
  }

  const collection = body as GeoJsonCollection;
  if (!collection || collection.type !== "FeatureCollection" || !Array.isArray(collection.features)) {
    console.warn(`[authority] ${source.id}: not a FeatureCollection`);
    return null;
  }

  const best = pickBestFeature(collection.features, source, args.lat, args.lon, radiusKm);
  if (!best) return null;

  return {
    source: source.name,
    postedTs: best.postedTs,
    containsDetection: best.containsDetection,
    rawFeatureId: source.extractFeatureId({ id: best.feature.id, properties: best.feature.properties ?? undefined }),
  };
}

type Picked = {
  feature: GeoJsonFeature;
  postedTs: string;
  containsDetection: boolean;
};

function pickBestFeature(
  features: GeoJsonFeature[],
  source: AuthoritySource,
  lat: number,
  lon: number,
  radiusKm: number,
): Picked | null {
  let best: Picked | null = null;
  let bestTs = -Infinity;
  for (const f of features) {
    if (!f || !f.geometry) continue;
    const polys = toPolygonRings(f.geometry);
    if (polys.length === 0) continue;
    const centroid = polygonsCentroid(polys);
    if (!centroid) continue;
    const d = haversineKm(lat, lon, centroid.lat, centroid.lon);
    if (d > radiusKm) continue;
    const postedTs = source.extractPostedTs((f.properties ?? {}) as Record<string, unknown>);
    if (!postedTs) continue;
    const tsMs = new Date(postedTs).getTime();
    if (!Number.isFinite(tsMs)) continue;
    if (tsMs <= bestTs) continue;
    bestTs = tsMs;
    best = {
      feature: f,
      postedTs,
      containsDetection: pointInAnyPolygon(lat, lon, polys),
    };
  }
  return best;
}

function toPolygonRings(geom: GeoJsonGeometry): number[][][][] {
  if (geom.type === "Polygon") return [geom.coordinates];
  if (geom.type === "MultiPolygon") return geom.coordinates;
  return [];
}

function polygonsCentroid(polys: number[][][][]): { lat: number; lon: number } | null {
  // Mean of the outer-ring vertices across all polygons. Good enough for the
  // radius filter — we don't need the exact area centroid.
  let sumLon = 0;
  let sumLat = 0;
  let count = 0;
  for (const poly of polys) {
    const outer = poly[0];
    if (!Array.isArray(outer)) continue;
    for (const pt of outer) {
      const lon = Number(pt[0]);
      const lat = Number(pt[1]);
      if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
      sumLon += lon;
      sumLat += lat;
      count += 1;
    }
  }
  if (count === 0) return null;
  return { lon: sumLon / count, lat: sumLat / count };
}

function pointInAnyPolygon(lat: number, lon: number, polys: number[][][][]): boolean {
  for (const poly of polys) {
    if (pointInPolygon(lon, lat, poly)) return true;
  }
  return false;
}

/** Ray casting on (x=lon, y=lat). Outer ring + inner holes (odd-even rule). */
function pointInPolygon(x: number, y: number, rings: number[][][]): boolean {
  let inside = false;
  for (const ring of rings) {
    let local = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i][0];
      const yi = ring[i][1];
      const xj = ring[j][0];
      const yj = ring[j][1];
      const intersect =
        yi > y !== yj > y &&
        x < ((xj - xi) * (y - yi)) / (yj - yi || Number.EPSILON) + xi;
      if (intersect) local = !local;
    }
    if (local) inside = !inside;
  }
  return inside;
}

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
}

function describeErr(err: unknown): string {
  if (err instanceof Error) return `${err.name}: ${err.message}`;
  return String(err);
}
