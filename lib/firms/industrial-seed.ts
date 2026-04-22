/**
 * Loads + helpers for the static industrial mask seed.
 *
 * The JSON seed lives at `db/seeds/industrial-mask-stage2.json` and stores
 * point centroids + a buffer radius (km). We expand each row into a small
 * GeoJSON polygon at insert / load time. The expansion uses an equirectangular
 * approximation around the centroid (sufficient at the 4–14 km scale of these
 * masks; ST_DWithin in production uses geography for the actual hit-testing).
 *
 * Source citations live in the JSON header.
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { GeoJSONPolygon } from "@/lib/geo/polygon";

export type IndustrialSeedRow = {
  kind: string;
  name: string;
  lat: number;
  lon: number;
  radiusKm: number;
};

export type IndustrialSeed = {
  polygons: IndustrialSeedRow[];
};

const SEED_PATH = join(
  process.cwd(),
  "db",
  "seeds",
  "industrial-mask-stage2.json",
);

const EARTH_KM_PER_DEG_LAT = 111.32;

export async function loadIndustrialMaskSeed(): Promise<IndustrialSeed> {
  const raw = await readFile(SEED_PATH, "utf8");
  const parsed = JSON.parse(raw) as { polygons: IndustrialSeedRow[] };
  if (!Array.isArray(parsed.polygons) || parsed.polygons.length === 0) {
    throw new Error(
      "industrial-mask-stage2.json: expected non-empty `polygons` array",
    );
  }
  for (const r of parsed.polygons) {
    if (
      typeof r.lat !== "number" ||
      typeof r.lon !== "number" ||
      typeof r.radiusKm !== "number" ||
      typeof r.kind !== "string" ||
      typeof r.name !== "string"
    ) {
      throw new Error(`industrial-mask seed row malformed: ${JSON.stringify(r)}`);
    }
  }
  return parsed;
}

/**
 * Expand a (lon, lat, radiusKm) point into a square GeoJSON polygon.
 *
 * Equirectangular approximation: 1° lat ≈ 111.32 km, 1° lon ≈ 111.32 * cos(lat) km.
 * The polygon is a 5-vertex closed ring per the GeoJSON spec.
 */
export function pointBoxToPolygon(
  lon: number,
  lat: number,
  radiusKm: number,
): GeoJSONPolygon {
  if (
    !Number.isFinite(lon) ||
    !Number.isFinite(lat) ||
    !Number.isFinite(radiusKm) ||
    radiusKm <= 0
  ) {
    throw new Error(
      `pointBoxToPolygon: invalid args lon=${lon} lat=${lat} radiusKm=${radiusKm}`,
    );
  }
  const dLat = radiusKm / EARTH_KM_PER_DEG_LAT;
  const cosLat = Math.cos((lat * Math.PI) / 180);
  // Guard against the pole (cos = 0). At |lat| > 89° the buffer wraps the
  // entire parallel; treat as a 1° lon span to keep PostGIS happy.
  const dLon = Math.abs(cosLat) < 1e-6
    ? 1
    : radiusKm / (EARTH_KM_PER_DEG_LAT * cosLat);
  const minLon = lon - dLon;
  const maxLon = lon + dLon;
  const minLat = Math.max(-90, lat - dLat);
  const maxLat = Math.min(90, lat + dLat);
  return {
    type: "Polygon",
    coordinates: [
      [
        [minLon, minLat],
        [maxLon, minLat],
        [maxLon, maxLat],
        [minLon, maxLat],
        [minLon, minLat],
      ],
    ],
  };
}
