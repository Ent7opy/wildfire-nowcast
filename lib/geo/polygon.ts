/**
 * Lightweight, dependency-free geometry helpers for the test path (PGlite).
 *
 * Production reads bbox / centroid / area straight from PostGIS via SQL;
 * tests need an in-process equivalent so the route handlers can run against
 * PGlite. Accuracy targets:
 *   - bbox / centroid: exact for planar polygons in WGS84
 *   - area_ha: spherical-excess approximation, ±1% vs. ST_Area::geography for
 *     polygons up to 100,000 ha (the v1 spec cap). Sufficient for the
 *     in-spec area validation.
 *
 * Inputs are GeoJSON Polygon or MultiPolygon objects (validated upstream by
 * Zod). Coordinates are [lon, lat] per the spec.
 */

export type LonLat = [number, number];
export type LinearRing = LonLat[];
export type GeoJSONPolygon = {
  type: "Polygon";
  coordinates: LinearRing[];
};
export type GeoJSONMultiPolygon = {
  type: "MultiPolygon";
  coordinates: LinearRing[][];
};
export type GeoJSONPoint = {
  type: "Point";
  coordinates: LonLat;
};

export function toMultiPolygon(
  geom: GeoJSONPolygon | GeoJSONMultiPolygon,
): GeoJSONMultiPolygon {
  if (geom.type === "MultiPolygon") return geom;
  return { type: "MultiPolygon", coordinates: [geom.coordinates] };
}

export function bboxOfMultiPolygon(geom: GeoJSONMultiPolygon): GeoJSONPolygon {
  let minLon = Infinity;
  let minLat = Infinity;
  let maxLon = -Infinity;
  let maxLat = -Infinity;
  for (const poly of geom.coordinates) {
    for (const ring of poly) {
      for (const [lon, lat] of ring) {
        if (lon < minLon) minLon = lon;
        if (lon > maxLon) maxLon = lon;
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
      }
    }
  }
  if (!Number.isFinite(minLon)) {
    throw new Error("bbox: empty geometry");
  }
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

export function centroidOfBbox(bbox: GeoJSONPolygon): GeoJSONPoint {
  // Mean of bbox corners — for a rectangle this IS the centroid.
  const ring = bbox.coordinates[0];
  let sumLon = 0;
  let sumLat = 0;
  // Skip the closing point (last == first).
  const pts = ring.slice(0, -1);
  for (const [lon, lat] of pts) {
    sumLon += lon;
    sumLat += lat;
  }
  return { type: "Point", coordinates: [sumLon / pts.length, sumLat / pts.length] };
}

const EARTH_RADIUS_M = 6378137;

/**
 * Spherical polygon area in m², via L'Huilier-style summation. Adapted from
 * the standard "Some Algorithms for Polygons on a Sphere" formula (Chamberlain
 * & Duquette 2007). Sufficient accuracy at AOI scale (<100,000 ha).
 */
function ringAreaM2(ring: LinearRing): number {
  if (ring.length < 4) return 0;
  let total = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const [lon1, lat1] = ring[i];
    const [lon2, lat2] = ring[i + 1];
    total +=
      ((lon2 - lon1) * Math.PI) / 180 *
      (2 + Math.sin((lat1 * Math.PI) / 180) + Math.sin((lat2 * Math.PI) / 180));
  }
  return Math.abs((total * EARTH_RADIUS_M * EARTH_RADIUS_M) / 2);
}

export function areaHaOfMultiPolygon(geom: GeoJSONMultiPolygon): number {
  let m2 = 0;
  for (const poly of geom.coordinates) {
    if (poly.length === 0) continue;
    // Outer ring positive, inner rings (holes) subtracted.
    m2 += ringAreaM2(poly[0]);
    for (let i = 1; i < poly.length; i++) {
      m2 -= ringAreaM2(poly[i]);
    }
  }
  return m2 / 10_000;
}
