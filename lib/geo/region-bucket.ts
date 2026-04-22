/**
 * 5°×5° tile bucket key derived from a (lon, lat) centroid.
 *
 * Used by the Stage 2 cron to coalesce FIRMS API calls: every AOI in the
 * same bucket is fetched in a single bbox query. Determinism is essential —
 * same centroid must always map to the same key, regardless of locale or
 * floating-point quirks.
 *
 * Format: `5x5:<E|W><lonAbs(3-digit)>_<N|S><latAbs(2-digit)>` referring to
 * the southwest corner of the 5° tile, e.g. `5x5:W015_N045` for a point
 * at (-12.4°, 47.8°). The centroid (-12.4, 47.8) lives in the tile whose SW
 * corner is (-15, 45).
 */

const TILE_DEG = 5;

export function regionBucketFromLonLat(lon: number, lat: number): string {
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
    throw new Error(`regionBucket: non-finite coords lon=${lon} lat=${lat}`);
  }
  if (lon < -180 || lon > 180 || lat < -90 || lat > 90) {
    throw new Error(`regionBucket: out-of-range coords lon=${lon} lat=${lat}`);
  }
  // Floor toward -infinity so e.g. -12.4 → -15, 47.8 → 45.
  const swLon = Math.floor(lon / TILE_DEG) * TILE_DEG;
  const swLat = Math.floor(lat / TILE_DEG) * TILE_DEG;
  const lonAbs = Math.abs(swLon).toString().padStart(3, "0");
  const latAbs = Math.abs(swLat).toString().padStart(2, "0");
  const lonHemi = swLon < 0 ? "W" : "E";
  const latHemi = swLat < 0 ? "S" : "N";
  return `5x5:${lonHemi}${lonAbs}_${latHemi}${latAbs}`;
}
