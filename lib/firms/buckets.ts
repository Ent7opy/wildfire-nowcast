/**
 * Bucket coalescing — pairs with `lib/geo/region-bucket.ts`.
 *
 * The cron groups AOIs by their `region_bucket` so each FIRMS API call covers
 * a 5°×5° tile that may contain many AOIs. Two helpers:
 *
 *   - `getActiveBuckets(db)` — distinct buckets among non-archived AOIs,
 *     ordered by AOI count desc (heaviest tiles first if we ever rate-limit).
 *
 *   - `bucketToBbox(bucket)` — inverse of `regionBucketFromLonLat`, producing
 *     the FIRMS-shaped bbox tuple `[minLon, minLat, maxLon, maxLat]`.
 *
 * The bucket key format (e.g. `5x5:W015_N045`) names the SW corner of the 5°
 * tile. `bucketToBbox` returns that corner plus a +5° offset on each axis.
 */
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";
import type { FirmsBbox } from "./client";

const TILE_DEG = 5;

const BUCKET_RE = /^5x5:([EW])(\d{3})_([NS])(\d{2})$/;

export type ActiveBucket = {
  bucket: string;
  aoiCount: number;
};

export async function getActiveBuckets(db: AppDb): Promise<ActiveBucket[]> {
  const result = (await db.execute(sql`
    SELECT "region_bucket" AS bucket, COUNT(*)::int AS aoi_count
    FROM "aois"
    WHERE "archived_at" IS NULL
    GROUP BY "region_bucket"
    ORDER BY aoi_count DESC, bucket ASC
  `)) as unknown as {
    rows?: Array<{ bucket: string; aoi_count: number }>;
  };
  const rows = (result.rows ?? (result as unknown as Array<{ bucket: string; aoi_count: number }>)) as Array<{
    bucket: string;
    aoi_count: number;
  }>;
  return rows.map((r) => ({ bucket: r.bucket, aoiCount: Number(r.aoi_count) }));
}

export function bucketToBbox(bucket: string): FirmsBbox {
  const m = BUCKET_RE.exec(bucket);
  if (!m) {
    throw new Error(`bucketToBbox: malformed bucket key "${bucket}"`);
  }
  const [, lonHemi, lonAbs, latHemi, latAbs] = m;
  const swLonAbs = parseInt(lonAbs, 10);
  const swLatAbs = parseInt(latAbs, 10);
  const swLon = lonHemi === "W" ? -swLonAbs : swLonAbs;
  const swLat = latHemi === "S" ? -swLatAbs : swLatAbs;
  // Clamp the NE corner to the valid range to avoid edge-of-world overflow.
  const neLon = Math.min(180, swLon + TILE_DEG);
  const neLat = Math.min(90, swLat + TILE_DEG);
  return [swLon, swLat, neLon, neLat];
}
