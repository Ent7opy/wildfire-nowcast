/**
 * Dedupe-hash for `aoi_events`.
 *
 * Same hash => same event (UPSERT, extends an open event window).
 * Different hash => new event (Stage 3's brief generator picks it up).
 *
 * Inputs:
 *   - aoi_id          (UUID, lowercased)
 *   - bucket          ("5x5:W015_N045")
 *   - centroid lat/lon, rounded to 0.01° (≈ 1 km) so detection cluster shifts
 *                     within a kilometre stay one event
 *   - source          ("VIIRS_NOAA20_NRT")
 *   - 24h window idx  (UTC day-of-epoch derived from detected_at)
 *
 * Hash: sha256, hex; truncated to 32 chars (collision risk negligible at our
 * scale and the column stays inspectable by hand).
 *
 * The window index is "days since UTC epoch". Two detections falling in the
 * same UTC day land in the same event by default; one straddling midnight
 * gets a new event row. That tradeoff is acceptable for v1: the rare edge
 * case (a fire active across midnight UTC) just produces two briefs instead
 * of one — still better than missing a real fire.
 */
import { createHash } from "node:crypto";

const COORD_PRECISION = 100; // 0.01° bins ≈ 1 km

export type DedupeArgs = {
  aoiId: string;
  bucket: string;
  centroidLat: number;
  centroidLon: number;
  source: string;
  detectedAt: Date;
};

export function computeDedupeHash(args: DedupeArgs): string {
  const lat = Math.floor(args.centroidLat * COORD_PRECISION) / COORD_PRECISION;
  const lon = Math.floor(args.centroidLon * COORD_PRECISION) / COORD_PRECISION;
  const dayIdx = Math.floor(args.detectedAt.getTime() / (24 * 60 * 60 * 1000));
  const material = [
    args.aoiId.toLowerCase(),
    args.bucket,
    lat.toFixed(2),
    lon.toFixed(2),
    args.source,
    String(dayIdx),
  ].join("|");
  return createHash("sha256").update(material).digest("hex").slice(0, 32);
}
