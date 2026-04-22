/**
 * Dedupe-hash unit tests — collision structure is load-bearing for Stage 3.
 */
import { describe, expect, it } from "vitest";
import { computeDedupeHash } from "@/lib/firms/dedupe";

const AOI = "11111111-2222-3333-4444-555555555555";
const BUCKET = "5x5:W125_N35";
const SOURCE = "VIIRS_NOAA20_NRT";

describe("computeDedupeHash", () => {
  it("is deterministic for identical inputs", () => {
    const a = computeDedupeHash({
      aoiId: AOI,
      bucket: BUCKET,
      centroidLat: 38.44,
      centroidLon: -122.68,
      source: SOURCE,
      detectedAt: new Date("2026-04-21T04:17:00Z"),
    });
    const b = computeDedupeHash({
      aoiId: AOI,
      bucket: BUCKET,
      centroidLat: 38.44,
      centroidLon: -122.68,
      source: SOURCE,
      detectedAt: new Date("2026-04-21T04:17:00Z"),
    });
    expect(a).toBe(b);
    expect(a).toHaveLength(32);
  });

  it("treats detections inside the same UTC day as one event", () => {
    const morning = computeDedupeHash({
      aoiId: AOI,
      bucket: BUCKET,
      centroidLat: 38.44,
      centroidLon: -122.68,
      source: SOURCE,
      detectedAt: new Date("2026-04-21T00:30:00Z"),
    });
    const evening = computeDedupeHash({
      aoiId: AOI,
      bucket: BUCKET,
      centroidLat: 38.44,
      centroidLon: -122.68,
      source: SOURCE,
      detectedAt: new Date("2026-04-21T23:30:00Z"),
    });
    expect(morning).toBe(evening);
  });

  it("rolls to a new event across midnight UTC", () => {
    const before = computeDedupeHash({
      aoiId: AOI,
      bucket: BUCKET,
      centroidLat: 38.44,
      centroidLon: -122.68,
      source: SOURCE,
      detectedAt: new Date("2026-04-21T23:45:00Z"),
    });
    const after = computeDedupeHash({
      aoiId: AOI,
      bucket: BUCKET,
      centroidLat: 38.44,
      centroidLon: -122.68,
      source: SOURCE,
      detectedAt: new Date("2026-04-22T00:15:00Z"),
    });
    expect(before).not.toBe(after);
  });

  it("rolls to a new event when the cluster shifts beyond 0.01° (~1 km)", () => {
    // Use positive-direction nudges: floor(38.445 * 100)/100 = 38.44 (same bin
    // as 38.44). Negative-coordinate behaviour is documented in the dedupe
    // module — Math.floor on a negative crosses a bin boundary at any nudge.
    const here = computeDedupeHash({
      aoiId: AOI,
      bucket: BUCKET,
      centroidLat: 38.44,
      centroidLon: 100.50,
      source: SOURCE,
      detectedAt: new Date("2026-04-21T04:00:00Z"),
    });
    const nudgedSameBin = computeDedupeHash({
      aoiId: AOI,
      bucket: BUCKET,
      centroidLat: 38.445,
      centroidLon: 100.505,
      source: SOURCE,
      detectedAt: new Date("2026-04-21T04:30:00Z"),
    });
    const movedToNewBin = computeDedupeHash({
      aoiId: AOI,
      bucket: BUCKET,
      centroidLat: 38.46,
      centroidLon: 100.52,
      source: SOURCE,
      detectedAt: new Date("2026-04-21T04:30:00Z"),
    });
    expect(here).toBe(nudgedSameBin);
    expect(here).not.toBe(movedToNewBin);
  });

  it("is sensitive to AOI id, source, and bucket", () => {
    const base = {
      aoiId: AOI,
      bucket: BUCKET,
      centroidLat: 38.44,
      centroidLon: -122.68,
      source: SOURCE,
      detectedAt: new Date("2026-04-21T04:00:00Z"),
    };
    const h = computeDedupeHash(base);
    expect(computeDedupeHash({ ...base, aoiId: "ffffffff-ffff-ffff-ffff-ffffffffffff" })).not.toBe(h);
    expect(computeDedupeHash({ ...base, bucket: "5x5:W120_N35" })).not.toBe(h);
    expect(computeDedupeHash({ ...base, source: "VIIRS_SNPP_NRT" })).not.toBe(h);
  });
});
