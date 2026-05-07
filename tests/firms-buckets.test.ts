/**
 * Bucket coalescing unit tests — `bucketToBbox` round-trips against
 * `regionBucketFromLonLat` for known centroids, and errors on malformed keys.
 */
import { describe, expect, it } from "vitest";
import { bucketToBbox } from "@/lib/firms/buckets";
import { regionBucketFromLonLat } from "@/lib/geo/region-bucket";

const PLACES: ReadonlyArray<{ name: string; lon: number; lat: number }> = [
  { name: "Athens", lon: 23.72, lat: 37.98 },
  { name: "Lisbon", lon: -9.14, lat: 38.72 },
  { name: "San Francisco", lon: -122.42, lat: 37.77 },
  { name: "Darwin", lon: 130.84, lat: -12.46 },
  { name: "Cape Town", lon: 18.42, lat: -33.92 },
  { name: "Longyearbyen", lon: 15.65, lat: 78.22 },
];

describe("bucketToBbox round-trips the centroid", () => {
  for (const p of PLACES) {
    it(`contains ${p.name}`, () => {
      const key = regionBucketFromLonLat(p.lon, p.lat);
      const [minLon, minLat, maxLon, maxLat] = bucketToBbox(key);
      expect(p.lon).toBeGreaterThanOrEqual(minLon);
      expect(p.lon).toBeLessThanOrEqual(maxLon);
      expect(p.lat).toBeGreaterThanOrEqual(minLat);
      expect(p.lat).toBeLessThanOrEqual(maxLat);
      expect(maxLon - minLon).toBeCloseTo(5, 6);
      // At |lat| ≥ 85° the NE corner gets clamped to 90; otherwise it's 5°.
      if (maxLat - minLat < 5) expect(maxLat).toBe(90);
      else expect(maxLat - minLat).toBeCloseTo(5, 6);
    });
  }
});

describe("bucketToBbox error handling", () => {
  it("throws on malformed keys", () => {
    expect(() => bucketToBbox("not-a-bucket")).toThrow();
    expect(() => bucketToBbox("5x5:X125_N35")).toThrow();
    expect(() => bucketToBbox("5x5:W999_N99")).not.toThrow(); // technically malformed lat but allowed by regex
  });
});

describe("bucketToBbox boundary clamping", () => {
  // Pin the documented "+5° offset, clamped to 90/180" behavior so a future
  // refactor can't silently emit a bbox like maxLat=95 that FIRMS would reject.
  it("clamps the NE latitude at the north pole tile", () => {
    // SW corner (0, 85) → NE would be (5, 90); 90 is the clamp boundary itself.
    expect(bucketToBbox("5x5:E000_N85")).toEqual([0, 85, 5, 90]);
  });

  it("returns a degenerate-height bbox at the polar edge tile", () => {
    // regionBucketFromLonLat(0, 90) → swLat = 90 → bbox max = clamp(95, 90) = 90.
    const key = regionBucketFromLonLat(0, 90);
    const [, minLat, , maxLat] = bucketToBbox(key);
    expect(minLat).toBe(90);
    expect(maxLat).toBe(90);
  });

  it("clamps the NE longitude at the anti-meridian tile", () => {
    // regionBucketFromLonLat(180, 0) → swLon = 180 → bbox max = clamp(185, 180) = 180.
    const key = regionBucketFromLonLat(180, 0);
    const [minLon, , maxLon] = bucketToBbox(key);
    expect(minLon).toBe(180);
    expect(maxLon).toBe(180);
  });

  it("treats lon=-180 as a normal western tile (no degenerate width)", () => {
    // Anti-meridian asymmetry: -180 floors to -180, NE lon = -175 (no clamp).
    const key = regionBucketFromLonLat(-180, 0);
    expect(key).toBe("5x5:W180_N00");
    expect(bucketToBbox(key)).toEqual([-180, 0, -175, 5]);
  });

  it("brackets the south pole and equator without clamping", () => {
    expect(bucketToBbox("5x5:E000_S90")).toEqual([0, -90, 5, -85]);
    expect(bucketToBbox("5x5:E000_N00")).toEqual([0, 0, 5, 5]);
  });
});

describe("bucketToBbox determinism", () => {
  it("produces identical output across repeated calls", () => {
    const key = "5x5:W125_N35";
    const a = bucketToBbox(key);
    const b = bucketToBbox(key);
    expect(a).toEqual(b);
  });
});
