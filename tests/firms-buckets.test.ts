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
