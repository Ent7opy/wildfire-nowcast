import { describe, expect, it } from "vitest";
import { regionBucketFromLonLat } from "@/lib/geo/region-bucket";

describe("regionBucketFromLonLat", () => {
  it("buckets a northern-hemisphere positive-longitude point", () => {
    // Sofia, Bulgaria: 23.32, 42.70 → SW corner (20, 40)
    expect(regionBucketFromLonLat(23.32, 42.7)).toBe("5x5:E020_N40");
  });

  it("buckets a western-hemisphere point (Sonoma County)", () => {
    // -122.7, 38.4 → SW corner (-125, 35)
    expect(regionBucketFromLonLat(-122.7, 38.4)).toBe("5x5:W125_N35");
  });

  it("buckets a southern-hemisphere point", () => {
    // Sydney area: 151.0, -33.9 → SW corner (150, -35)
    expect(regionBucketFromLonLat(151.0, -33.9)).toBe("5x5:E150_S35");
  });

  it("is deterministic on tile boundaries (snaps to lower tile)", () => {
    expect(regionBucketFromLonLat(0, 0)).toBe("5x5:E000_N00");
    expect(regionBucketFromLonLat(5, 5)).toBe("5x5:E005_N05");
    expect(regionBucketFromLonLat(-0.0001, -0.0001)).toBe("5x5:W005_S05");
  });

  it("throws on non-finite or out-of-range coordinates", () => {
    expect(() => regionBucketFromLonLat(Number.NaN, 0)).toThrow();
    expect(() => regionBucketFromLonLat(0, 91)).toThrow();
    expect(() => regionBucketFromLonLat(181, 0)).toThrow();
  });
});
