import { describe, expect, it } from "vitest";

import { eventLimitForZoom, shouldLoadFronts, shouldRenderCentroids } from "../utils/mapMath";

describe("map LOD behavior", () => {
  it("adjusts event limits by zoom", () => {
    expect(eventLimitForZoom(1)).toBe(2000);
    expect(eventLimitForZoom(3)).toBe(4000);
    expect(eventLimitForZoom(5)).toBe(10000);
  });

  it("loads fronts only at higher zoom", () => {
    expect(shouldLoadFronts(4.9)).toBe(false);
    expect(shouldLoadFronts(5)).toBe(true);
  });

  it("uses centroids only at low zoom", () => {
    expect(shouldRenderCentroids(3.9)).toBe(true);
    expect(shouldRenderCentroids(4)).toBe(false);
  });
});
