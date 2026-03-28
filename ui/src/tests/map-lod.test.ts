import { describe, expect, it } from "vitest";

import { eventLimitForZoom, shouldLoadFronts, shouldRenderCentroids, viewportBbox } from "../utils/mapMath";

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

describe("viewportBbox", () => {
  const baseView = { zoom: 6, longitude: 10, latitude: 48, pitch: 0, bearing: 0 };

  it("returns bbox wider than a single tile at zoom 6 with 1440px viewport", () => {
    const [minLon, , maxLon] = viewportBbox(baseView, 1440, 900);
    const lonSpan = maxLon - minLon;
    // A single tile at zoom 6 is ~5.625°; 1440px / 256px ≈ 5.625 tiles wide → span ≈ 31.6°
    expect(lonSpan).toBeGreaterThan(20);
  });

  it("clamps longitude to [-180, 180]", () => {
    const view = { ...baseView, zoom: 1, longitude: 170 };
    const [minLon, , maxLon] = viewportBbox(view, 1440, 900);
    expect(minLon).toBeGreaterThanOrEqual(-180);
    expect(maxLon).toBeLessThanOrEqual(180);
  });

  it("uses Math.min for southern bound so it can reach -85", () => {
    const view = { ...baseView, latitude: -70 };
    const [, minLat] = viewportBbox(view, 1440, 900);
    expect(minLat).toBeLessThanOrEqual(-85);
  });

  it("uses default viewport dimensions when none are supplied", () => {
    const bbox = viewportBbox(baseView);
    expect(bbox).toHaveLength(4);
    expect(bbox[2] - bbox[0]).toBeGreaterThan(20);
  });
});
