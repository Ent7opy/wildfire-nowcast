/**
 * Industrial-mask seed: verify the JSON loads, every row has the right shape,
 * and a few known hotspots round-trip through `pointBoxToPolygon` with the
 * centroid landing inside the generated polygon.
 */
import { describe, expect, it } from "vitest";
import {
  loadIndustrialMaskSeed,
  pointBoxToPolygon,
} from "@/lib/firms/industrial-seed";

describe("industrial-mask seed", () => {
  it("loads and validates", async () => {
    const seed = await loadIndustrialMaskSeed();
    expect(seed.polygons.length).toBeGreaterThanOrEqual(50);
    expect(seed.polygons.length).toBeLessThanOrEqual(150);
    for (const r of seed.polygons) {
      expect(r.lat).toBeGreaterThan(-90);
      expect(r.lat).toBeLessThan(90);
      expect(r.lon).toBeGreaterThan(-180);
      expect(r.lon).toBeLessThan(180);
      expect(r.radiusKm).toBeGreaterThan(0);
      expect(r.name.length).toBeGreaterThan(0);
      expect(["gas_flare", "refinery", "industrial", "volcanic"]).toContain(r.kind);
    }
  });

  it("pointBoxToPolygon contains the centroid for every seed row", async () => {
    const seed = await loadIndustrialMaskSeed();
    for (const r of seed.polygons) {
      const poly = pointBoxToPolygon(r.lon, r.lat, r.radiusKm);
      const ring = poly.coordinates[0];
      let minLon = Infinity;
      let minLat = Infinity;
      let maxLon = -Infinity;
      let maxLat = -Infinity;
      for (const [lon, lat] of ring) {
        if (lon < minLon) minLon = lon;
        if (lon > maxLon) maxLon = lon;
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
      }
      expect(r.lon).toBeGreaterThanOrEqual(minLon);
      expect(r.lon).toBeLessThanOrEqual(maxLon);
      expect(r.lat).toBeGreaterThanOrEqual(minLat);
      expect(r.lat).toBeLessThanOrEqual(maxLat);
    }
  });

  it("guards the polar edge case (lat=±90)", () => {
    const polar = pointBoxToPolygon(0, 89.9, 10);
    const ring = polar.coordinates[0];
    for (const [, lat] of ring) {
      expect(lat).toBeLessThanOrEqual(90);
      expect(lat).toBeGreaterThanOrEqual(-90);
    }
  });
});
