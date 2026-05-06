import { describe, expect, it } from "vitest";
import {
  areaHaOfMultiPolygon,
  bboxOfMultiPolygon,
  centroidOfBbox,
  toMultiPolygon,
  type GeoJSONMultiPolygon,
} from "@/lib/geo/polygon";

const SQUARE_1DEG: GeoJSONMultiPolygon = {
  type: "MultiPolygon",
  coordinates: [
    [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],
      ],
    ],
  ],
};

describe("polygon helpers", () => {
  it("normalises Polygon → MultiPolygon", () => {
    const mp = toMultiPolygon({
      type: "Polygon",
      coordinates: SQUARE_1DEG.coordinates[0],
    });
    expect(mp.type).toBe("MultiPolygon");
    expect(mp.coordinates).toHaveLength(1);
  });

  it("computes the bbox of a 1°×1° square", () => {
    const bbox = bboxOfMultiPolygon(SQUARE_1DEG);
    expect(bbox.coordinates[0]).toEqual([
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [0, 0],
    ]);
  });

  it("centroid of a square's bbox is the geometric centre", () => {
    const c = centroidOfBbox(bboxOfMultiPolygon(SQUARE_1DEG));
    expect(c.coordinates[0]).toBeCloseTo(0.5, 6);
    expect(c.coordinates[1]).toBeCloseTo(0.5, 6);
  });

  it("approximates area at the equator (1°² ≈ 12,309 km² ≈ 1,230,866 ha)", () => {
    // Closed-form: a 1°×1° equatorial square is ~12,309 km² → ~1,230,866 ha.
    // We accept ±2% from the spherical-excess approximation.
    const area = areaHaOfMultiPolygon(SQUARE_1DEG);
    expect(area).toBeGreaterThan(1_206_000);
    expect(area).toBeLessThan(1_256_000);
  });

  it("subtracts the area of holes", () => {
    const withHole: GeoJSONMultiPolygon = {
      type: "MultiPolygon",
      coordinates: [
        [
          // outer 1°×1°
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0],
          ],
          // hole 0.5°×0.5° centered
          [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
            [0.25, 0.25],
          ],
        ],
      ],
    };
    const outerOnly = areaHaOfMultiPolygon(SQUARE_1DEG);
    const punched = areaHaOfMultiPolygon(withHole);
    // Hole is 0.25 of the outer (in the planar approx).
    expect(punched).toBeLessThan(outerOnly);
    expect(punched / outerOnly).toBeCloseTo(0.75, 1);
  });
});
