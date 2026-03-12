import { describe, expect, it } from "vitest";

import { selectionViewFromBounds } from "../utils/mapSelection";

describe("selection zoom fit", () => {
  it("centers on geometry bounds midpoint", () => {
    const view = selectionViewFromBounds([10, 20, 14, 28], { targetOccupancy: 0.3 });
    expect(view.longitude).toBe(12);
    expect(view.latitude).toBe(24);
  });

  it("zooms in more for smaller geometries", () => {
    const coarse = selectionViewFromBounds([0, 0, 6, 6], { targetOccupancy: 0.3 });
    const fine = selectionViewFromBounds([0, 0, 1, 1], { targetOccupancy: 0.3 });
    expect(fine.zoom).toBeGreaterThan(coarse.zoom);
  });

  it("respects configured zoom clamps", () => {
    const tiny = selectionViewFromBounds([0, 0, 0.0001, 0.0001], {
      minZoom: 4,
      maxZoom: 9,
      targetOccupancy: 0.3
    });
    expect(tiny.zoom).toBeLessThanOrEqual(9);
    expect(tiny.zoom).toBeGreaterThanOrEqual(4);
  });
});
