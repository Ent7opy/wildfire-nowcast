import { describe, expect, it } from "vitest";

import { buildFrontIndexByEvent, geometryBounds } from "../map/layerUtils";

describe("layer utils", () => {
  it("extracts bounds from polygon geometry", () => {
    const bounds = geometryBounds({
      type: "Polygon",
      coordinates: [
        [
          [23.0, 42.0],
          [23.3, 42.0],
          [23.3, 42.4],
          [23.0, 42.4],
          [23.0, 42.0]
        ]
      ]
    });

    expect(bounds).toEqual([23.0, 42.0, 23.3, 42.4]);
  });

  it("extracts bounds from geometry collection payload", () => {
    const bounds = geometryBounds({
      type: "GeometryCollection",
      geometries: [
        { type: "Point", coordinates: [10, 20] },
        { type: "Point", coordinates: [12, 18] }
      ]
    });

    expect(bounds).toEqual([10, 18, 12, 20]);
  });

  it("indexes fronts by event and keeps highest-detection front", () => {
    const index = buildFrontIndexByEvent([
      { front_id: "front-1", event_id: "event-1", detection_count: 10 },
      { front_id: "front-2", event_id: "event-1", detection_count: 25 },
      { front_id: "front-3", event_id: "event-2", detection_count: 5 }
    ]);

    expect(index["event-1"]).toEqual({ frontId: "front-2", detectionCount: 25 });
    expect(index["event-2"]).toEqual({ frontId: "front-3", detectionCount: 5 });
  });
});
