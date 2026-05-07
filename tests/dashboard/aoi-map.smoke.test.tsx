/**
 * Smoke test: <AoiMap> mounts without throwing in static markup. jsdom can't
 * run WebGL, so we don't assert on tile rendering — only that the container
 * element is present.
 */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server.node";
import { AoiMap } from "@/app/dashboard/_components/aoi-map";

const POLYGON = {
  type: "Polygon" as const,
  coordinates: [
    [
      [0, 0] as [number, number],
      [1, 0] as [number, number],
      [1, 1] as [number, number],
      [0, 1] as [number, number],
      [0, 0] as [number, number],
    ],
  ],
};
const BBOX = POLYGON;
const CENTROID = { type: "Point" as const, coordinates: [0.5, 0.5] as [number, number] };

describe("<AoiMap>", () => {
  it("renders a map container in view mode without throwing", () => {
    const html = renderToStaticMarkup(
      <AoiMap mode="view" polygon={POLYGON} bbox={BBOX} centroid={CENTROID} detections={[]} />,
    );
    expect(html).toContain("aoi-map-container");
  });

  it("renders a map container in draw mode without throwing", () => {
    const html = renderToStaticMarkup(
      <AoiMap mode="draw" onPolygon={() => {}} />,
    );
    expect(html).toContain("aoi-map-container");
  });
});
