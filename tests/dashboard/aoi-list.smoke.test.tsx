/**
 * Smoke test for the dashboard AOI list. Renders to static markup via
 * react-dom/server.node — no jsdom, no testing-library.
 */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server.node";
import { AoiList } from "@/app/dashboard/_components/aoi-list";
import type { AoiListRow } from "@/lib/db/aoi-repository";

const baseRow: AoiListRow = {
  id: "11111111-1111-1111-1111-111111111111",
  userId: "user_test",
  name: "Spring Creek Preserve",
  polygon: { type: "MultiPolygon", coordinates: [] },
  bbox: { type: "Polygon", coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]] },
  centroid: { type: "Point", coordinates: [0.5, 0.5] },
  regionBucket: "5x5:W125_N35",
  areaHa: 123.4,
  createdAt: new Date("2026-01-15T10:00:00Z"),
  archivedAt: null,
  lastBriefAt: new Date("2026-04-20T10:00:00Z"),
  pausedUntil: null,
};

describe("<AoiList>", () => {
  it("empty state shows the create CTA", () => {
    const html = renderToStaticMarkup(<AoiList rows={[]} />);
    expect(html).toContain("No AOIs yet");
    expect(html).toContain("/dashboard/aoi/new");
  });

  it("renders each AOI row with link to its editor", () => {
    const html = renderToStaticMarkup(<AoiList rows={[baseRow]} />);
    expect(html).toContain("Spring Creek Preserve");
    expect(html).toContain(`/dashboard/aoi/${baseRow.id}`);
    expect(html).toContain("123.4");
    expect(html).toContain("2026-04-20");
    expect(html).toContain("active");
  });

  it("indicates paused status when pausedUntil is in the future", () => {
    const future = new Date(Date.now() + 86400_000);
    const html = renderToStaticMarkup(
      <AoiList rows={[{ ...baseRow, pausedUntil: future }]} />,
    );
    expect(html).toContain("paused");
  });
});
