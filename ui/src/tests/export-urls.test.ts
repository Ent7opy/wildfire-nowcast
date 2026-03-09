import { describe, expect, it } from "vitest";

import { buildFiresCsvExportUrl, buildMapPngExportUrl } from "../api/client";

describe("export URLs", () => {
  const start = new Date("2026-03-01T00:00:00Z");
  const end = new Date("2026-03-01T12:00:00Z");

  it("builds fire CSV export URL", () => {
    const url = buildFiresCsvExportUrl("http://localhost:8000", {
      bbox: [20, 40, 21, 41],
      startTime: start,
      endTime: end,
      limit: 1000
    });

    const parsed = new URL(url);
    expect(parsed.pathname).toBe("/fires/export");
    expect(parsed.searchParams.get("format")).toBe("csv");
    expect(parsed.searchParams.get("limit")).toBe("1000");
  });

  it("builds map PNG export URL with run id", () => {
    const url = buildMapPngExportUrl("http://localhost:8000", {
      bbox: [20, 40, 21, 41],
      startTime: start,
      endTime: end,
      minLikelihood: 0.7,
      includeRisk: true,
      runId: "abc123"
    });

    const parsed = new URL(url);
    expect(parsed.pathname).toBe("/map.png");
    expect(parsed.searchParams.get("min_fire_likelihood")).toBe("0.70");
    expect(parsed.searchParams.get("include_risk")).toBe("true");
    expect(parsed.searchParams.get("run_id")).toBe("abc123");
  });
});
