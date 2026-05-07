/**
 * Unit tests for the pure helpers exported from `lib/firms/matcher.ts` via
 * `_internal`. These branches (MODIS numeric confidence, unknown-token
 * fail-open, 3-digit acqTime padding) are not exercised by the PostGIS
 * integration suite, which only runs the happy "n" / "0417" path.
 */
import { describe, expect, it } from "vitest";
import { _internal } from "@/lib/firms/matcher";
import type { FirmsDetection } from "@/lib/firms/client";

const { confidencePassesGate, detectionTimestamp } = _internal;

describe("confidencePassesGate", () => {
  it("treats VIIRS letter tokens against the AOI minimum", () => {
    expect(confidencePassesGate("h", "nominal")).toBe(true);
    expect(confidencePassesGate("n", "nominal")).toBe(true);
    expect(confidencePassesGate("l", "nominal")).toBe(false);
    expect(confidencePassesGate("L", "low")).toBe(true);
  });

  it("buckets MODIS numeric confidence at the 30/80 thresholds", () => {
    // <30 -> low, 30..79 -> nominal, >=80 -> high.
    expect(confidencePassesGate("0", "nominal")).toBe(false);
    expect(confidencePassesGate("29", "nominal")).toBe(false);
    expect(confidencePassesGate("30", "nominal")).toBe(true);
    expect(confidencePassesGate("79", "high")).toBe(false);
    expect(confidencePassesGate("80", "high")).toBe(true);
    expect(confidencePassesGate("100", "high")).toBe(true);
  });

  it("fails open as 'nominal' for unknown tokens (avoid silent data drop)", () => {
    expect(confidencePassesGate("garbage", "nominal")).toBe(true);
    expect(confidencePassesGate("garbage", "high")).toBe(false);
    expect(confidencePassesGate("garbage", "low")).toBe(true);
  });

  it("treats null confidence as passing only when AOI minimum is 'low'", () => {
    expect(confidencePassesGate(null, "low")).toBe(true);
    expect(confidencePassesGate(null, "nominal")).toBe(false);
    expect(confidencePassesGate(null, "high")).toBe(false);
  });
});

describe("detectionTimestamp", () => {
  function det(partial: Partial<FirmsDetection>): FirmsDetection {
    return {
      latitude: 0,
      longitude: 0,
      brightTi4: 0,
      brightTi5: 0,
      scan: 0,
      track: 0,
      acqDate: "2026-04-21",
      acqTime: "0417",
      satellite: "1",
      instrument: "VIIRS",
      confidence: "n",
      version: "2.0NRT",
      frp: 1,
      daynight: "N",
      ...partial,
    };
  }

  it("pads 3-digit acqTime (NASA drops the leading zero)", () => {
    // "417" must parse as 04:17 UTC, not 41:70 or NaN.
    const t = detectionTimestamp(det({ acqTime: "417" }));
    expect(t).not.toBeNull();
    expect(t!.toISOString()).toBe("2026-04-21T04:17:00.000Z");
  });

  it("parses 4-digit acqTime as UTC HHMM", () => {
    const t = detectionTimestamp(det({ acqDate: "2026-01-02", acqTime: "2359" }));
    expect(t!.toISOString()).toBe("2026-01-02T23:59:00.000Z");
  });

  it("returns null for missing date or time", () => {
    expect(detectionTimestamp(det({ acqDate: "" }))).toBeNull();
    expect(detectionTimestamp(det({ acqTime: "" }))).toBeNull();
  });

  it("returns null for an unparseable date", () => {
    expect(detectionTimestamp(det({ acqDate: "not-a-date" }))).toBeNull();
  });
});
