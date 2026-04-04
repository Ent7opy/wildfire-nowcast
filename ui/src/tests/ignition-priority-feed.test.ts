import { describe, expect, it } from "vitest";

import {
  firstCriticalCellWithoutNearbyFire,
  highOrCriticalCellsNear,
} from "../utils/ignition";
import type { IgnitionCell } from "../types/api";
import type { FireEvent } from "../types/api";

// ── helpers ──────────────────────────────────────────────────────────────────

function cell(lat: number, lon: number, level: IgnitionCell["level"]): IgnitionCell {
  return { lat, lon, probability: level === "critical" ? 0.95 : level === "high" ? 0.75 : level === "elevated" ? 0.45 : 0.1, level };
}

function fire(lat: number, lon: number): FireEvent {
  return { event_id: `${lat},${lon}`, lat, lon, event_score: 0.9 };
}

// ── firstCriticalCellWithoutNearbyFire ────────────────────────────────────────

describe("firstCriticalCellWithoutNearbyFire", () => {
  it("returns null when there are no critical cells", () => {
    const cells = [cell(34, -118, "elevated"), cell(35, -117, "high")];
    expect(firstCriticalCellWithoutNearbyFire(cells, [])).toBeNull();
  });

  it("returns null when all critical cells have a fire within 50 km", () => {
    // 34.0, -118.0  →  fire at 34.1, -118.0 is ~11 km away
    const cells = [cell(34.0, -118.0, "critical")];
    const fires = [fire(34.1, -118.0)];
    expect(firstCriticalCellWithoutNearbyFire(cells, fires)).toBeNull();
  });

  it("returns the critical cell when no fire is within 50 km", () => {
    // Cell in Los Angeles; fire >50 km away in San Diego
    const criticalCell = cell(34.05, -118.24, "critical");
    const fires = [fire(32.72, -117.16)]; // San Diego, ~180 km away
    const result = firstCriticalCellWithoutNearbyFire([criticalCell], fires);
    expect(result).toBe(criticalCell);
  });

  it("skips non-critical cells even when no fire is nearby", () => {
    const cells = [cell(34.0, -118.0, "high"), cell(35.0, -119.0, "elevated")];
    expect(firstCriticalCellWithoutNearbyFire(cells, [])).toBeNull();
  });

  it("returns first critical cell with no nearby fire when multiple exist", () => {
    const c1 = cell(34.0, -118.0, "critical");
    const c2 = cell(40.0, -105.0, "critical"); // Denver area, no fires
    const fires = [fire(34.1, -118.0)];        // covers c1 but not c2
    const result = firstCriticalCellWithoutNearbyFire([c1, c2], fires);
    expect(result).toBe(c2);
  });

  it("uses the supplied radiusKm override", () => {
    const criticalCell = cell(34.0, -118.0, "critical");
    // Fire ~11 km away
    const fires = [fire(34.1, -118.0)];
    // With 5 km radius the fire is outside → should return the cell
    expect(firstCriticalCellWithoutNearbyFire([criticalCell], fires, 5)).toBe(criticalCell);
    // With 50 km radius the fire is inside → should return null
    expect(firstCriticalCellWithoutNearbyFire([criticalCell], fires, 50)).toBeNull();
  });

  it("returns null when fires array is empty but there are no critical cells", () => {
    expect(firstCriticalCellWithoutNearbyFire([], [])).toBeNull();
  });

  it("returns a critical cell when fires array is empty", () => {
    const c = cell(50.0, 10.0, "critical");
    expect(firstCriticalCellWithoutNearbyFire([c], [])).toBe(c);
  });
});

// ── highOrCriticalCellsNear ───────────────────────────────────────────────────

describe("highOrCriticalCellsNear", () => {
  it("returns 0 when cells array is empty", () => {
    expect(highOrCriticalCellsNear([], 34.0, -118.0)).toBe(0);
  });

  it("counts only high and critical cells within the radius", () => {
    const cells = [
      cell(34.05, -118.2, "critical"),  // very close
      cell(34.1,  -118.2, "high"),      // close
      cell(34.2,  -118.0, "elevated"),  // close but not high/critical
      cell(34.15, -118.1, "low"),       // close but low
    ];
    expect(highOrCriticalCellsNear(cells, 34.0, -118.24, 50)).toBe(2);
  });

  it("excludes cells outside the radius", () => {
    const cells = [
      cell(34.0, -118.0, "critical"),  // ~within 1 km of query point
      cell(40.0, -105.0, "high"),      // Denver — very far from LA
    ];
    expect(highOrCriticalCellsNear(cells, 34.0, -118.0, 50)).toBe(1);
  });

  it("counts both high and critical cells", () => {
    const cells = [
      cell(34.01, -118.01, "critical"),
      cell(34.02, -118.02, "high"),
      cell(34.03, -118.03, "high"),
    ];
    expect(highOrCriticalCellsNear(cells, 34.0, -118.0, 50)).toBe(3);
  });

  it("respects a custom radius", () => {
    // Cell is ~11 km away
    const cells = [cell(34.1, -118.0, "critical")];
    expect(highOrCriticalCellsNear(cells, 34.0, -118.0, 5)).toBe(0);
    expect(highOrCriticalCellsNear(cells, 34.0, -118.0, 50)).toBe(1);
  });
});
