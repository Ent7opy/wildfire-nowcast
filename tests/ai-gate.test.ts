/**
 * Pure-TS tests for the Stage 3 LLM gate. Covers every branch in
 * docs/SPEC-A-prime-v1.md §Flow 6 plus the paused / already-briefed rejects.
 */
import { describe, expect, it } from "vitest";
import { evaluateGate, type GateInputs } from "@/lib/ai/gate";

const NOW = new Date("2026-04-21T05:00:00Z");

function inputs(overrides: Partial<GateInputs> = {}): GateInputs {
  return {
    pausedUntil: null,
    lastBriefAt: null,
    lastAoiEventBriefedAt: null,
    detectionCountInEvent: 1,
    peakFrpMw: 1.0,
    nearestDistanceKm: 20,
    alertDistanceKm: 25,
    minFrpMw: 5,
    now: NOW,
    ...overrides,
  };
}

describe("evaluateGate — reject conditions", () => {
  it("rejects when AOI is paused", () => {
    const r = evaluateGate(
      inputs({
        pausedUntil: new Date("2026-04-22T00:00:00Z"),
        // Even when other conditions would pass, paused short-circuits.
        detectionCountInEvent: 5,
      }),
    );
    expect(r).toEqual({ pass: false, reason: "paused" });
  });

  it("treats a paused_until in the past as not paused", () => {
    const r = evaluateGate(
      inputs({
        pausedUntil: new Date("2026-04-20T00:00:00Z"),
        // Other conditions still need to pass; prior_absence does because
        // lastAoiEventBriefedAt is null.
      }),
    );
    expect(r.pass).toBe(true);
  });

  it("rejects when this event already has a brief (already_briefed)", () => {
    const r = evaluateGate(
      inputs({
        lastBriefAt: new Date("2026-04-21T04:30:00Z"),
        detectionCountInEvent: 5, // would otherwise pass multi_pixel
      }),
    );
    expect(r).toEqual({ pass: false, reason: "already_briefed" });
  });
});

describe("evaluateGate — pass conditions", () => {
  it("passes prior_absence when no prior brief on file", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: null,
        // Nothing else qualifies — single pixel, low FRP, far from AOI.
        peakFrpMw: 0.1,
        nearestDistanceKm: 24,
      }),
    );
    expect(r).toEqual({ pass: true, reason: "prior_absence" });
  });

  it("passes prior_absence when last brief was > 72h ago", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-17T05:00:00Z"), // 96h ago
        peakFrpMw: 0.1,
        nearestDistanceKm: 24,
      }),
    );
    expect(r).toEqual({ pass: true, reason: "prior_absence" });
  });

  it("passes multi_pixel when detection count >= 2 (even with recent prior brief)", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-21T03:00:00Z"), // 2h ago
        detectionCountInEvent: 2,
      }),
    );
    expect(r).toEqual({ pass: true, reason: "multi_pixel" });
  });

  it("passes high_frp when peak FRP exceeds the AOI's min_frp_mw", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-21T03:00:00Z"),
        detectionCountInEvent: 1,
        peakFrpMw: 11,
        minFrpMw: 5,
        nearestDistanceKm: 24,
      }),
    );
    expect(r).toEqual({ pass: true, reason: "high_frp" });
  });

  it("does not pass high_frp when FRP equals the threshold", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-21T03:00:00Z"),
        detectionCountInEvent: 1,
        peakFrpMw: 5,
        minFrpMw: 5,
        nearestDistanceKm: 24,
      }),
    );
    expect(r).toEqual({ pass: false, reason: "no_signal" });
  });

  it("passes close_proximity when distance ≤ 0.5 × alert_distance_km", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-21T03:00:00Z"),
        detectionCountInEvent: 1,
        peakFrpMw: 0.1,
        nearestDistanceKm: 12, // 0.5 × 25 = 12.5
        alertDistanceKm: 25,
      }),
    );
    expect(r).toEqual({ pass: true, reason: "close_proximity" });
  });
});

describe("evaluateGate — no-signal", () => {
  it("rejects when nothing qualifies", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-21T03:00:00Z"), // recent brief
        detectionCountInEvent: 1,
        peakFrpMw: 0.5,
        minFrpMw: 5,
        nearestDistanceKm: 24, // > 12.5 km half-buffer
        alertDistanceKm: 25,
      }),
    );
    expect(r).toEqual({ pass: false, reason: "no_signal" });
  });
});

// FIRMS occasionally returns detections with NULL FRP. Lock in how each
// pass-condition branch behaves when peakFrpMw is null, so a future refactor
// that removes the null guard fails loudly here.
describe("evaluateGate — peakFrpMw null branches", () => {
  it("passes multi_pixel when peakFrpMw is null (pixel count, not FRP)", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-21T03:00:00Z"), // recent brief, skip prior_absence
        detectionCountInEvent: 2,
        peakFrpMw: null,
      }),
    );
    expect(r).toEqual({ pass: true, reason: "multi_pixel" });
  });

  it("does NOT pass high_frp when peakFrpMw is null", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-21T03:00:00Z"),
        detectionCountInEvent: 1,
        peakFrpMw: null,
        minFrpMw: 5,
        nearestDistanceKm: 24, // > 12.5 km half-buffer, skip close_proximity
        alertDistanceKm: 25,
      }),
    );
    expect(r).toEqual({ pass: false, reason: "no_signal" });
  });

  it("passes prior_absence when peakFrpMw is null and no prior brief", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: null,
        detectionCountInEvent: 1,
        peakFrpMw: null,
        nearestDistanceKm: 24,
      }),
    );
    expect(r).toEqual({ pass: true, reason: "prior_absence" });
  });

  it("passes prior_absence when peakFrpMw is null and last brief > 72h ago", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-17T05:00:00Z"), // 96h ago
        detectionCountInEvent: 1,
        peakFrpMw: null,
        nearestDistanceKm: 24,
      }),
    );
    expect(r).toEqual({ pass: true, reason: "prior_absence" });
  });

  it("passes close_proximity when peakFrpMw is null (distance-based)", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-21T03:00:00Z"),
        detectionCountInEvent: 1,
        peakFrpMw: null,
        nearestDistanceKm: 12, // 0.5 × 25 = 12.5
        alertDistanceKm: 25,
      }),
    );
    expect(r).toEqual({ pass: true, reason: "close_proximity" });
  });

  it("rejects with no_signal when peakFrpMw is null and no other condition fires", () => {
    const r = evaluateGate(
      inputs({
        lastAoiEventBriefedAt: new Date("2026-04-21T03:00:00Z"), // recent
        detectionCountInEvent: 1,
        peakFrpMw: null,
        minFrpMw: 5,
        nearestDistanceKm: 24, // > half-buffer
        alertDistanceKm: 25,
      }),
    );
    expect(r).toEqual({ pass: false, reason: "no_signal" });
  });
});
