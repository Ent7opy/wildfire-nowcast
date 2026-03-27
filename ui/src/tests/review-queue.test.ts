import { describe, expect, it } from "vitest";

import { FIRE_COLORS, toRenderEvent } from "../map/layerUtils";
import type { FireEvent } from "../types/api";

describe("review queue visual style", () => {
  const baseEvent: FireEvent = {
    event_id: "evt-1",
    lat: 34.5,
    lon: -118.2,
    event_score: 0.9,
    denoiser_decision: "pass",
    review_required: false,
    detection_count: 5
  };

  it("renders normal high-severity event with red fill", () => {
    const rendered = toRenderEvent(baseEvent);
    expect(rendered).not.toBeNull();
    // High severity → red fill (veryHighFill[0] = 220)
    expect(rendered!.fill_r).toBe(FIRE_COLORS.veryHighFill[0]);
    expect(rendered!.review_required).toBe(false);
  });

  it("renders review_required event with amber fill", () => {
    const reviewEvent: FireEvent = { ...baseEvent, review_required: true };
    const rendered = toRenderEvent(reviewEvent);
    expect(rendered).not.toBeNull();
    expect(rendered!.fill_r).toBe(FIRE_COLORS.reviewFill[0]);
    expect(rendered!.fill_g).toBe(FIRE_COLORS.reviewFill[1]);
    expect(rendered!.fill_b).toBe(FIRE_COLORS.reviewFill[2]);
  });

  it("renders review_required event with amber outline", () => {
    const reviewEvent: FireEvent = { ...baseEvent, review_required: true };
    const rendered = toRenderEvent(reviewEvent);
    expect(rendered).not.toBeNull();
    expect(rendered!.line_r).toBe(FIRE_COLORS.reviewOutline[0]);
    expect(rendered!.line_g).toBe(FIRE_COLORS.reviewOutline[1]);
    expect(rendered!.line_b).toBe(FIRE_COLORS.reviewOutline[2]);
  });

  it("review amber fill differs from normal high-severity fill", () => {
    expect(FIRE_COLORS.reviewFill[0]).not.toBe(FIRE_COLORS.veryHighFill[0]);
  });

  it("returns null for event missing lat/lon", () => {
    const noCoords: FireEvent = { event_id: "evt-x", review_required: true };
    expect(toRenderEvent(noCoords)).toBeNull();
  });

  it("preserves review_required flag on rendered event", () => {
    const reviewEvent: FireEvent = { ...baseEvent, review_required: true };
    const rendered = toRenderEvent(reviewEvent);
    expect(rendered!.review_required).toBe(true);
  });

  it("low-severity review event still gets amber color, not severity-based color", () => {
    const lowReview: FireEvent = {
      ...baseEvent,
      event_score: 0.1,
      review_required: true
    };
    const rendered = toRenderEvent(lowReview);
    expect(rendered).not.toBeNull();
    // Should be amber, not the low-severity veryLowFill
    expect(rendered!.fill_r).toBe(FIRE_COLORS.reviewFill[0]);
    expect(rendered!.fill_r).not.toBe(FIRE_COLORS.veryLowFill[0]);
  });
});
