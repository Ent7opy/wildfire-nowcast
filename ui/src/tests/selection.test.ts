import { describe, expect, it } from "vitest";

import { normalizePickedEvent } from "../utils/selection";

describe("selection normalization", () => {
  it("returns null for non-object picks", () => {
    expect(normalizePickedEvent(null)).toBeNull();
    expect(normalizePickedEvent("x")).toBeNull();
  });

  it("extracts feature properties and validates lat/lon", () => {
    const normalized = normalizePickedEvent({
      properties: {
        event_id: "evt-1",
        lat: "42.1",
        lon: "23.5"
      }
    });

    expect(normalized).not.toBeNull();
    expect(normalized?.event_id).toBe("evt-1");
    expect(normalized?.lat).toBe(42.1);
    expect(normalized?.lon).toBe(23.5);
  });
});
