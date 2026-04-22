import { describe, expect, it } from "vitest";
import { aoiCreateSchema, rulesUpsertSchema } from "@/lib/validators/aoi";

const VALID_POLYGON = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-122.7, 38.4],
      [-122.6, 38.4],
      [-122.6, 38.5],
      [-122.7, 38.5],
      [-122.7, 38.4],
    ],
  ],
};

describe("aoiCreateSchema", () => {
  it("accepts a well-formed Polygon", () => {
    const out = aoiCreateSchema.parse({
      name: "Spring Creek",
      geometry: VALID_POLYGON,
    });
    expect(out.geometry.type).toBe("Polygon");
  });

  it("accepts a MultiPolygon", () => {
    const out = aoiCreateSchema.parse({
      name: "Test MP",
      geometry: { type: "MultiPolygon", coordinates: [VALID_POLYGON.coordinates] },
    });
    expect(out.geometry.type).toBe("MultiPolygon");
  });

  it("rejects an open (unclosed) ring", () => {
    const open = {
      type: "Polygon" as const,
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
        ],
      ],
    };
    expect(() => aoiCreateSchema.parse({ name: "x", geometry: open })).toThrow();
  });

  it("rejects out-of-range coordinates", () => {
    const bad = {
      type: "Polygon" as const,
      coordinates: [
        [
          [200, 0],
          [201, 0],
          [201, 1],
          [200, 1],
          [200, 0],
        ],
      ],
    };
    expect(() => aoiCreateSchema.parse({ name: "x", geometry: bad })).toThrow();
  });

  it("rejects a missing name", () => {
    expect(() =>
      aoiCreateSchema.parse({ name: "", geometry: VALID_POLYGON }),
    ).toThrow();
  });
});

describe("rulesUpsertSchema", () => {
  it("applies defaults when fields are omitted", () => {
    const out = rulesUpsertSchema.parse({});
    expect(out.distanceBufferKm).toBe(25);
    expect(out.minConfidence).toBe("nominal");
    expect(out.minFrpMw).toBe(5);
    expect(out.quietHours).toBeNull();
    expect(out.notifyChannels).toEqual([]);
  });

  it("accepts an email channel", () => {
    const out = rulesUpsertSchema.parse({
      notifyChannels: [{ type: "email", target: "ranger@example.org" }],
    });
    expect(out.notifyChannels[0].type).toBe("email");
  });

  it("accepts a webhook channel", () => {
    const out = rulesUpsertSchema.parse({
      notifyChannels: [{ type: "webhook", target: "https://hooks.example.org/x" }],
    });
    expect(out.notifyChannels[0].type).toBe("webhook");
  });

  it("rejects an unsupported channel type", () => {
    expect(() =>
      rulesUpsertSchema.parse({
        notifyChannels: [{ type: "sms", target: "+15551234567" }],
      }),
    ).toThrow();
  });

  it("accepts quiet hours with a valid IANA tz", () => {
    const out = rulesUpsertSchema.parse({
      quietHours: { tz: "America/Los_Angeles", startHour: 22, endHour: 7 },
    });
    expect(out.quietHours?.tz).toBe("America/Los_Angeles");
  });
});
