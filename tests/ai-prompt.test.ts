/**
 * Snapshot tests for `lib/ai/prompt.ts`.
 *
 * The prompt strings encode anti-fabrication instructions that the LLM relies
 * on (no inventing weather, authority perimeters, or bearings). A silent
 * deletion of one of those lines would degrade brief quality without any other
 * test failing — `generate.ts` exercises this transitively but does not pin
 * the exact text. These inline snapshots make the contract visible in-file.
 */
import { describe, expect, it } from "vitest";
import { SYSTEM_PROMPT, buildUserPrompt, type BriefContext } from "@/lib/ai/prompt";

const BASE: BriefContext = {
  aoi: { id: "00000000-0000-4000-8000-000000000001", name: "Spring Creek Preserve", areaHa: 2040 },
  event: {
    nearestDistanceKm: 14,
    bearingFromAoiDeg: 357,
    detectionCount: 1,
    peakFrpMw: 11,
    windowHours: 1,
    satellites: ["VIIRS_NOAA20"],
    firstSeenAt: "2026-04-01T04:17:00Z",
    lastSeenAt: "2026-04-01T04:17:00Z",
  },
  weather: null,
  authorityPerimeter: null,
  priorEvents: [
    { date: "2024-08-12", description: "Lightning ignition 6 km W", outcome: "contained at 3 ha" },
  ],
};

describe("SYSTEM_PROMPT", () => {
  it("pins the anti-fabrication ruleset", () => {
    expect(SYSTEM_PROMPT).toMatchInlineSnapshot(`
      "You are the Wildfire Nowcast situation-brief writer.
      You produce a single L2-style brief for one Area of Interest (AOI), in valid JSON conforming to the provided schema.

      Rules:
      - Do NOT invent values. If an input is null (e.g. weather, authority_perimeter), reflect that faithfully.
      - The summary is 1–2 sentences, like a staffer's radio report.
      - The uncertainty field is mandatory; be explicit about what is NOT known.
      - Recommended watch items are concrete observations the steward can make, not imperatives.
      - Numeric fields (distances, bearings, FRP, counts) MUST equal the values supplied in the user message; do not round or transform them.
      - The schema_version field must be exactly 1."
    `);
  });
});

describe("buildUserPrompt", () => {
  it("renders happy-path context (one detection, weather/authority null, one prior event)", () => {
    expect(buildUserPrompt(BASE)).toMatchInlineSnapshot(`
      "AOI: Spring Creek Preserve (id=00000000-0000-4000-8000-000000000001, area_ha=2040)

      Event:
        nearest_detection_km: 14
        bearing_from_aoi_deg: 357
        detection_count_in_window: 1
        max_frp_mw: 11
        window_hours: 1
        satellites: [VIIRS_NOAA20]
        first_seen_at: 2026-04-01T04:17:00Z
        last_seen_at: 2026-04-01T04:17:00Z

      Weather note (null if no data): null

      Authority perimeter:
        source: null
        posted_ts: null
        contains_detection: null

      Prior events on file:
        - 2024-08-12: Lightning ignition 6 km W — contained at 3 ha

      Produce the JSON brief. wind_dir_deg / wind_speed_kmh / wind_toward_aoi must be null unless wind data is provided above (currently they are not). bearing_from_aoi_deg must be null if the value above is null — do not invent a direction."
    `);
  });

  it("renders all-nullable-null context (no priors, no weather, no authority, null bearing/FRP)", () => {
    const ctx: BriefContext = {
      ...BASE,
      event: { ...BASE.event, bearingFromAoiDeg: null, peakFrpMw: null, satellites: [] },
      weather: null,
      authorityPerimeter: null,
      priorEvents: [],
    };
    expect(buildUserPrompt(ctx)).toMatchInlineSnapshot(`
      "AOI: Spring Creek Preserve (id=00000000-0000-4000-8000-000000000001, area_ha=2040)

      Event:
        nearest_detection_km: 14
        bearing_from_aoi_deg: null
        detection_count_in_window: 1
        max_frp_mw: null
        window_hours: 1
        satellites: []
        first_seen_at: 2026-04-01T04:17:00Z
        last_seen_at: 2026-04-01T04:17:00Z

      Weather note (null if no data): null

      Authority perimeter:
        source: null
        posted_ts: null
        contains_detection: null

      Prior events on file: none.

      Produce the JSON brief. wind_dir_deg / wind_speed_kmh / wind_toward_aoi must be null unless wind data is provided above (currently they are not). bearing_from_aoi_deg must be null if the value above is null — do not invent a direction."
    `);
  });

  it("includes authority perimeter when populated (not silently dropped)", () => {
    const ctx: BriefContext = {
      ...BASE,
      authorityPerimeter: {
        source: "CAL FIRE IRWIN",
        postedTs: "2026-04-01T05:00:00Z",
        containsDetection: true,
      },
      weather: { note: "RH 18%, gusts 35 km/h from SW" },
      priorEvents: [],
    };
    const out = buildUserPrompt(ctx);
    expect(out).toContain("source: CAL FIRE IRWIN");
    expect(out).toContain("posted_ts: 2026-04-01T05:00:00Z");
    expect(out).toContain("contains_detection: true");
    expect(out).toContain("Weather note (null if no data): RH 18%, gusts 35 km/h from SW");
  });

  it("includes all prior events in order with outcomes when present", () => {
    const ctx: BriefContext = {
      ...BASE,
      priorEvents: [
        { date: "2023-07-04", description: "Roadside ignition 2 km S", outcome: "contained at 1 ha" },
        { date: "2024-08-12", description: "Lightning ignition 6 km W", outcome: null },
        { date: "2025-09-21", description: "Escaped pile burn 4 km N", outcome: "8 ha mosaic" },
      ],
    };
    const out = buildUserPrompt(ctx);
    expect(out).toContain("  - 2023-07-04: Roadside ignition 2 km S — contained at 1 ha");
    expect(out).toContain("  - 2024-08-12: Lightning ignition 6 km W");
    expect(out).not.toContain("2024-08-12: Lightning ignition 6 km W —");
    expect(out).toContain("  - 2025-09-21: Escaped pile burn 4 km N — 8 ha mosaic");
  });
});
