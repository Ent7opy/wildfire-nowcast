/**
 * Brief schema round-trip + markdown renderer snapshot.
 *
 * The fixture is synthetic but plausible — based on the SPEC §LLM brief format
 * "Worked example" (Spring Creek Preserve, Sonoma County). Labelled as a
 * fixture, not a real reading.
 */
import { describe, expect, it } from "vitest";
import { BriefSchema, type Brief, SCHEMA_VERSION } from "@/lib/ai/schema";
import { renderBriefMarkdown } from "@/lib/ai/render";

const FIXTURE: Brief = {
  schema_version: SCHEMA_VERSION,
  aoi: {
    id: "00000000-0000-4000-8000-000000000001",
    name: "Spring Creek Preserve (fixture)",
    area_ha: 2040,
  },
  summary:
    "Two VIIRS detections 14 km N of Spring Creek Preserve at 04:17 UTC, max FRP 11 MW. Wind blowing the head away for now.",
  key_facts: {
    nearest_detection_km: 14.0,
    bearing_from_aoi_deg: 357,
    wind_dir_deg: 240,
    wind_speed_kmh: 28,
    wind_toward_aoi: false,
    detection_count_in_window: 2,
    max_frp_mw: 11.0,
    satellites: ["VIIRS_NOAA20"],
    window_hours: 1,
  },
  context: {
    weather_note: "RH ~22%, winds 240° @ 28 km/h pushing activity ENE away from the preserve.",
    authority_perimeter: { source: null, posted_ts: null, contains_detection: null },
    prior_events: [
      {
        date: "2020-08-20",
        description: "LNU Lightning Complex eastern edge.",
        outcome:
          "Perimeter reached within 3 km of the preserve's north boundary; no incursion.",
      },
    ],
  },
  recommended_watch_items: [
    "Re-check at 06:00 local — overnight inversion breakup can flip local winds.",
    "Watch CAL FIRE SoCo incident page for a posted perimeter; none yet.",
  ],
  uncertainty:
    "No authority perimeter published yet. 2 pixels is the floor for us to brief.",
  next_brief_hint: {
    when: "on polygon breach, else 06:00 local digest",
    trigger: "new detection < 10 km OR authority perimeter published",
  },
};

describe("BriefSchema round-trip", () => {
  it("accepts the worked-example fixture", () => {
    const parsed = BriefSchema.parse(FIXTURE);
    expect(parsed.schema_version).toBe(SCHEMA_VERSION);
    expect(parsed.aoi.name).toBe("Spring Creek Preserve (fixture)");
  });

  it("rejects a wrong schema_version", () => {
    const r = BriefSchema.safeParse({ ...FIXTURE, schema_version: 2 });
    expect(r.success).toBe(false);
  });

  it("rejects bearing outside [0,360]", () => {
    const r = BriefSchema.safeParse({
      ...FIXTURE,
      key_facts: { ...FIXTURE.key_facts, bearing_from_aoi_deg: 720 },
    });
    expect(r.success).toBe(false);
  });

  it("accepts an empty satellites list (no fabrication when DB has none)", () => {
    const r = BriefSchema.safeParse({
      ...FIXTURE,
      key_facts: { ...FIXTURE.key_facts, satellites: [] },
    });
    expect(r.success).toBe(true);
  });

  it("rejects a malformed prior_events.date", () => {
    const r = BriefSchema.safeParse({
      ...FIXTURE,
      context: {
        ...FIXTURE.context,
        prior_events: [
          { date: "08-20-2020", description: "x", outcome: null },
        ],
      },
    });
    expect(r.success).toBe(false);
  });
});

describe("renderBriefMarkdown — snapshot of the worked example", () => {
  it("produces deterministic markdown", () => {
    const md = renderBriefMarkdown(FIXTURE);
    expect(md).toMatchInlineSnapshot(`
      "# Spring Creek Preserve (fixture) — situation brief

      Two VIIRS detections 14 km N of Spring Creek Preserve at 04:17 UTC, max FRP 11 MW. Wind blowing the head away for now.

      ## Key facts
      - Nearest detection: 14.0 km @ 357° (N)
      - Wind: 240° @ 28 km/h (away from AOI)
      - Detections in 1 h window: 2 (max FRP 11.0 MW)
      - Satellites: VIIRS_NOAA20

      ## Context
      RH ~22%, winds 240° @ 28 km/h pushing activity ENE away from the preserve.

      Authority perimeter: none posted yet.
      Prior events:
      - 2020-08-20 — LNU Lightning Complex eastern edge. Perimeter reached within 3 km of the preserve's north boundary; no incursion.

      ## What to watch
      - Re-check at 06:00 local — overnight inversion breakup can flip local winds.
      - Watch CAL FIRE SoCo incident page for a posted perimeter; none yet.

      _Uncertainty: No authority perimeter published yet. 2 pixels is the floor for us to brief._

      _Next brief: on polygon breach, else 06:00 local digest (trigger: new detection < 10 km OR authority perimeter published)._"
    `);
  });

  it("falls back gracefully when wind / perimeter / prior_events are absent", () => {
    const minimal: Brief = {
      ...FIXTURE,
      key_facts: {
        ...FIXTURE.key_facts,
        wind_dir_deg: null,
        wind_speed_kmh: null,
        wind_toward_aoi: null,
        max_frp_mw: null,
      },
      context: {
        weather_note: null,
        authority_perimeter: { source: null, posted_ts: null, contains_detection: null },
        prior_events: [],
      },
      recommended_watch_items: [],
    };
    const md = renderBriefMarkdown(minimal);
    expect(md).toContain("Wind: unavailable");
    expect(md).toContain("Authority perimeter: none posted yet.");
    expect(md).toContain("Prior events: no prior events on file.");
    expect(md).toContain("- (none)");
  });
});
