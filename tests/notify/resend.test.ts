/**
 * Stage 4 Resend client unit tests — pure envelope + config_missing path.
 * No live HTTP. Subject truncation, markdown→html, RESEND_TEST_MODE flag,
 * and missing-key behaviour.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildEnvelope, sendEmail, truncate } from "@/lib/notify/resend";
import { renderMarkdownToHtml } from "@/lib/notify/markdown";
import { renderBriefMarkdown } from "@/lib/ai/render";
import type { Brief } from "@/lib/ai/schema";
import { SCHEMA_VERSION } from "@/lib/ai/schema";

const SPRING_CREEK: Brief = {
  schema_version: SCHEMA_VERSION,
  aoi: {
    id: "00000000-0000-4000-8000-000000000099",
    name: "Spring Creek Preserve",
    area_ha: 2040,
  },
  summary: "Two VIIRS detections 14 km north of the preserve; light winds, no incursion.",
  key_facts: {
    nearest_detection_km: 14,
    bearing_from_aoi_deg: 357,
    wind_dir_deg: null,
    wind_speed_kmh: null,
    wind_toward_aoi: null,
    detection_count_in_window: 2,
    max_frp_mw: 11,
    satellites: ["VIIRS_NOAA20_NRT"],
    window_hours: 1,
  },
  context: {
    weather_note: null,
    authority_perimeter: { source: null, posted_ts: null, contains_detection: null },
    prior_events: [
      {
        date: "2020-08-20",
        description: "LNU Lightning Complex eastern edge.",
        outcome: "no incursion",
      },
    ],
  },
  recommended_watch_items: ["Re-check at next cron tick.", "Monitor wind shift."],
  uncertainty: "Confidence is nominal.",
  next_brief_hint: { when: "next tick", trigger: "any new detection" },
};

describe("Resend client envelope", () => {
  const ENV_KEYS = ["RESEND_API_KEY", "RESEND_TEST_MODE", "NOTIFY_FROM_ADDRESS"] as const;
  const original: Partial<Record<(typeof ENV_KEYS)[number], string | undefined>> = {};

  beforeEach(() => {
    for (const k of ENV_KEYS) original[k] = process.env[k];
    for (const k of ENV_KEYS) delete process.env[k];
  });

  afterEach(() => {
    for (const k of ENV_KEYS) {
      const v = original[k];
      if (v == null) delete process.env[k];
      else process.env[k] = v;
    }
  });

  it("truncates subject at 90 chars", () => {
    const long = "a".repeat(120);
    expect(truncate(long, 90)).toHaveLength(90);
    const env = buildEnvelope({
      to: "x@example.org",
      subject: long,
      markdown: "body",
    });
    expect(env.subject.length).toBeLessThanOrEqual(90);
  });

  it("renders the canonical Spring Creek brief markdown to HTML", () => {
    const md = renderBriefMarkdown(SPRING_CREEK);
    const html = renderMarkdownToHtml(md);
    expect(html).toContain("<h1>Spring Creek Preserve — situation brief</h1>");
    expect(html).toContain("<h2>Key facts</h2>");
    expect(html).toContain("<ul>");
    expect(html).toContain("Re-check at next cron tick.");
    expect(html).toContain("<em>Uncertainty: Confidence is nominal.</em>");
  });

  it("RESEND_TEST_MODE=1 rewrites from and adds [TEST] suffix", () => {
    process.env.RESEND_TEST_MODE = "1";
    process.env.NOTIFY_FROM_ADDRESS = "alerts@configured.example";
    const env = buildEnvelope({
      to: "x@example.org",
      subject: "Hello",
      markdown: "body",
    });
    expect(env.from).toBe("onboarding@resend.dev");
    expect(env.subject).toBe("Hello [TEST]");
  });

  it("uses NOTIFY_FROM_ADDRESS when configured and not in test mode", () => {
    process.env.NOTIFY_FROM_ADDRESS = "alerts@configured.example";
    const env = buildEnvelope({
      to: "x@example.org",
      subject: "Hello",
      markdown: "body",
    });
    expect(env.from).toBe("alerts@configured.example");
    expect(env.subject).toBe("Hello");
  });

  it("missing RESEND_API_KEY → config_missing, no throw", async () => {
    const result = await sendEmail({
      to: "x@example.org",
      subject: "Hello",
      markdown: "body",
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.code).toBe("config_missing");
    }
  });
});
