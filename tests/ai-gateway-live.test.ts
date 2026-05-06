/**
 * Live AI Gateway smoke test — gated behind `AI_GATEWAY_LIVE=1`.
 *
 * Off by default (CI and local). Run manually after Vanyo wires up
 * `AI_GATEWAY_API_KEY` to confirm real Gemini 2.5 Flash-Lite via the gateway
 * returns a schema-valid object for our prompt:
 *
 *   AI_GATEWAY_LIVE=1 AI_GATEWAY_API_KEY=... pnpm vitest run tests/ai-gateway-live
 *
 * Skipped silently when the flag isn't set so `pnpm test` stays clean.
 */
import { describe, expect, it } from "vitest";
import { generateBriefViaGateway } from "@/lib/ai/gateway";
import { SYSTEM_PROMPT, buildUserPrompt } from "@/lib/ai/prompt";

const live = process.env.AI_GATEWAY_LIVE === "1";
const describeLive = live ? describe : describe.skip;

if (!live) {
  console.warn(
    "[live] Skipping AI Gateway live test — set AI_GATEWAY_LIVE=1 (and AI_GATEWAY_API_KEY) to enable.",
  );
}

describeLive("AI Gateway live — Gemini 2.5 Flash-Lite", () => {
  it("returns a schema-valid brief for the worked-example context", async () => {
    const userPrompt = buildUserPrompt({
      aoi: {
        id: "00000000-0000-4000-8000-000000000001",
        name: "Spring Creek Preserve (live test fixture)",
        areaHa: 2040,
      },
      event: {
        nearestDistanceKm: 14,
        bearingFromAoiDeg: 357,
        detectionCount: 2,
        peakFrpMw: 11,
        windowHours: 1,
        satellites: ["VIIRS_NOAA20_NRT"],
        firstSeenAt: "2026-04-21T04:17:00Z",
        lastSeenAt: "2026-04-21T04:18:00Z",
      },
      weather: null,
      authorityPerimeter: null,
      priorEvents: [
        {
          date: "2020-08-20",
          description: "LNU Lightning Complex eastern edge.",
          outcome: "Perimeter reached within 3 km of the preserve's north boundary; no incursion.",
        },
      ],
    });
    const result = await generateBriefViaGateway({
      systemPrompt: SYSTEM_PROMPT,
      userPrompt,
      timeoutMs: 30_000,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.brief.schema_version).toBe(1);
    expect(result.brief.aoi.name).toContain("Spring Creek");
    expect(result.brief.summary.length).toBeGreaterThan(20);
    expect(result.brief.uncertainty.length).toBeGreaterThan(0);
  }, 60_000);
});
