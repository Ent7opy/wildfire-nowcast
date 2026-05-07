/**
 * Stage 8 — live endpoint smoke. Skipped unless AUTHORITY_PERIMETER_LIVE=1
 * (mirrors the FIRMS_LIVE pattern). Run before merge to catch endpoint drift.
 *
 * Each source is hit with a known recent-fire region. Returning null is OK
 * (no active fires today); throwing or rejecting is not — that surfaces an
 * endpoint regression we should file as a blocker.
 */
import { describe, it, expect } from "vitest";
import { fetchAuthorityPerimeter } from "@/lib/ai/authority/fetch";

const live = process.env.AUTHORITY_PERIMETER_LIVE === "1";
const d = live ? describe : describe.skip;

d("authority perimeter live endpoints (AUTHORITY_PERIMETER_LIVE=1)", () => {
  it("NIFC WFIGS — Northern California region", async () => {
    const r = await fetchAuthorityPerimeter({
      lat: 38.5,
      lon: -122.7,
      regionBucket: "5x5:W125_N35",
      radiusKm: 500,
      timeoutMs: 30_000,
    });
    // Either a hit (active fire near point) or null (none in radius). Both pass —
    // we only fail on thrown errors or non-FeatureCollection bodies.
    expect(r === null || typeof r.source === "string").toBe(true);
  }, 35_000);

  it("CWFIS — central Canada region", async () => {
    const r = await fetchAuthorityPerimeter({
      lat: 53.5,
      lon: -113.5,
      regionBucket: "5x5:W115_N50",
      radiusKm: 500,
      timeoutMs: 30_000,
    });
    expect(r === null || typeof r.source === "string").toBe(true);
  }, 35_000);
});
