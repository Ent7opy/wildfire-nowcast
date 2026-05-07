/**
 * FIRMS client — pure parsing + token-bucket + config-missing tests.
 *
 * Never hits live FIRMS. Every test that exercises `fetchAreaCsv` passes a
 * stub `fetchImpl` that returns a canned response.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  _resetTokenBucket,
  fetchAreaCsv,
  parseFirmsCsv,
  type FirmsBbox,
} from "@/lib/firms/client";

const SONOMA_BBOX: FirmsBbox = [-125, 35, -120, 40];

async function loadFixture(): Promise<string> {
  const p = join(process.cwd(), "tests", "fixtures", "firms-sample.csv");
  return await readFile(p, "utf8");
}

describe("parseFirmsCsv", () => {
  it("parses the Stage 2 fixture into typed detections", async () => {
    const csv = await loadFixture();
    const result = parseFirmsCsv(csv, "VIIRS_NOAA20_NRT", SONOMA_BBOX, 1);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.detections).toHaveLength(5); // one row has blank lat/lon, skipped
    expect(result.detections[0]).toMatchObject({
      latitude: 38.445,
      longitude: -122.68,
      acqDate: "2026-04-21",
      acqTime: "0417",
      confidence: "n",
      daynight: "N",
      frp: 11.2,
    });
    expect(result.detections[1].confidence).toBe("h");
    expect(result.detections[3]).toMatchObject({
      latitude: 28.92,
      longitude: 47.93,
      frp: 85.4,
    });
    expect(result.emptyArea).toBe(false);
  });

  it("returns emptyArea=true on the no-data sentinel", () => {
    const result = parseFirmsCsv(
      "No fire data for the requested area or date range",
      "VIIRS_NOAA20_NRT",
      SONOMA_BBOX,
      1,
    );
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.emptyArea).toBe(true);
    expect(result.detections).toHaveLength(0);
  });

  it("returns emptyArea=true on a header-only response", () => {
    const header =
      "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight";
    const result = parseFirmsCsv(header, "VIIRS_NOAA20_NRT", SONOMA_BBOX, 1);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.emptyArea).toBe(true);
  });

  it("fails with parse_error on a CSV missing latitude column", () => {
    const bad = "lon,lat,time\n-120,38,0417";
    const result = parseFirmsCsv(bad, "VIIRS_NOAA20_NRT", SONOMA_BBOX, 1);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.code).toBe("parse_error");
  });

  it("maps MODIS brightness / bright_t31 columns into the canonical slots", () => {
    const modis =
      "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight\n" +
      "40.0,-120.0,330.0,1.0,1.0,2026-04-21,0500,Aqua,MODIS,80,6.3NRT,290.0,15.0,D";
    const result = parseFirmsCsv(modis, "MODIS_NRT", SONOMA_BBOX, 1);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.detections[0]).toMatchObject({
      brightTi4: 330,
      brightTi5: 290,
      confidence: "80",
    });
  });
});

describe("fetchAreaCsv", () => {
  beforeEach(() => {
    _resetTokenBucket();
    process.env.FIRMS_MAP_KEY = "test-key";
  });

  it("returns config_missing when FIRMS_MAP_KEY is unset", async () => {
    delete process.env.FIRMS_MAP_KEY;
    const r = await fetchAreaCsv({
      source: "VIIRS_NOAA20_NRT",
      bbox: SONOMA_BBOX,
    });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.code).toBe("config_missing");
  });

  it("validates dayRange bounds", async () => {
    const r = await fetchAreaCsv({
      source: "VIIRS_NOAA20_NRT",
      bbox: SONOMA_BBOX,
      dayRange: 11,
    });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.code).toBe("parse_error");
  });

  it("builds the correct URL and returns detections on 200", async () => {
    const csv = await loadFixture();
    let capturedUrl = "";
    const stubFetch = async (url: string): Promise<Response> => {
      capturedUrl = String(url);
      return new Response(csv, { status: 200 });
    };
    const r = await fetchAreaCsv({
      source: "VIIRS_NOAA20_NRT",
      bbox: SONOMA_BBOX,
      dayRange: 1,
      fetchImpl: stubFetch as unknown as typeof fetch,
    });
    expect(capturedUrl).toContain(
      "firms.modaps.eosdis.nasa.gov/api/area/csv/test-key/VIIRS_NOAA20_NRT/-125,35,-120,40/1",
    );
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.detections.length).toBeGreaterThan(0);
  });

  it("retries on 5xx and eventually returns upstream_error", async () => {
    let calls = 0;
    const stubFetch = async (): Promise<Response> => {
      calls += 1;
      return new Response("internal server error", { status: 502 });
    };
    const r = await fetchAreaCsv({
      source: "VIIRS_NOAA20_NRT",
      bbox: SONOMA_BBOX,
      fetchImpl: stubFetch as unknown as typeof fetch,
      sleepMs: async () => {},
    });
    expect(calls).toBe(3);
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.code).toBe("upstream_error");
  });

  it("surfaces rate_limited on 429 after 3 retries", async () => {
    const stubFetch = async (): Promise<Response> =>
      new Response("rate limited", { status: 429 });
    const r = await fetchAreaCsv({
      source: "VIIRS_NOAA20_NRT",
      bbox: SONOMA_BBOX,
      fetchImpl: stubFetch as unknown as typeof fetch,
      sleepMs: async () => {},
    });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.code).toBe("rate_limited");
  });

  it("returns network_error when fetch throws on every attempt", async () => {
    let calls = 0;
    const stubFetch = async (): Promise<Response> => {
      calls += 1;
      throw new Error("ECONNRESET");
    };
    const r = await fetchAreaCsv({
      source: "VIIRS_NOAA20_NRT",
      bbox: SONOMA_BBOX,
      fetchImpl: stubFetch as unknown as typeof fetch,
      sleepMs: async () => {},
    });
    expect(calls).toBe(3);
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.code).toBe("network_error");
    expect(r.message).toContain("ECONNRESET");
  });

  it("returns upstream_error immediately on 4xx without retrying", async () => {
    let calls = 0;
    const stubFetch = async (): Promise<Response> => {
      calls += 1;
      return new Response("invalid map key", { status: 403 });
    };
    const r = await fetchAreaCsv({
      source: "VIIRS_NOAA20_NRT",
      bbox: SONOMA_BBOX,
      fetchImpl: stubFetch as unknown as typeof fetch,
      sleepMs: async () => {},
    });
    expect(calls).toBe(1);
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.code).toBe("upstream_error");
    expect(r.status).toBe(403);
    expect(r.message).toContain("invalid map key");
  });

  it("recovers when a 5xx is followed by a 200", async () => {
    const csv = await loadFixture();
    let calls = 0;
    const stubFetch = async (): Promise<Response> => {
      calls += 1;
      if (calls === 1) return new Response("bad gateway", { status: 502 });
      return new Response(csv, { status: 200 });
    };
    const r = await fetchAreaCsv({
      source: "VIIRS_NOAA20_NRT",
      bbox: SONOMA_BBOX,
      fetchImpl: stubFetch as unknown as typeof fetch,
      sleepMs: async () => {},
    });
    expect(calls).toBe(2);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.detections.length).toBeGreaterThan(0);
  });

  it("redacts FIRMS_MAP_KEY from 4xx error messages", async () => {
    const key = "super-secret-map-key-abc123";
    process.env.FIRMS_MAP_KEY = key;
    const stubFetch = async (url: string): Promise<Response> =>
      new Response(`Forbidden for url ${String(url)}`, { status: 403 });
    const r = await fetchAreaCsv({
      source: "VIIRS_NOAA20_NRT",
      bbox: SONOMA_BBOX,
      fetchImpl: stubFetch as unknown as typeof fetch,
      sleepMs: async () => {},
    });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.message).not.toContain(key);
    expect(r.message).toContain("[REDACTED]");
  });

  it("throttles locally when the token bucket is exhausted", async () => {
    const stubFetch = async (): Promise<Response> =>
      new Response("", { status: 200 });
    // 6 tokens available initially; 7th call should be throttled locally.
    for (let i = 0; i < 6; i++) {
      const r = await fetchAreaCsv({
        source: "VIIRS_NOAA20_NRT",
        bbox: SONOMA_BBOX,
        fetchImpl: stubFetch as unknown as typeof fetch,
      });
      expect(r.ok).toBe(true);
    }
    const throttled = await fetchAreaCsv({
      source: "VIIRS_NOAA20_NRT",
      bbox: SONOMA_BBOX,
      fetchImpl: stubFetch as unknown as typeof fetch,
    });
    expect(throttled.ok).toBe(false);
    if (throttled.ok) return;
    expect(throttled.code).toBe("throttled_local");
  });
});
