/**
 * Stage 8 — `fetchAuthorityPerimeter` unit tests with stubbed HTTP.
 */
import { describe, expect, it, vi } from "vitest";
import { fetchAuthorityPerimeter } from "@/lib/ai/authority/fetch";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function squarePolygon(centerLon: number, centerLat: number, halfDeg = 0.05): number[][][] {
  // Roughly 5–10 km per side at mid-latitudes; small enough to keep test fixtures sane.
  return [
    [
      [centerLon - halfDeg, centerLat - halfDeg],
      [centerLon + halfDeg, centerLat - halfDeg],
      [centerLon + halfDeg, centerLat + halfDeg],
      [centerLon - halfDeg, centerLat + halfDeg],
      [centerLon - halfDeg, centerLat - halfDeg],
    ],
  ];
}

const NIFC_BUCKET = "5x5:W125_N40"; // covered by NIFC

describe("fetchAuthorityPerimeter", () => {
  it("returns null when no source covers the bucket without making a network call", async () => {
    const fetchImpl = vi.fn();
    const r = await fetchAuthorityPerimeter({
      lat: -33.9,
      lon: 151.2,
      regionBucket: "5x5:E150_S35", // Sydney — no source
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).toBeNull();
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("returns the in-radius feature with most recent timestamp", async () => {
    const detLat = 38.5;
    const detLon = -122.7;
    const olderTs = 1_770_000_000_000; // older
    const newerTs = 1_775_000_000_000; // newer
    const farTs = 1_780_000_000_000;   // newest, but outside radius

    const collection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          id: 1,
          geometry: { type: "Polygon", coordinates: squarePolygon(detLon + 0.1, detLat + 0.05) },
          properties: { poly_PolygonDateTime: olderTs },
        },
        {
          type: "Feature",
          id: 2,
          geometry: { type: "Polygon", coordinates: squarePolygon(detLon, detLat) },
          properties: { poly_PolygonDateTime: newerTs },
        },
        {
          type: "Feature",
          id: 3,
          // ~500 km away — outside default 25 km radius
          geometry: { type: "Polygon", coordinates: squarePolygon(detLon + 5, detLat + 5) },
          properties: { poly_PolygonDateTime: farTs },
        },
      ],
    };

    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(collection));
    const r = await fetchAuthorityPerimeter({
      lat: detLat,
      lon: detLon,
      regionBucket: NIFC_BUCKET,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).not.toBeNull();
    expect(r!.source).toBe("NIFC WFIGS");
    expect(r!.postedTs).toBe(new Date(newerTs).toISOString());
    expect(r!.containsDetection).toBe(true);
    expect(r!.rawFeatureId).toBe("2");
  });

  it("returns containsDetection=false when feature is in radius but does not contain the point", async () => {
    const detLat = 38.5;
    const detLon = -122.7;
    const collection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          id: 1,
          // Centroid ~12 km away (within 25 km), but the polygon is small and offset.
          geometry: { type: "Polygon", coordinates: squarePolygon(detLon + 0.15, detLat) },
          properties: { poly_PolygonDateTime: 1_775_000_000_000 },
        },
      ],
    };
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(collection));
    const r = await fetchAuthorityPerimeter({
      lat: detLat,
      lon: detLon,
      regionBucket: NIFC_BUCKET,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).not.toBeNull();
    expect(r!.containsDetection).toBe(false);
  });

  it("returns null on 404", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(new Response("nope", { status: 404 }));
    const r = await fetchAuthorityPerimeter({
      lat: 38.5,
      lon: -122.7,
      regionBucket: NIFC_BUCKET,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).toBeNull();
  });

  it("returns null on rate limit (429)", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(new Response("slow down", { status: 429 }));
    const r = await fetchAuthorityPerimeter({
      lat: 38.5,
      lon: -122.7,
      regionBucket: NIFC_BUCKET,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).toBeNull();
  });

  it("returns null on network rejection", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError("ECONNRESET"));
    const r = await fetchAuthorityPerimeter({
      lat: 38.5,
      lon: -122.7,
      regionBucket: NIFC_BUCKET,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).toBeNull();
  });

  it("returns null on malformed JSON body", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      new Response("not-json{", { status: 200, headers: { "content-type": "application/json" } }),
    );
    const r = await fetchAuthorityPerimeter({
      lat: 38.5,
      lon: -122.7,
      regionBucket: NIFC_BUCKET,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).toBeNull();
  });

  it("returns null when no features are in radius", async () => {
    const collection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          id: 1,
          geometry: { type: "Polygon", coordinates: squarePolygon(-100, 30) },
          properties: { poly_PolygonDateTime: 1_775_000_000_000 },
        },
      ],
    };
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(collection));
    const r = await fetchAuthorityPerimeter({
      lat: 38.5,
      lon: -122.7,
      regionBucket: NIFC_BUCKET,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).toBeNull();
  });
});
