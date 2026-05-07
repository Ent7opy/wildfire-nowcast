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

  it("returns null when the FeatureCollection has an empty features array", async () => {
    // Reviewer flagged: pin that an empty upstream result returns null (not throw,
    // not a fabricated record). This is the build-without-blocking path.
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(jsonResponse({ type: "FeatureCollection", features: [] }));
    const r = await fetchAuthorityPerimeter({
      lat: 38.5,
      lon: -122.7,
      regionBucket: NIFC_BUCKET,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).toBeNull();
    expect(fetchImpl).toHaveBeenCalledOnce();
  });

  it("picks the most recent perimeter when several features are in radius", async () => {
    // Reviewer flagged: pin the timestamp tie-break independent of distance.
    // All three features share the same centroid (and are within radius); only
    // the timestamp differs. The newest must win.
    const detLat = 38.5;
    const detLon = -122.7;
    const t1 = 1_770_000_000_000;
    const t2 = 1_775_000_000_000;
    const t3 = 1_772_000_000_000;
    const collection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          id: "older",
          geometry: { type: "Polygon", coordinates: squarePolygon(detLon, detLat) },
          properties: { poly_PolygonDateTime: t1 },
        },
        {
          type: "Feature",
          id: "newest",
          geometry: { type: "Polygon", coordinates: squarePolygon(detLon, detLat) },
          properties: { poly_PolygonDateTime: t2 },
        },
        {
          type: "Feature",
          id: "middle",
          geometry: { type: "Polygon", coordinates: squarePolygon(detLon, detLat) },
          properties: { poly_PolygonDateTime: t3 },
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
    expect(r!.rawFeatureId).toBe("newest");
    expect(r!.postedTs).toBe(new Date(t2).toISOString());
  });

  it("supports MultiPolygon geometry", async () => {
    // Code branches on geometry.type; pin that MultiPolygon is iterated and PIP
    // is evaluated against any constituent polygon.
    const detLat = 38.5;
    const detLon = -122.7;
    const ts = 1_775_000_000_000;
    const collection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          id: 42,
          geometry: {
            type: "MultiPolygon",
            coordinates: [
              // First polygon: ~500 km offset (centroid out of radius if alone).
              squarePolygon(detLon + 5, detLat + 5),
              // Second polygon: contains the detection.
              squarePolygon(detLon, detLat),
            ],
          },
          properties: { poly_PolygonDateTime: ts },
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
    // The mean-of-vertices centroid sits between the two polygons; with two
    // 0.05° squares centered ~5° apart it's ~2.5° from the detection — outside
    // the 25 km default radius. Bump radius to confirm MultiPolygon iteration
    // and PIP both work end-to-end.
    expect(r).toBeNull();

    const fetchImpl2 = vi.fn().mockResolvedValue(jsonResponse(collection));
    const r2 = await fetchAuthorityPerimeter({
      lat: detLat,
      lon: detLon,
      regionBucket: NIFC_BUCKET,
      radiusKm: 1000,
      fetchImpl: fetchImpl2 as unknown as typeof fetch,
    });
    expect(r2).not.toBeNull();
    expect(r2!.containsDetection).toBe(true);
    expect(r2!.rawFeatureId).toBe("42");
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
