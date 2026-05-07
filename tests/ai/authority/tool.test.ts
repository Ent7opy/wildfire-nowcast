/**
 * Stage 8 — `runAuthorityPerimeterTool` returns the snake_case tool shape on
 * hit and the all-null shape on miss; never throws. Future-proofs Path B.
 */
import { describe, expect, it, vi } from "vitest";
import { runAuthorityPerimeterTool } from "@/lib/ai/authority/tool";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const NIFC_BUCKET = "5x5:W125_N40";

describe("runAuthorityPerimeterTool", () => {
  it("returns snake_case fields on a hit", async () => {
    const detLat = 38.5;
    const detLon = -122.7;
    const ts = 1_775_000_000_000;
    const collection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          id: 7,
          geometry: {
            type: "Polygon",
            coordinates: [
              [
                [detLon - 0.05, detLat - 0.05],
                [detLon + 0.05, detLat - 0.05],
                [detLon + 0.05, detLat + 0.05],
                [detLon - 0.05, detLat + 0.05],
                [detLon - 0.05, detLat - 0.05],
              ],
            ],
          },
          properties: { poly_PolygonDateTime: ts },
        },
      ],
    };
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(collection));
    const r = await runAuthorityPerimeterTool({
      lat: detLat,
      lon: detLon,
      regionBucket: NIFC_BUCKET,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).toEqual({
      source: "NIFC WFIGS",
      posted_ts: new Date(ts).toISOString(),
      contains_detection: true,
    });
  });

  it("returns all-null on miss without throwing", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(new Response("nope", { status: 404 }));
    const r = await runAuthorityPerimeterTool({
      lat: 38.5,
      lon: -122.7,
      regionBucket: NIFC_BUCKET,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(r).toEqual({ source: null, posted_ts: null, contains_detection: null });
  });
});
