/**
 * GET /api/export/aois.geojson — all of the user's non-archived AOIs as a
 * GeoJSON FeatureCollection. Properties mirror the per-AOI export shape.
 */
import { NextResponse } from "next/server";
import { listAois, getRulesByAoiId } from "@/lib/db/aoi-repository";
import { withDb } from "@/lib/api/handlers";

export async function GET(): Promise<NextResponse> {
  return withDb(async ({ db, userId }) => {
    const aois = await listAois(db, userId);
    const features = await Promise.all(
      aois.map(async (a) => {
        const rules = await getRulesByAoiId(db, a.id);
        return {
          type: "Feature" as const,
          geometry: a.polygon,
          properties: {
            id: a.id,
            name: a.name,
            areaHa: a.areaHa,
            regionBucket: a.regionBucket,
            createdAt: a.createdAt.toISOString(),
            rules: rules
              ? {
                  distanceBufferKm: rules.distanceBufferKm,
                  minConfidence: rules.minConfidence,
                  minFrpMw: rules.minFrpMw,
                  quietHours: rules.quietHours,
                  pausedUntil: rules.pausedUntil?.toISOString() ?? null,
                  notifyChannels: rules.notifyChannels,
                }
              : null,
          },
        };
      }),
    );
    const body = JSON.stringify({
      type: "FeatureCollection",
      features,
    });
    return new NextResponse(body, {
      status: 200,
      headers: {
        "Content-Type": "application/geo+json",
        "Content-Disposition": `attachment; filename="aois.geojson"`,
      },
    });
  });
}
