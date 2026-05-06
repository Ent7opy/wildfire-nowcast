/**
 * /api/aoi — list + create AOIs.
 *
 * Spec: docs/SPEC-A-prime-v1.md §API surface.
 * Auth: single-user stub until Stage 5 (Clerk).
 * Runtime: Node.js (default) — pg driver is not Edge-compatible.
 */
import { NextResponse, type NextRequest } from "next/server";
import { aoiCreateSchema } from "@/lib/validators/aoi";
import { createAoi, listAois } from "@/lib/db/aoi-repository";
import { parseJson, withDb } from "@/lib/api/handlers";

export async function GET(): Promise<NextResponse> {
  return withDb(async ({ db, userId }) => {
    const rows = await listAois(db, userId);
    return NextResponse.json({
      aois: rows.map((r) => ({
        id: r.id,
        name: r.name,
        regionBucket: r.regionBucket,
        areaHa: r.areaHa,
        createdAt: r.createdAt.toISOString(),
      })),
    });
  });
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  const parsed = await parseJson(req, aoiCreateSchema);
  if (!parsed.ok) return parsed.response;
  return withDb(async ({ db, userId }) => {
    const { aoi, rules } = await createAoi(db, {
      userId,
      name: parsed.value.name,
      geometry: parsed.value.geometry,
    });
    return NextResponse.json(
      {
        aoi: {
          id: aoi.id,
          name: aoi.name,
          regionBucket: aoi.regionBucket,
          areaHa: aoi.areaHa,
          createdAt: aoi.createdAt.toISOString(),
          polygon: aoi.polygon,
          bbox: aoi.bbox,
          centroid: aoi.centroid,
        },
        rules: {
          distanceBufferKm: rules.distanceBufferKm,
          minConfidence: rules.minConfidence,
          minFrpMw: rules.minFrpMw,
          quietHours: rules.quietHours,
          pausedUntil: rules.pausedUntil?.toISOString() ?? null,
          notifyChannels: rules.notifyChannels,
        },
      },
      { status: 201 },
    );
  });
}
