/**
 * /api/aoi/[id] — read, update, soft-delete one AOI.
 *
 * Next.js 16 dynamic-route params are async; we await `params` per the
 * App Router contract.
 */
import { NextResponse, type NextRequest } from "next/server";
import {
  archiveAoi,
  AoiNotFoundError,
  getAoiById,
  getRulesByAoiId,
  updateAoi,
} from "@/lib/db/aoi-repository";
import { aoiUpdateSchema } from "@/lib/validators/aoi";
import { apiError } from "@/lib/api/errors";
import { parseJson, withDb } from "@/lib/api/handlers";

type Ctx = { params: Promise<{ id: string }> };

export async function GET(
  _req: NextRequest,
  { params }: Ctx,
): Promise<NextResponse> {
  const { id } = await params;
  return withDb(async ({ db, userId }) => {
    const aoi = await getAoiById(db, userId, id);
    if (!aoi) return apiError("not_found", `AOI ${id} not found`);
    const rules = await getRulesByAoiId(db, id);
    return NextResponse.json({
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
    });
  });
}

export async function PATCH(
  req: NextRequest,
  { params }: Ctx,
): Promise<NextResponse> {
  const { id } = await params;
  const parsed = await parseJson(req, aoiUpdateSchema);
  if (!parsed.ok) return parsed.response;
  return withDb(async ({ db, userId }) => {
    const aoi = await updateAoi(db, {
      userId,
      aoiId: id,
      patch: parsed.value,
    });
    return NextResponse.json({
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
    });
  });
}

export async function DELETE(
  _req: NextRequest,
  { params }: Ctx,
): Promise<NextResponse> {
  const { id } = await params;
  return withDb(async ({ db, userId }) => {
    try {
      await archiveAoi(db, { userId, aoiId: id });
    } catch (err) {
      if (err instanceof AoiNotFoundError) {
        return apiError("not_found", err.message);
      }
      throw err;
    }
    return NextResponse.json({ ok: true, archivedId: id });
  });
}
