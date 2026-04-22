/**
 * /api/aoi/[id]/rules — upsert per-AOI monitoring rules.
 *
 * Spec: docs/SPEC-A-prime-v1.md §API surface (US-2).
 */
import { NextResponse, type NextRequest } from "next/server";
import { rulesUpsertSchema } from "@/lib/validators/aoi";
import { upsertRules } from "@/lib/db/aoi-repository";
import { parseJson, withDb } from "@/lib/api/handlers";

type Ctx = { params: Promise<{ id: string }> };

export async function PUT(
  req: NextRequest,
  { params }: Ctx,
): Promise<NextResponse> {
  const { id } = await params;
  const parsed = await parseJson(req, rulesUpsertSchema);
  if (!parsed.ok) return parsed.response;
  return withDb(async ({ db, userId }) => {
    const rules = await upsertRules(db, {
      userId,
      aoiId: id,
      rules: parsed.value,
    });
    return NextResponse.json({
      rules: {
        distanceBufferKm: rules.distanceBufferKm,
        minConfidence: rules.minConfidence,
        minFrpMw: rules.minFrpMw,
        quietHours: rules.quietHours,
        pausedUntil: rules.pausedUntil?.toISOString() ?? null,
        notifyChannels: rules.notifyChannels,
        updatedAt: rules.updatedAt.toISOString(),
      },
    });
  });
}
