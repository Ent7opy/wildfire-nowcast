/**
 * Stage 6 — share-token mint / clear for an authenticated user's brief.
 *
 * POST  → idempotent mint. Returns existing token if still valid.
 * DELETE → revokes (NULLs token + expiry).
 */
import { NextResponse, type NextRequest } from "next/server";
import {
  clearBriefShareToken,
  setBriefShareToken,
} from "@/lib/db/aoi-repository";
import { apiError } from "@/lib/api/errors";
import { withDb } from "@/lib/api/handlers";
import { mintShareToken } from "@/lib/share/token";
import { publicShareUrl } from "@/lib/share/url";

type Ctx = { params: Promise<{ id: string }> };

export async function POST(
  _req: NextRequest,
  { params }: Ctx,
): Promise<NextResponse> {
  const { id } = await params;
  return withDb(async ({ db, userId }) => {
    const result = await setBriefShareToken(db, {
      userId,
      briefId: id,
      mintToken: mintShareToken,
    });
    if (!result) return apiError("not_found", `Brief ${id} not found`);
    return NextResponse.json({
      token: result.token,
      expiresAt: result.expiresAt.toISOString(),
      publicUrl: publicShareUrl(result.token),
    });
  });
}

export async function DELETE(
  _req: NextRequest,
  { params }: Ctx,
): Promise<NextResponse> {
  const { id } = await params;
  return withDb(async ({ db, userId }) => {
    const ok = await clearBriefShareToken(db, { userId, briefId: id });
    if (!ok) return apiError("not_found", `Brief ${id} not found`);
    return NextResponse.json({ ok: true });
  });
}
