/**
 * GET /api/notify/pause/[token] — public, idempotent.
 * Token is the bearer secret. Pauses the AOI indefinitely.
 */
import type { NextRequest, NextResponse } from "next/server";
import { handleAction, type ActionRouteContext } from "../../_lib/handle";

export const runtime = "nodejs";

export async function GET(
  _req: NextRequest,
  ctx: ActionRouteContext,
): Promise<NextResponse> {
  return handleAction("pause", ctx);
}
