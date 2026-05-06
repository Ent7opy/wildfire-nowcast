/**
 * GET /api/notify/unsubscribe/[token] — public, idempotent.
 * Removes the email channel from the AOI's notify_channels. If the resulting
 * channel list is empty, the AOI is auto-paused so the user is not silently
 * polling-without-delivery.
 */
import type { NextRequest, NextResponse } from "next/server";
import { handleAction, type ActionRouteContext } from "../../_lib/handle";

export const runtime = "nodejs";

export async function GET(
  _req: NextRequest,
  ctx: ActionRouteContext,
): Promise<NextResponse> {
  return handleAction("unsubscribe", ctx);
}
