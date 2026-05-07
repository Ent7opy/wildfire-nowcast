/**
 * GET /api/notify/feedback/[token]?v=yes|no — public, idempotent.
 * Records (or flips) brief feedback. Re-clicking with the opposite value
 * updates the row instead of inserting a duplicate (unique index on
 * (brief_id, recipient_token)).
 */
import { NextResponse, type NextRequest } from "next/server";
import { handleAction, type ActionRouteContext } from "../../_lib/handle";

export const runtime = "nodejs";

export async function GET(
  req: NextRequest,
  ctx: ActionRouteContext,
): Promise<NextResponse> {
  const v = new URL(req.url).searchParams.get("v");
  const feedbackValue = v === "yes" ? "yes" : v === "no" ? "no" : undefined;
  return handleAction("feedback", ctx, { feedbackValue });
}
