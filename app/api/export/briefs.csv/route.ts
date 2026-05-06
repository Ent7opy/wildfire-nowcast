/**
 * GET /api/export/briefs.csv?since=YYYY-MM-DD — recent briefs as CSV.
 *
 * Default window: last 12 months. Streamed via a `ReadableStream` so large
 * users don't buffer thousands of rows; PGlite (test backend) gets the same
 * code path because the underlying `listAllBriefsWithPayloadForUser` already
 * paginates internally with a LIMIT.
 */
import { NextResponse, type NextRequest } from "next/server";
import { listAllBriefsWithPayloadForUser } from "@/lib/db/aoi-repository";
import { apiError } from "@/lib/api/errors";
import { withDb } from "@/lib/api/handlers";
import { csvRow } from "@/lib/export/csv";

const HEADER = [
  "brief_id",
  "aoi_id",
  "aoi_name",
  "created_at",
  "gate_reason",
  "model",
  "latency_ms",
  "cost_usd_est",
  "last_notified_at",
  "summary",
];

export async function GET(req: NextRequest): Promise<NextResponse> {
  const url = new URL(req.url);
  const sinceStr = url.searchParams.get("since");
  let since: Date | undefined;
  if (sinceStr) {
    const d = new Date(sinceStr);
    if (Number.isNaN(d.getTime())) {
      return apiError(
        "validation_failed",
        `Invalid since=${sinceStr}; expected YYYY-MM-DD`,
      );
    }
    since = d;
  }

  return withDb(async ({ db, userId }) => {
    const briefs = await listAllBriefsWithPayloadForUser(db, { userId, since });

    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(csvRow(HEADER) + "\n"));
        for (const b of briefs) {
          const summary =
            b.payload && typeof b.payload === "object" && "summary" in b.payload
              ? String((b.payload as { summary?: unknown }).summary ?? "")
              : "";
          controller.enqueue(
            encoder.encode(
              csvRow([
                b.id,
                b.aoiId,
                b.aoiName,
                b.createdAt.toISOString(),
                b.gateReason,
                b.model,
                b.latencyMs ?? "",
                b.costUsdEst ?? "",
                b.lastNotifiedAt?.toISOString() ?? "",
                summary,
              ]) + "\n",
            ),
          );
        }
        controller.close();
      },
    });

    return new NextResponse(stream, {
      status: 200,
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Content-Disposition": `attachment; filename="briefs.csv"`,
      },
    });
  });
}
