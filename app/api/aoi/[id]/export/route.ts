/**
 * GET /api/aoi/[id]/export?format=geojson|markdown
 *
 * Per docs/SPEC-A-prime-v1.md US-6 — single-AOI portability.
 *
 * - geojson: a single GeoJSON Feature with `properties` carrying the rules.
 * - markdown: the AOI metadata header + every brief's rendered_markdown
 *   concatenated reverse-chron, separated by `---`. Footer carries the
 *   canonical positioning line and a link back to the dashboard.
 *
 * Hard cap of 500 briefs per the spec's US-6 acceptance #1.
 */
import { NextResponse, type NextRequest } from "next/server";
import {
  getAoiById,
  getRulesByAoiId,
  listBriefsForAoiWithPayload,
} from "@/lib/db/aoi-repository";
import { apiError } from "@/lib/api/errors";
import { withDb } from "@/lib/api/handlers";
import { slugify } from "@/lib/export/slug";
import { POSITIONING_LINE } from "@/lib/export/positioning";

type Ctx = { params: Promise<{ id: string }> };

const MAX_BRIEFS = 500;

export async function GET(req: NextRequest, { params }: Ctx): Promise<NextResponse> {
  const { id } = await params;
  const url = new URL(req.url);
  const format = url.searchParams.get("format") ?? "geojson";
  if (format !== "geojson" && format !== "markdown") {
    return apiError(
      "validation_failed",
      `Unsupported format "${format}". Expected geojson | markdown.`,
    );
  }
  return withDb(async ({ db, userId }) => {
    const aoi = await getAoiById(db, userId, id);
    if (!aoi) return apiError("not_found", `AOI ${id} not found`);
    const rules = await getRulesByAoiId(db, id);

    if (format === "geojson") {
      const feature = {
        type: "Feature" as const,
        geometry: aoi.polygon,
        properties: {
          id: aoi.id,
          name: aoi.name,
          areaHa: aoi.areaHa,
          regionBucket: aoi.regionBucket,
          createdAt: aoi.createdAt.toISOString(),
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
      const body = JSON.stringify(feature);
      return new NextResponse(body, {
        status: 200,
        headers: {
          "Content-Type": "application/geo+json",
          "Content-Disposition": `attachment; filename="${slugify(aoi.name)}.geojson"`,
        },
      });
    }

    const briefs = await listBriefsForAoiWithPayload(db, {
      userId,
      aoiId: id,
      limit: MAX_BRIEFS,
    });
    const body = renderAoiMarkdown(aoi, rules, briefs);
    return new NextResponse(body, {
      status: 200,
      headers: {
        "Content-Type": "text/markdown; charset=utf-8",
        "Content-Disposition": `attachment; filename="${slugify(aoi.name)}.md"`,
      },
    });
  });
}

type AoiSummary = {
  id: string;
  name: string;
  areaHa: number;
  regionBucket: string;
  createdAt: Date;
};

type RulesSummary = {
  distanceBufferKm: number;
  minConfidence: string;
  minFrpMw: number;
} | null;

function renderAoiMarkdown(
  aoi: AoiSummary,
  rules: RulesSummary,
  briefs: Array<{ createdAt: Date; renderedMarkdown: string }>,
): string {
  const lines: string[] = [];
  lines.push(`# ${aoi.name}`);
  lines.push("");
  lines.push(`- Area: ${aoi.areaHa.toFixed(1)} ha`);
  lines.push(`- Region bucket: ${aoi.regionBucket}`);
  lines.push(`- Created: ${aoi.createdAt.toISOString()}`);
  if (rules) {
    lines.push(`- Rules: ${rules.distanceBufferKm} km buffer, min FRP ${rules.minFrpMw} MW, confidence ≥ ${rules.minConfidence}`);
  }
  lines.push("");
  lines.push(`Briefs: ${briefs.length}`);
  lines.push("");

  for (const b of briefs) {
    lines.push("---");
    lines.push("");
    lines.push(`_Brief generated ${b.createdAt.toISOString()}_`);
    lines.push("");
    lines.push(b.renderedMarkdown);
    lines.push("");
  }

  lines.push("---");
  lines.push("");
  lines.push(`_${POSITIONING_LINE}_`);
  lines.push("");
  lines.push(`[Open in dashboard](/dashboard/aoi/${aoi.id})`);
  return lines.join("\n");
}
