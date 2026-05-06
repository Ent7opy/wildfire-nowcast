/**
 * Prompt builder for Stage 3 brief generation.
 *
 * Inputs are the structured `BriefContext` the orchestrator gathers from the
 * DB; outputs are the system + user prompt strings fed to `generateObject`.
 *
 * Two design rules:
 *   1. Determinism in the *structured* fields. The summary prose can vary;
 *      the schema fields must derive from the inputs we pass in.
 *   2. No invention. The prompt explicitly tells the model not to fabricate
 *      authority perimeters or weather data when the inputs are null.
 */
import { SCHEMA_VERSION } from "./schema";

export type BriefContext = {
  aoi: {
    id: string;
    name: string;
    areaHa: number;
  };
  event: {
    nearestDistanceKm: number;
    /**
     * Great-circle bearing from AOI centroid to the nearest detection, deg
     * (0=N, 90=E, …). Null when no detection lat/lon is available to derive
     * it — we never inject a hardcoded value (AGENTS.md no-fabrication).
     */
    bearingFromAoiDeg: number | null;
    detectionCount: number;
    peakFrpMw: number | null;
    windowHours: number;
    satellites: string[];
    firstSeenAt: string;
    lastSeenAt: string;
  };
  /**
   * v1 leaves authority_perimeter null and weather null — Stage 3 ships
   * without authority/weather ingests (per SPEC §Open questions). The
   * orchestrator passes nulls; the prompt instructs the model not to
   * fabricate values.
   */
  weather: { note: string | null } | null;
  authorityPerimeter: {
    source: string | null;
    postedTs: string | null;
    containsDetection: boolean | null;
  } | null;
  priorEvents: Array<{
    date: string;
    description: string;
    outcome: string | null;
  }>;
};

export const SYSTEM_PROMPT = [
  "You are the Wildfire Nowcast situation-brief writer.",
  "You produce a single L2-style brief for one Area of Interest (AOI), in valid JSON conforming to the provided schema.",
  "",
  "Rules:",
  "- Do NOT invent values. If an input is null (e.g. weather, authority_perimeter), reflect that faithfully.",
  "- The summary is 1–2 sentences, like a staffer's radio report.",
  "- The uncertainty field is mandatory; be explicit about what is NOT known.",
  "- Recommended watch items are concrete observations the steward can make, not imperatives.",
  "- Numeric fields (distances, bearings, FRP, counts) MUST equal the values supplied in the user message; do not round or transform them.",
  `- The schema_version field must be exactly ${SCHEMA_VERSION}.`,
].join("\n");

export function buildUserPrompt(ctx: BriefContext): string {
  const event = ctx.event;
  const weather = ctx.weather?.note ?? null;
  const perim = ctx.authorityPerimeter ?? {
    source: null,
    postedTs: null,
    containsDetection: null,
  };
  const lines: string[] = [];
  lines.push(`AOI: ${ctx.aoi.name} (id=${ctx.aoi.id}, area_ha=${ctx.aoi.areaHa})`);
  lines.push("");
  lines.push("Event:");
  lines.push(`  nearest_detection_km: ${event.nearestDistanceKm}`);
  lines.push(
    `  bearing_from_aoi_deg: ${event.bearingFromAoiDeg ?? "null"}`,
  );
  lines.push(`  detection_count_in_window: ${event.detectionCount}`);
  lines.push(`  max_frp_mw: ${event.peakFrpMw ?? "null"}`);
  lines.push(`  window_hours: ${event.windowHours}`);
  lines.push(`  satellites: [${event.satellites.join(", ")}]`);
  lines.push(`  first_seen_at: ${event.firstSeenAt}`);
  lines.push(`  last_seen_at: ${event.lastSeenAt}`);
  lines.push("");
  lines.push(`Weather note (null if no data): ${weather ?? "null"}`);
  lines.push("");
  lines.push("Authority perimeter:");
  lines.push(`  source: ${perim.source ?? "null"}`);
  lines.push(`  posted_ts: ${perim.postedTs ?? "null"}`);
  lines.push(`  contains_detection: ${formatNullableBool(perim.containsDetection)}`);
  lines.push("");
  if (ctx.priorEvents.length === 0) {
    lines.push("Prior events on file: none.");
  } else {
    lines.push("Prior events on file:");
    for (const p of ctx.priorEvents) {
      lines.push(`  - ${p.date}: ${p.description}${p.outcome ? ` — ${p.outcome}` : ""}`);
    }
  }
  lines.push("");
  lines.push(
    "Produce the JSON brief. wind_dir_deg / wind_speed_kmh / wind_toward_aoi must be null unless wind data is provided above (currently they are not). bearing_from_aoi_deg must be null if the value above is null — do not invent a direction.",
  );
  return lines.join("\n");
}

function formatNullableBool(b: boolean | null): string {
  if (b === null) return "null";
  return b ? "true" : "false";
}
