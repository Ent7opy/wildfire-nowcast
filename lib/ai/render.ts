/**
 * Deterministic Markdown renderer for an `aoi_briefs.payload` Brief.
 *
 * Reads only the validated brief object — no LLM call here. Output shape:
 *
 *   # {AOI name} — situation brief
 *
 *   {summary}
 *
 *   ## Key facts
 *   - Nearest detection: 14.0 km @ 357° (N)
 *   - Wind: 240° @ 28 km/h (away from AOI)
 *   - Detections in 1 h window: 2 (max FRP 11.0 MW)
 *   - Satellites: VIIRS_NOAA20
 *
 *   ## Context
 *   {weather_note prose}
 *   Authority perimeter: PT-ICNF (posted 2026-04-21 04:00Z), no incursion.
 *   Prior events:
 *   - 2020-08-20 — LNU Lightning Complex eastern edge. Perimeter reached
 *     within 3 km of the preserve's north boundary; no incursion.
 *
 *   ## What to watch
 *   - …
 *
 *   _Uncertainty: …_
 *
 *   _Next brief: on polygon breach, else 06:00 local digest._
 *
 * The Stage 4 email channel will append snooze/pause/unsubscribe links — those
 * are notification-layer concerns and live outside this renderer.
 */
import type { Brief } from "./schema";

export function renderBriefMarkdown(brief: Brief): string {
  const lines: string[] = [];
  lines.push(`# ${brief.aoi.name} — situation brief`);
  lines.push("");
  lines.push(brief.summary);
  lines.push("");

  lines.push("## Key facts");
  const f = brief.key_facts;
  lines.push(
    `- Nearest detection: ${formatKm(f.nearest_detection_km)}${
      f.bearing_from_aoi_deg == null
        ? ""
        : ` @ ${formatBearing(f.bearing_from_aoi_deg)}`
    }`,
  );
  lines.push(`- Wind: ${formatWind(f)}`);
  lines.push(
    `- Detections in ${f.window_hours} h window: ${f.detection_count_in_window}` +
      (f.max_frp_mw == null ? "" : ` (max FRP ${f.max_frp_mw.toFixed(1)} MW)`),
  );
  lines.push(
    `- Satellites: ${f.satellites.length === 0 ? "unknown" : f.satellites.join(", ")}`,
  );
  lines.push("");

  lines.push("## Context");
  if (brief.context.weather_note) {
    lines.push(brief.context.weather_note);
    lines.push("");
  }
  lines.push(`Authority perimeter: ${formatPerimeter(brief.context.authority_perimeter)}`);
  if (brief.context.prior_events.length === 0) {
    lines.push("Prior events: no prior events on file.");
  } else {
    lines.push("Prior events:");
    for (const p of brief.context.prior_events) {
      const outcome = p.outcome ? ` ${p.outcome}` : "";
      lines.push(`- ${p.date} — ${p.description}${outcome}`);
    }
  }
  lines.push("");

  lines.push("## What to watch");
  if (brief.recommended_watch_items.length === 0) {
    lines.push("- (none)");
  } else {
    for (const item of brief.recommended_watch_items) {
      lines.push(`- ${item}`);
    }
  }
  lines.push("");

  lines.push(`_Uncertainty: ${brief.uncertainty}_`);
  lines.push("");
  lines.push(
    `_Next brief: ${brief.next_brief_hint.when} (trigger: ${brief.next_brief_hint.trigger})._`,
  );
  return lines.join("\n");
}

function formatKm(km: number): string {
  return `${km.toFixed(1)} km`;
}

function formatBearing(deg: number): string {
  const compass = bearingToCompass(deg);
  return `${Math.round(deg)}° (${compass})`;
}

function bearingToCompass(deg: number): string {
  const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  const idx = Math.round(((deg % 360) / 45)) % 8;
  return dirs[idx];
}

function formatWind(f: Brief["key_facts"]): string {
  if (f.wind_dir_deg == null || f.wind_speed_kmh == null) {
    return "unavailable";
  }
  const dir = `${Math.round(f.wind_dir_deg)}° @ ${Math.round(f.wind_speed_kmh)} km/h`;
  if (f.wind_toward_aoi == null) return dir;
  return `${dir} (${f.wind_toward_aoi ? "toward AOI" : "away from AOI"})`;
}

function formatPerimeter(p: Brief["context"]["authority_perimeter"]): string {
  if (p.source == null) return "none posted yet.";
  const posted = p.posted_ts ? ` (posted ${p.posted_ts})` : "";
  const contains =
    p.contains_detection == null
      ? ""
      : p.contains_detection
        ? ", contains detection"
        : ", does not contain detection";
  return `${p.source}${posted}${contains}.`;
}
