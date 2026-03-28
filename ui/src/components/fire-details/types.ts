import type { FireEvent, ReverseGeocodeResponse } from "../../types/api";
import { toFiniteNumber as safeNumber } from "../../utils/priorityFeed";
import { HIGH_CONFIDENCE_THRESHOLD } from "../map/layers/layerConfig";

export { safeNumber, HIGH_CONFIDENCE_THRESHOLD };

export interface GdeltArticle {
  title: string;
  url: string;
  socialimage?: string;
  seendate: string;
  sourcecountry?: string;
}

export interface IntensityDescriptor {
  label: string;
  value: number;
  unit: "MW" | "K";
}

export const STRONG_WILDFIRE_TERMS = [
  "wildfire", "wildfires", "bushfire", "bushfires", "forest fire", "forest fires",
  "brush fire", "brush fires", "grass fire", "grass fires",
  "fire evacuation", "fire evacuations", "fire season", "acres burned",
  "fire containment", "fire crews", "firefighter", "firefighters",
  "prescribed burn", "prescribed fire", "controlled burn",
  "fire weather", "red flag warning", "structure fire", "fire behavior",
  "fire perimeter", "fire spread", "fire retardant", "air tanker"
];

export const FIRE_CONTEXT_TERMS = [
  "evacuate", "evacuation", "blaze", "flames", "contained", "containment",
  "smoke", "acres", "crews", "perimeter", "hotspot", "embers",
  "arson", "drought", "fire line", "backfire", "torching", "spotting"
];

export const EXCLUDE_TERMS = [
  "gunfire", "ceasefire", "cease-fire", "opens fire", "open fire",
  "fired on", "fired at", "under fire", "crossfire", "hail of fire",
  "fire sale", "fired from", "firing squad", "return fire", "friendly fire",
  "rapid fire", "spitfire", "fire someone", "fired over", "drew fire",
  "facing fire", "political fire", "israel", "gaza", "ukraine", "russia",
  "shooting", "gunman", "military", "soldier", "missile", "bomb"
];

export function isWildfireArticle(title: string): boolean {
  const lower = title.toLowerCase();
  if (EXCLUDE_TERMS.some((term) => lower.includes(term))) return false;
  if (STRONG_WILDFIRE_TERMS.some((term) => lower.includes(term))) return true;
  if (lower.includes("fire") && FIRE_CONTEXT_TERMS.some((term) => lower.includes(term))) return true;
  return false;
}


export function severity(event: FireEvent): number {
  const score = safeNumber(event.event_score);
  if (score === null) return 0;
  return Math.max(0, Math.min(score, 1));
}

export function coordinateKey(lat: number | null, lon: number | null): string | null {
  if (lat === null || lon === null) {
    return null;
  }
  return `${lat.toFixed(4)},${lon.toFixed(4)}`;
}

export function hasDirectLocation(event: FireEvent): boolean {
  const candidates = [event.location_name, event.region_name, event.admin1_name, event.admin0_name, event.country];
  return candidates.some((candidate) => typeof candidate === "string" && candidate.trim().length > 0);
}

export function locationLabel(event: FireEvent, resolved?: ReverseGeocodeResponse | null): string {
  const candidates = [
    resolved?.location_name,
    event.location_name,
    event.region_name,
    resolved?.admin1_name,
    event.admin1_name,
    event.admin0_name,
    resolved?.country,
    event.country,
    resolved?.display_name
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim();
    }
  }
  return "Unresolved location";
}

export function confidenceLabel(event: FireEvent): "High" | "Nominal" {
  return severity(event) >= HIGH_CONFIDENCE_THRESHOLD ? "High" : "Nominal";
}

export function formattedTime(value: unknown): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    return "n/a";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "n/a";
  }
  return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function primaryIntensity(event: FireEvent): IntensityDescriptor | null {
  const frpMax = safeNumber(event.frp_max);
  if (frpMax !== null) {
    return { label: "Peak FRP", value: frpMax, unit: "MW" };
  }

  const frpMean = safeNumber(event.frp_mean);
  if (frpMean !== null) {
    return { label: "Mean FRP", value: frpMean, unit: "MW" };
  }

  const brightnessMax = safeNumber(event.brightness_max);
  if (brightnessMax !== null) {
    return { label: "Peak Brightness", value: brightnessMax, unit: "K" };
  }

  const brightnessMean = safeNumber(event.brightness_mean);
  if (brightnessMean !== null) {
    return { label: "Mean Brightness", value: brightnessMean, unit: "K" };
  }

  return null;
}

export function formatIntensity(value: number, unit: "MW" | "K"): string {
  if (!Number.isFinite(value)) {
    return `n/a ${unit}`;
  }
  if (unit === "MW") {
    return `${value.toFixed(2)} ${unit}`;
  }
  return `${value.toFixed(1)} ${unit}`;
}

export function frpHumanLabel(frpMw: number): string {
  if (frpMw >= 500) return "Extreme Intensity / Rapid Spread";
  if (frpMw >= 100) return "Intense Fire Activity";
  if (frpMw >= 10)  return "Moderate Activity";
  return "Smoldering / Low Intensity";
}

export function riskTierFromScore(score: number): { label: string; color: string } {
  if (score >= 0.75) return { label: "Critical", color: "#ef4444" };
  if (score >= 0.5)  return { label: "High",     color: "#f97316" };
  if (score >= 0.25) return { label: "Moderate", color: "#eab308" };
  return { label: "Low", color: "#22c55e" };
}

export function observationSummary(event: FireEvent): string {
  if (event.review_required) {
    return "This event is flagged for analyst review. Treat the perimeter and intensity as provisional until verified.";
  }
  const time = typeof event.start_time === "string" && event.start_time.trim().length > 0
    ? ` at ${formattedTime(event.start_time)}`
    : "";
  const provenance = String(event.geom_source || "").toLowerCase() === "authoritative"
    ? "Authoritative perimeter from official source."
    : "Perimeter is estimated from detection cluster.";
  const fronts = Number(event.front_count || 0);
  const frontStr = fronts === 1 ? "1 active front tracked." : fronts > 1 ? `${fronts} active fronts tracked.` : "No fronts tracked yet.";
  return `Satellite thermal anomaly detected${time}. ${provenance} ${frontStr}`;
}

export function satelliteLabel(source?: string | null, sensor?: string | null): string {
  const s = `${source || ""} ${sensor || ""}`.toUpperCase();
  if (s.includes("VIIRS") && (s.includes("NOAA20") || s.includes("NOAA-20"))) return "VIIRS · NOAA-20";
  if (s.includes("VIIRS") && (s.includes("SNPP") || s.includes("NPP"))) return "VIIRS · Suomi-NPP";
  if (s.includes("MODIS") && s.includes("TERRA")) return "MODIS · Terra";
  if (s.includes("MODIS") && s.includes("AQUA")) return "MODIS · Aqua";
  if (s.includes("CLUSTER") || s.includes("AGGREGATED")) return "Multi-sensor cluster";
  return [source, sensor].filter(Boolean).join(" · ") || "Unknown sensor";
}

export function geometryProvenanceLabel(event: FireEvent): "Authoritative perimeter" | "Estimated perimeter" {
  return String(event.geom_source || "").toLowerCase() === "authoritative"
    ? "Authoritative perimeter"
    : "Estimated perimeter";
}

export function formatSeenDate(seendate: string): string {
  return new Date(
    seendate.replace(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z/, "$1-$2-$3T$4:$5:$6Z")
  ).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
