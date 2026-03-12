import type { FireEvent } from "../types/api";

function toFiniteNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function toTimestamp(value: unknown): number | null {
  if (typeof value !== "string" || value.trim().length === 0) {
    return null;
  }
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : null;
}

interface IntensitySortKey {
  tier: number;
  value: number;
}

function intensitySortKey(event: FireEvent): IntensitySortKey {
  const frpMax = toFiniteNumber(event.frp_max);
  if (frpMax !== null) return { tier: 4, value: frpMax };

  const frpMean = toFiniteNumber(event.frp_mean);
  if (frpMean !== null) return { tier: 3, value: frpMean };

  const brightnessMax = toFiniteNumber(event.brightness_max);
  if (brightnessMax !== null) return { tier: 2, value: brightnessMax };

  const brightnessMean = toFiniteNumber(event.brightness_mean);
  if (brightnessMean !== null) return { tier: 1, value: brightnessMean };

  return { tier: 0, value: 0 };
}

export function comparePriorityFeedEvents(a: FireEvent, b: FireEvent): number {
  const aKey = intensitySortKey(a);
  const bKey = intensitySortKey(b);

  const tierDiff = bKey.tier - aKey.tier;
  if (tierDiff !== 0) return tierDiff;

  const intensityDiff = bKey.value - aKey.value;
  if (intensityDiff !== 0) return intensityDiff;

  const scoreDiff = (toFiniteNumber(b.event_score) || 0) - (toFiniteNumber(a.event_score) || 0);
  if (scoreDiff !== 0) return scoreDiff;

  const detectionsDiff = (toFiniteNumber(b.detection_count) || 0) - (toFiniteNumber(a.detection_count) || 0);
  if (detectionsDiff !== 0) return detectionsDiff;

  const recencyDiff = (toTimestamp(b.end_time) || 0) - (toTimestamp(a.end_time) || 0);
  if (recencyDiff !== 0) return recencyDiff;

  return String(a.event_id || "").localeCompare(String(b.event_id || ""));
}
