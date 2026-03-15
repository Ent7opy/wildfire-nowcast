import type { FiltersState } from "../types/state";

export interface FilterPreset {
  name: string;
  hoursStart: number;
  hoursEnd: number;
  likelihood: number;
}

export const FILTER_PRESETS: FilterPreset[] = [
  { name: "Last Hour High", hoursStart: 1, hoursEnd: 0, likelihood: 0.6 },
  { name: "Last 6h All", hoursStart: 6, hoursEnd: 0, likelihood: 0.0 },
  { name: "Last 6h Medium+", hoursStart: 6, hoursEnd: 0, likelihood: 0.33 },
  { name: "Last 24h High", hoursStart: 24, hoursEnd: 0, likelihood: 0.6 },
  { name: "Last 24h All", hoursStart: 24, hoursEnd: 0, likelihood: 0.0 }
];

export function matchingPreset(filters: FiltersState): string | null {
  const found = FILTER_PRESETS.find(
    (preset) =>
      preset.hoursStart === filters.hoursStart &&
      preset.hoursEnd === filters.hoursEnd &&
      Math.abs(preset.likelihood - filters.minLikelihood) < 0.01
  );
  return found ? found.name : null;
}

export function parseBoolFlag(raw: string | null): boolean | null {
  if (raw === null) {
    return null;
  }
  const normalized = raw.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return null;
}
