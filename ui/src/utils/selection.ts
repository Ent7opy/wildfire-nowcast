import type { FireEvent } from "../types/api";

export function normalizePickedEvent(object: unknown): FireEvent | null {
  if (!object || typeof object !== "object") {
    return null;
  }

  const maybeFeature = object as { properties?: unknown };
  const source = maybeFeature.properties && typeof maybeFeature.properties === "object" ? maybeFeature.properties : object;
  if (!source || typeof source !== "object") {
    return null;
  }

  const candidate = source as FireEvent;
  const lat = typeof candidate.lat === "number" ? candidate.lat : Number(candidate.lat);
  const lon = typeof candidate.lon === "number" ? candidate.lon : Number(candidate.lon);

  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return null;
  }

  return {
    ...candidate,
    lat,
    lon
  };
}
