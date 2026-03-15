import type { FireEvent } from "../types/api";

const EARTH_RADIUS_KM = 6371;

/** Haversine great-circle distance in kilometres. */
export function haversineKm(
  lat1: number, lon1: number,
  lat2: number, lon2: number
): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return EARTH_RADIUS_KM * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/** Returns the distance to the nearest fire in km, or null if no events have coordinates. */
export function nearestFireKm(
  userLat: number,
  userLon: number,
  events: FireEvent[]
): number | null {
  let min: number | null = null;
  for (const e of events) {
    if (e.lat == null || e.lon == null) continue;
    const d = haversineKm(userLat, userLon, e.lat, e.lon);
    if (min === null || d < min) min = d;
  }
  return min;
}

/** Filters events to those within radiusKm of a user location. */
export function eventsWithinRadius(
  userLat: number,
  userLon: number,
  radiusKm: number,
  events: FireEvent[]
): FireEvent[] {
  return events.filter(
    (e) =>
      e.lat != null &&
      e.lon != null &&
      haversineKm(userLat, userLon, e.lat, e.lon) <= radiusKm
  );
}
