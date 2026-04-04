import type { FireEvent } from "../types/api";
import type { IgnitionCell } from "../types/api";
import { haversineKm } from "./geo";

/**
 * Scan critical ignition cells in the viewport to find any that have no
 * confirmed fire within 50 km. Returns the first such cell, or null.
 *
 * Used to drive the priority-feed warning: "conditions critical but no
 * active fires detected nearby".
 */
export function firstCriticalCellWithoutNearbyFire(
  cells: IgnitionCell[],
  fires: FireEvent[],
  radiusKm = 50
): IgnitionCell | null {
  for (const cell of cells) {
    if (cell.level !== 'critical') continue;
    const hasNearbyFire = fires.some(
      (f) =>
        f.lat != null &&
        f.lon != null &&
        haversineKm(cell.lat, cell.lon, f.lat, f.lon) <= radiusKm
    );
    if (!hasNearbyFire) return cell;
  }
  return null;
}

/**
 * Count high + critical ignition cells within radiusKm of a point.
 *
 * Used to drive the fire-details context block: "N high-risk cells within
 * 50 km".
 */
export function highOrCriticalCellsNear(
  cells: IgnitionCell[],
  lat: number,
  lon: number,
  radiusKm = 50
): number {
  return cells.filter(
    (c) =>
      (c.level === 'high' || c.level === 'critical') &&
      haversineKm(lat, lon, c.lat, c.lon) <= radiusKm
  ).length;
}
