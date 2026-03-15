import { useEffect } from "react";
import type { FireEvent } from "../types/api";
import { useAppStore } from "../state/store";
import { nearestFireKm } from "../utils/geo";

/**
 * Keeps `nearestFireDistanceKm` and `safetyTier` in the store in sync with
 * the current visible events and user location whenever safety mode is active.
 *
 * Call from App.tsx after visible events are computed:
 *   useSafetyMetrics(visibleEvents);
 */
export function useSafetyMetrics(visibleEvents: FireEvent[]): void {
  const enabled = useAppStore((s) => s.safety.enabled);
  const userLocation = useAppStore((s) => s.safety.userLocation);
  const updateSafetyMetrics = useAppStore((s) => s.updateSafetyMetrics);

  useEffect(() => {
    if (!enabled || !userLocation) {
      updateSafetyMetrics(null);
      return;
    }
    const nearest = nearestFireKm(userLocation.lat, userLocation.lon, visibleEvents);
    updateSafetyMetrics(nearest);
  }, [enabled, userLocation, visibleEvents, updateSafetyMetrics]);
}
