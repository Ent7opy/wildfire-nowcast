import { useCallback } from "react";
import { useAppStore } from "../state/store";

/**
 * Exposes a `requestLocation` function that triggers the browser Geolocation
 * API and writes the result into the safety slice of the store.
 */
export function useGeolocation(): { requestLocation: () => void } {
  const setSafetyLocation = useAppStore((s) => s.setSafetyLocation);
  const setSafetyLocationPermission = useAppStore((s) => s.setSafetyLocationPermission);

  const requestLocation = useCallback(() => {
    if (!navigator.geolocation) {
      setSafetyLocationPermission('denied');
      return;
    }
    setSafetyLocationPermission('requesting');
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setSafetyLocation({
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
          accuracyM: pos.coords.accuracy,
          acquiredAt: Date.now(),
        });
        // setSafetyLocation sets locationPermission to 'granted' automatically
      },
      (err) => {
        setSafetyLocation(null);
        // GeolocationPositionError.PERMISSION_DENIED === 1
        setSafetyLocationPermission(err.code === 1 ? 'denied' : 'unknown');
      },
      { enableHighAccuracy: false, timeout: 10_000, maximumAge: 60_000 }
    );
  }, [setSafetyLocation, setSafetyLocationPermission]);

  return { requestLocation };
}
