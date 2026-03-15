import { ScatterplotLayer } from "@deck.gl/layers";
import type { UserLocationState } from "../types/state";

/**
 * Builds Deck.GL layers to display the user's location:
 * - A pulsing blue dot at the user position
 * - A larger, semi-transparent ring representing the proximity radius
 */
export function buildUserLocationLayers(
  userLocation: UserLocationState,
  _proximityRadiusKm: number
): ScatterplotLayer[] {
  const position: [number, number] = [userLocation.lon, userLocation.lat];

  // Inner dot
  const dot = new ScatterplotLayer({
    id: "user-location-dot",
    data: [{ position }],
    getPosition: (d: { position: [number, number] }) => d.position,
    getRadius: 8,
    radiusUnits: "pixels",
    getFillColor: [59, 130, 246, 255],
    getLineColor: [255, 255, 255, 200],
    lineWidthMinPixels: 2,
    stroked: true,
  });

  // Outer pulse ring
  const ring = new ScatterplotLayer({
    id: "user-location-ring",
    data: [{ position }],
    getPosition: (d: { position: [number, number] }) => d.position,
    getRadius: 18,
    radiusUnits: "pixels",
    getFillColor: [59, 130, 246, 40],
    getLineColor: [59, 130, 246, 140],
    lineWidthMinPixels: 1.5,
    stroked: true,
  });

  return [ring, dot];
}
