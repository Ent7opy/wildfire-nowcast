import type { BBox } from "../types/api";
import type { MapViewState } from "../types/state";

export function viewportBbox(view: MapViewState, viewportWidth = 1440, viewportHeight = 900): BBox {
  const degPerTile = 360.0 / 2 ** view.zoom;
  const tilesX = viewportWidth / 256;
  const tilesY = viewportHeight / 256;
  const halfLon = (degPerTile * tilesX) / 2;
  const halfLat = (degPerTile * tilesY) / 2;
  return [
    Math.max(view.longitude - halfLon, -180),
    Math.min(view.latitude - halfLat, -85),
    Math.min(view.longitude + halfLon, 180),
    Math.min(view.latitude + halfLat, 85)
  ];
}

export function eventLimitForZoom(zoom: number): number {
  if (zoom >= 4) {
    return 10000;
  }
  if (zoom >= 2) {
    return 4000;
  }
  return 2000;
}

export function frontLimitForZoom(zoom: number): number {
  return zoom >= 7 ? 1000 : 600;
}

export function shouldLoadFronts(zoom: number): boolean {
  return zoom >= 5;
}

export function shouldRenderCentroids(zoom: number): boolean {
  return zoom < 4;
}
