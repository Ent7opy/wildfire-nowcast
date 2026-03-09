import type { BBox } from "../types/api";
import type { MapViewState } from "../types/state";

export function viewportBbox(view: MapViewState): BBox {
  const degPerTile = 360.0 / 2 ** view.zoom;
  const half = degPerTile * 0.5;
  return [
    Math.max(view.longitude - half, -180),
    Math.max(view.latitude - half, -85),
    Math.min(view.longitude + half, 180),
    Math.min(view.latitude + half, 85)
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
