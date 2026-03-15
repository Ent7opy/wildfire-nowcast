export interface SelectionView {
  latitude: number;
  longitude: number;
  zoom: number;
}

interface SelectionViewOptions {
  minZoom?: number;
  maxZoom?: number;
  minSpanDeg?: number;
  targetOccupancy?: number;
}

const DEFAULT_MIN_SPAN_DEG = 0.01;
const DEFAULT_TARGET_OCCUPANCY = 0.3;

export function selectionViewFromBounds(
  bounds: [number, number, number, number],
  options: SelectionViewOptions = {}
): SelectionView {
  const [minLon, minLat, maxLon, maxLat] = bounds;
  const minZoom = Number.isFinite(options.minZoom) ? Number(options.minZoom) : 6;
  const maxZoom = Number.isFinite(options.maxZoom) ? Number(options.maxZoom) : 14;
  const minSpanDeg = Number.isFinite(options.minSpanDeg) ? Number(options.minSpanDeg) : DEFAULT_MIN_SPAN_DEG;
  const occupancy = Number.isFinite(options.targetOccupancy) && Number(options.targetOccupancy) > 0 && Number(options.targetOccupancy) <= 1
    ? Number(options.targetOccupancy)
    : DEFAULT_TARGET_OCCUPANCY;

  const longitude = (minLon + maxLon) / 2;
  const latitude = (minLat + maxLat) / 2;
  const lonSpan = Math.max(maxLon - minLon, 0);
  const latSpan = Math.max(maxLat - minLat, 0);
  const span = Math.max(lonSpan, latSpan, minSpanDeg);
  const viewportSpan = span / occupancy;
  const zoom = Math.max(minZoom, Math.min(maxZoom, Math.log2(360 / viewportSpan)));

  return { latitude, longitude, zoom };
}
