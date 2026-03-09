import type { FireEvent, FireFront } from "../types/api";
import type { Feature, Geometry } from "geojson";

export const FIRE_THRESHOLDS = {
  veryHigh: 0.8,
  high: 0.6,
  medium: 0.4,
  low: 0.2
} as const;

export const RISK_THRESHOLDS = {
  medium: 0.3,
  high: 0.6
} as const;

export const FORECAST_DEFAULT_HORIZONS = [24];
export const FORECAST_DEFAULT_THRESHOLDS = [0.7];

export const FIRE_COLORS = {
  veryHighFill: [220, 38, 38, 240],
  highFill: [239, 68, 68, 230],
  mediumFill: [255, 107, 53, 220],
  lowFill: [251, 191, 36, 200],
  veryLowFill: [253, 224, 71, 180],
  unscoredFill: [128, 128, 128, 150],
  outlineHigh: [255, 107, 53, 200],
  outlineDefault: [255, 255, 255, 100]
} as const;

export interface RenderEvent extends FireEvent {
  lat: number;
  lon: number;
  fill_r: number;
  fill_g: number;
  fill_b: number;
  fill_a: number;
  line_r: number;
  line_g: number;
  line_b: number;
  line_a: number;
  radius_m: number;
  cluster_event_count: number;
  _severity: number;
}

const EVENT_CIRCLE_SEGMENTS = 40;

export function safeFloat(value: unknown): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

export function eventSeverity(event: Partial<FireEvent>): number {
  const score = safeFloat(event.event_score);
  if (score === null) {
    return 0;
  }
  return Math.max(0, Math.min(score, 1));
}

export function fireFillRgba(severity: number): number[] {
  if (severity >= FIRE_THRESHOLDS.veryHigh) {
    return [...FIRE_COLORS.veryHighFill];
  }
  if (severity >= FIRE_THRESHOLDS.high) {
    return [...FIRE_COLORS.highFill];
  }
  if (severity >= FIRE_THRESHOLDS.medium) {
    return [...FIRE_COLORS.mediumFill];
  }
  if (severity >= FIRE_THRESHOLDS.low) {
    return [...FIRE_COLORS.lowFill];
  }
  if (severity >= 0) {
    return [...FIRE_COLORS.veryLowFill];
  }
  return [...FIRE_COLORS.unscoredFill];
}

export function fireLineRgba(severity: number): number[] {
  if (severity >= FIRE_THRESHOLDS.high) {
    return [...FIRE_COLORS.outlineHigh];
  }
  return [...FIRE_COLORS.outlineDefault];
}

export function eventRadiusM(detectionCount: number | null): number {
  if (detectionCount === null) {
    return 3_000;
  }
  if (detectionCount >= 50) {
    return 12_000;
  }
  if (detectionCount >= 20) {
    return 8_000;
  }
  if (detectionCount >= 5) {
    return 5_000;
  }
  return 3_000;
}

export function frontLineWidth(detectionCount: number | null): number {
  if (detectionCount === null) {
    return 2;
  }
  if (detectionCount >= 50) {
    return 5;
  }
  if (detectionCount >= 20) {
    return 4;
  }
  if (detectionCount >= 5) {
    return 3;
  }
  return 2;
}

export function isActiveCandidate(event: Partial<FireEvent>): boolean {
  const severity = eventSeverity(event);
  const decision = String(event.denoiser_decision || "").trim().toLowerCase();
  if (event.review_required) {
    return true;
  }
  if (decision === "pass" || decision === "downweight") {
    return true;
  }
  return severity >= 0.6;
}

export function toRenderEvent(event: FireEvent): RenderEvent | null {
  const lat = safeFloat(event.lat);
  const lon = safeFloat(event.lon);
  if (lat === null || lon === null) {
    return null;
  }
  const severity = eventSeverity(event);
  const fill = fireFillRgba(severity);
  const line = fireLineRgba(severity);
  const detectionCount = safeFloat(event.detection_count);
  return {
    ...event,
    lat,
    lon,
    fill_r: fill[0],
    fill_g: fill[1],
    fill_b: fill[2],
    fill_a: fill[3],
    line_r: line[0],
    line_g: line[1],
    line_b: line[2],
    line_a: line[3],
    radius_m: eventRadiusM(detectionCount),
    cluster_event_count: 1,
    _severity: severity
  };
}

export function clusterEventPoints(points: RenderEvent[], zoom: number): RenderEvent[] {
  if (points.length === 0) {
    return points;
  }

  const z = Math.max(1, Math.min(zoom, 10));
  const cellDeg = Math.max(0.08, 8 / 2 ** z);
  const buckets = new Map<string, {
    count: number;
    sumLat: number;
    sumLon: number;
    maxSeverity: number;
    totalDetections: number;
    latestTime: string;
    sample: RenderEvent;
  }>();

  for (const point of points) {
    const key = `${Math.floor(point.lat / cellDeg)}:${Math.floor(point.lon / cellDeg)}`;
    const current = buckets.get(key);
    if (!current) {
      buckets.set(key, {
        count: 1,
        sumLat: point.lat,
        sumLon: point.lon,
        maxSeverity: point._severity,
        totalDetections: Number(point.detection_count || 0),
        latestTime: String(point.end_time || ""),
        sample: point
      });
      continue;
    }

    current.count += 1;
    current.sumLat += point.lat;
    current.sumLon += point.lon;
    current.maxSeverity = Math.max(current.maxSeverity, point._severity);
    current.totalDetections += Number(point.detection_count || 0);
    const currentEndTime = String(point.end_time || "");
    if (currentEndTime > current.latestTime) {
      current.latestTime = currentEndTime;
      current.sample = point;
    }
  }

  const clustered: RenderEvent[] = [];
  for (const [key, value] of buckets.entries()) {
    const sample = { ...value.sample };
    const count = value.count;
    sample.lat = value.sumLat / count;
    sample.lon = value.sumLon / count;
    sample.cluster_event_count = count;
    sample.detection_count = value.totalDetections;
    sample.end_time = value.latestTime;
    sample.event_score = value.maxSeverity;
    sample.event_id = `cluster_${key.replace(':', '_')}`;
    sample.denoiser_decision = "pass";
    sample.review_required = false;
    sample.sensor = "Cluster";
    sample.source = "Aggregated events";
    sample.radius_m = Math.max(sample.radius_m, 8000 * Math.sqrt(Math.max(count, 1)));
    sample._severity = value.maxSeverity;
    clustered.push(sample);
  }

  return clustered;
}

export function eventRingCoords(lon: number, lat: number, radiusM: number): number[][] {
  const radius = Math.max(radiusM, 300);
  const latDelta = radius / 111_000;
  const lonDelta = radius / (111_000 * Math.max(Math.abs(Math.cos((Math.PI / 180) * lat)), 0.1));

  const ring: number[][] = [];
  for (let i = 0; i < EVENT_CIRCLE_SEGMENTS; i += 1) {
    const theta = (2 * Math.PI * i) / EVENT_CIRCLE_SEGMENTS;
    let px = lon + lonDelta * Math.cos(theta);
    let py = lat + latDelta * Math.sin(theta);
    py = Math.max(Math.min(py, 85), -85);
    if (px < -180) {
      px += 360;
    } else if (px > 180) {
      px -= 360;
    }
    ring.push([px, py]);
  }
  ring.push(ring[0]);
  return ring;
}

export function normalizeGeometry(rawGeom: unknown): Geometry | null {
  if (typeof rawGeom === "string") {
    try {
      return normalizeGeometry(JSON.parse(rawGeom));
    } catch {
      return null;
    }
  }

  if (!rawGeom || typeof rawGeom !== "object") {
    return null;
  }

  const geo = rawGeom as { type?: string; geometry?: unknown };
  if (geo.type === "Feature") {
    return normalizeGeometry(geo.geometry);
  }

  if (typeof geo.type === "string") {
    return geo as Geometry;
  }

  return null;
}

export function eventFeature(event: RenderEvent): Feature {
  const fillAlpha = Math.min(Math.max(event.fill_a || 70, 45), 110);
  const lineAlpha = Math.min(Math.max(event.line_a || 180, 120), 220);

  const geometry = normalizeGeometry(event.geom_geojson) || {
    type: "Polygon",
    coordinates: [eventRingCoords(event.lon, event.lat, Math.min(Math.max(event.radius_m || 500, 500), 20_000))]
  };

  return {
    type: "Feature",
    geometry,
    properties: {
      ...event,
      lat: event.lat,
      lon: event.lon,
      fill_a: fillAlpha,
      line_a: lineAlpha
    }
  };
}

export function frontFeature(front: FireFront): Feature | null {
  const geometry = normalizeGeometry(front.geom_geojson);
  if (!geometry) {
    return null;
  }
  const severity = eventSeverity(front);
  const line = fireLineRgba(severity);
  return {
    type: "Feature",
    geometry,
    properties: {
      front_id: front.front_id,
      event_id: front.event_id,
      event_score: front.event_score,
      detection_count: front.detection_count,
      line_r: line[0],
      line_g: line[1],
      line_b: line[2],
      line_a: Math.max(120, line[3]),
      line_width: frontLineWidth(safeFloat(front.detection_count))
    }
  };
}

export function buildFrontIndexByEvent(fronts: FireFront[]): Record<string, { frontId: string; detectionCount: number }> {
  const index: Record<string, { frontId: string; detectionCount: number }> = {};
  for (const front of fronts) {
    if (!front.event_id || !front.front_id) {
      continue;
    }
    const score = Number(safeFloat(front.detection_count) || 0);
    const key = String(front.event_id);
    const current = index[key];
    if (!current || score > current.detectionCount) {
      index[key] = {
        frontId: String(front.front_id),
        detectionCount: score
      };
    }
  }
  return index;
}

export function isForecastContourVisible(properties: { horizon_hours?: number; threshold?: number }, horizons: number[], thresholds: number[]): boolean {
  const horizon = Number(properties.horizon_hours);
  const threshold = Number(properties.threshold);
  const horizonMatch = horizons.includes(horizon);
  const thresholdMatch = thresholds.some((target) => Math.abs(target - threshold) <= 0.001);
  return horizonMatch && thresholdMatch;
}
