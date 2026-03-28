import type {
  BBox,
  EventsResponse,
  FireEvent,
  FrontsResponse,
  ReverseGeocodeResponse,
  RiskFeatureCollection
} from "../types/api";
import { getJson, isoFormat, toSearchParams } from "./http";

export async function getFireEvents(args: {
  bbox: BBox;
  startTime: Date;
  endTime: Date;
  minEventScore: number;
  includeReviewRequired: boolean;
  limit: number;
}): Promise<EventsResponse> {
  const [minLon, minLat, maxLon, maxLat] = args.bbox;
  return getJson<EventsResponse>(
    "/fires/events",
    {
      min_lon: minLon,
      min_lat: minLat,
      max_lon: maxLon,
      max_lat: maxLat,
      start_time: isoFormat(args.startTime),
      end_time: isoFormat(args.endTime),
      min_event_score: args.minEventScore,
      include_review_required: args.includeReviewRequired,
      limit: args.limit
    },
    { slowPath: true }
  );
}

export async function getFireFronts(args: {
  bbox: BBox;
  startTime: Date;
  endTime: Date;
  minEventScore: number;
  includeReviewRequired: boolean;
  limit: number;
}): Promise<FrontsResponse> {
  const [minLon, minLat, maxLon, maxLat] = args.bbox;
  return getJson<FrontsResponse>(
    "/fires/fronts",
    {
      min_lon: minLon,
      min_lat: minLat,
      max_lon: maxLon,
      max_lat: maxLat,
      start_time: isoFormat(args.startTime),
      end_time: isoFormat(args.endTime),
      min_event_score: args.minEventScore,
      include_review_required: args.includeReviewRequired,
      limit: args.limit
    },
    { slowPath: true }
  );
}

export async function getReverseGeocode(args: {
  lat: number;
  lon: number;
}): Promise<ReverseGeocodeResponse> {
  return getJson<ReverseGeocodeResponse>(
    "/fires/reverse-geocode",
    { lat: args.lat, lon: args.lon },
    { slowPath: true }
  );
}

export async function getRiskGrid(args: { bbox: BBox }): Promise<RiskFeatureCollection> {
  const [minLon, minLat, maxLon, maxLat] = args.bbox;
  return getJson<RiskFeatureCollection>("/risk", {
    min_lon: minLon,
    min_lat: minLat,
    max_lon: maxLon,
    max_lat: maxLat
  });
}

export function buildEventKey(event: FireEvent, lat: number, lon: number): string {
  if (event.event_id && String(event.event_id).trim().length > 0) {
    return `event_id:${event.event_id}`;
  }
  return `point:${lat.toFixed(4)}:${lon.toFixed(4)}:${String(event.end_time || "")}`;
}

export function buildFiresCsvExportUrl(baseUrl: string, args: {
  bbox: BBox;
  startTime: Date;
  endTime: Date;
  limit: number;
}): string {
  const [minLon, minLat, maxLon, maxLat] = args.bbox;
  const params = toSearchParams({
    min_lon: minLon,
    min_lat: minLat,
    max_lon: maxLon,
    max_lat: maxLat,
    start_time: isoFormat(args.startTime),
    end_time: isoFormat(args.endTime),
    format: "csv",
    limit: args.limit
  });
  return `${baseUrl}/fires/export?${params}`;
}

export function buildMapPngExportUrl(baseUrl: string, args: {
  bbox: BBox;
  startTime: Date;
  endTime: Date;
  minLikelihood: number;
  includeRisk: boolean;
  runId?: string;
}): string {
  const [minLon, minLat, maxLon, maxLat] = args.bbox;
  const params = new URLSearchParams({
    min_lon: String(minLon),
    min_lat: String(minLat),
    max_lon: String(maxLon),
    max_lat: String(maxLat),
    start_time: isoFormat(args.startTime),
    end_time: isoFormat(args.endTime),
    min_fire_likelihood: args.minLikelihood.toFixed(2),
    include_fires: "true",
    include_risk: args.includeRisk ? "true" : "false",
    include_forecast: "true"
  });
  if (args.runId) {
    params.set("run_id", args.runId);
  }
  return `${baseUrl}/map.png?${params.toString()}`;
}
