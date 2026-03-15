import { apiBaseUrlCandidates } from "../config/runtime";
import type {
  ActiveModelsResponse,
  BBox,
  DataFreshnessResponse,
  EventsResponse,
  FireEvent,
  FrontsResponse,
  JitCreateResponse,
  JitForecastStatus,
  ReverseGeocodeResponse,
  RiskFeatureCollection
} from "../types/api";

export class ApiError extends Error {
  statusCode?: number;
  url?: string;
  responseText?: string;

  constructor(message: string, options?: { statusCode?: number; url?: string; responseText?: string }) {
    super(message);
    this.name = "ApiError";
    this.statusCode = options?.statusCode;
    this.url = options?.url;
    this.responseText = options?.responseText;
  }
}

export class ApiUnavailableError extends ApiError {
  constructor(message: string, options?: { url?: string }) {
    super(message, options);
    this.name = "ApiUnavailableError";
  }
}

const GET_CONNECT_TIMEOUT = 2_000;
const GET_READ_TIMEOUT = 8_000;
const GET_RETRY_READ_TIMEOUT = 15_000;

function isoFormat(date: Date): string {
  const d = new Date(date);
  d.setMilliseconds(0);
  return d.toISOString().replace(".000", "");
}

function withTimeout(timeoutMs: number): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const id = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  return {
    signal: controller.signal,
    cancel: () => globalThis.clearTimeout(id)
  };
}

function toSearchParams(params: Record<string, unknown>): string {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === null || value === undefined) {
      return;
    }
    if (typeof value === "boolean") {
      searchParams.set(key, value ? "true" : "false");
      return;
    }
    searchParams.set(key, String(value));
  });
  return searchParams.toString();
}

async function getJson<T>(
  path: string,
  params: Record<string, unknown>,
  options?: { slowPath?: boolean }
): Promise<T> {
  const candidates = apiBaseUrlCandidates();
  let lastUnavailable: ApiUnavailableError | null = null;

  for (const base of candidates) {
    const query = toSearchParams(params);
    const url = `${base}${path}${query ? `?${query}` : ""}`;

    const firstAttempt = withTimeout(GET_CONNECT_TIMEOUT + GET_READ_TIMEOUT);
    try {
      const response = await fetch(url, { method: "GET", signal: firstAttempt.signal });
      firstAttempt.cancel();
      if (!response.ok) {
        const text = await response.text();
        throw new ApiError("Non-200 response from API", {
          statusCode: response.status,
          url,
          responseText: text
        });
      }
      return (await response.json()) as T;
    } catch (error) {
      firstAttempt.cancel();
      const aborted = error instanceof DOMException && error.name === "AbortError";
      const shouldRetry = options?.slowPath && aborted;

      if (shouldRetry) {
        const secondAttempt = withTimeout(GET_CONNECT_TIMEOUT + GET_RETRY_READ_TIMEOUT);
        try {
          const response = await fetch(url, { method: "GET", signal: secondAttempt.signal });
          secondAttempt.cancel();
          if (!response.ok) {
            const text = await response.text();
            throw new ApiError("Non-200 response from API", {
              statusCode: response.status,
              url,
              responseText: text
            });
          }
          return (await response.json()) as T;
        } catch (innerErr) {
          secondAttempt.cancel();
          const message = innerErr instanceof Error ? innerErr.message : "API unavailable";
          lastUnavailable = new ApiUnavailableError(message, { url });
          continue;
        }
      }

      if (error instanceof ApiError) {
        throw error;
      }

      const message = error instanceof Error ? error.message : "API unavailable";
      lastUnavailable = new ApiUnavailableError(message, { url });
    }
  }

  if (lastUnavailable) {
    throw lastUnavailable;
  }

  throw new ApiUnavailableError("API unavailable");
}

async function postJson<T>(path: string, payload: Record<string, unknown>, acceptedStatus = 200): Promise<T> {
  const candidates = apiBaseUrlCandidates();
  let lastUnavailable: ApiUnavailableError | null = null;

  for (const base of candidates) {
    const url = `${base}${path}`;
    const timeout = withTimeout(15_000);
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload),
        signal: timeout.signal
      });
      timeout.cancel();

      if (response.status !== acceptedStatus) {
        const text = await response.text();
        let message = `Non-${acceptedStatus} response from API`;
        try {
          const parsed = JSON.parse(text) as { detail?: string; message?: string };
          message = parsed.message || parsed.detail || message;
        } catch {
          // ignore parse errors
        }
        throw new ApiError(message, { statusCode: response.status, url, responseText: text });
      }

      return (await response.json()) as T;
    } catch (error) {
      timeout.cancel();
      if (error instanceof ApiError) {
        throw error;
      }
      const message = error instanceof Error ? error.message : "API unavailable";
      lastUnavailable = new ApiUnavailableError(message, { url });
    }
  }

  if (lastUnavailable) {
    throw lastUnavailable;
  }

  throw new ApiUnavailableError("API unavailable");
}

export async function getDataFreshnessStatus(): Promise<DataFreshnessResponse> {
  return getJson<DataFreshnessResponse>("/health/data-freshness", {}, { slowPath: true });
}

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

export async function getJitForecastStatus(jobId: string): Promise<JitForecastStatus> {
  return getJson<JitForecastStatus>(`/forecast/jit/${jobId}`, {});
}

export async function getActiveSpreadModelId(): Promise<string> {
  const payload = await getJson<ActiveModelsResponse>("/internal/models/active", {});
  const modelId = payload.models?.spread?.model_id;
  if (modelId && modelId.trim().length > 0) {
    return modelId.trim();
  }
  throw new ApiError("No active spread model is promoted. Promote a spread model and retry.", {
    statusCode: 422
  });
}

export async function createJitForecast(args: {
  bbox: BBox;
  forecastReferenceTime: Date;
  horizonsHours: number[];
  modelId: string;
}): Promise<JitCreateResponse> {
  return postJson<JitCreateResponse>(
    "/forecast/jit",
    {
      bbox: args.bbox,
      forecast_reference_time: isoFormat(args.forecastReferenceTime),
      horizons_hours: args.horizonsHours,
      model_id: args.modelId
    },
    202
  );
}

export async function createJitForecastFromFront(args: {
  frontId: string;
  bufferKm: number;
  forecastReferenceTime: Date;
  horizonsHours: number[];
  modelId: string;
}): Promise<JitCreateResponse> {
  return postJson<JitCreateResponse>(
    "/forecast/jit/from-front",
    {
      front_id: args.frontId,
      buffer_km: args.bufferKm,
      forecast_reference_time: isoFormat(args.forecastReferenceTime),
      horizons_hours: args.horizonsHours,
      model_id: args.modelId
    },
    202
  );
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

export interface ArchiveAvailabilityResponse {
  has_data: boolean;
  detection_count: number;
}

export interface ArchiveIngestResponse {
  job_id: string;
  estimated_minutes: number;
}

export interface ArchiveIngestStatusResponse {
  status: string;
  error: string | null;
}

export async function getArchiveIngestStatus(jobId: string): Promise<ArchiveIngestStatusResponse> {
  return getJson<ArchiveIngestStatusResponse>(`/fires/archive/ingest/${jobId}`);
}

export async function checkArchiveAvailability(
  date: string,
  timeframe: string
): Promise<ArchiveAvailabilityResponse> {
  return getJson<ArchiveAvailabilityResponse>("/fires/archive/availability", { date, timeframe });
}

export async function triggerArchiveIngest(
  date: string,
  timeframe: string
): Promise<ArchiveIngestResponse> {
  return postJson<ArchiveIngestResponse>("/fires/archive/ingest", { date, timeframe }, 202);
}

export function buildEventKey(event: FireEvent, lat: number, lon: number): string {
  if (event.event_id && String(event.event_id).trim().length > 0) {
    return `event_id:${event.event_id}`;
  }
  return `point:${lat.toFixed(4)}:${lon.toFixed(4)}:${String(event.end_time || "")}`;
}
