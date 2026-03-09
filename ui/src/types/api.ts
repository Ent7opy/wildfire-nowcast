import type { Feature, FeatureCollection, MultiPolygon, Polygon } from "geojson";

export type BBox = [number, number, number, number];

export interface FireEvent {
  event_id?: string;
  lat?: number;
  lon?: number;
  geom_geojson?: unknown;
  event_score?: number;
  denoiser_decision?: string;
  review_required?: boolean;
  detection_count?: number;
  front_count?: number;
  source?: string;
  sensor?: string;
  start_time?: string;
  end_time?: string;
  [key: string]: unknown;
}

export interface FireFront {
  front_id?: string;
  event_id?: string;
  geom_geojson?: unknown;
  event_score?: number;
  detection_count?: number;
  [key: string]: unknown;
}

export interface EventsResponse {
  count: number;
  events: FireEvent[];
}

export interface FrontsResponse {
  count: number;
  fronts: FireFront[];
}

export interface FreshnessSource {
  state?: string;
  age_minutes?: number;
  last_seen_at?: string;
}

export interface DataFreshnessResponse {
  as_of?: string;
  overall_state?: string;
  sources?: Record<string, FreshnessSource>;
  stale_behavior?: Record<string, unknown>;
  idempotency_dashboard?: Record<string, unknown>;
}

export interface JitForecastStatus {
  job_id: string;
  status: "pending" | "ingesting_terrain" | "ingesting_weather" | "running_forecast" | "completed" | "failed" | string;
  progress_message?: string;
  result?: {
    run_id?: string | number;
    [key: string]: unknown;
  };
  error?: string;
}

export interface JitCreateResponse {
  job_id: string;
  status: string;
  front_id?: string;
  bbox?: number[];
}

export interface ActiveModelsResponse {
  as_of?: string;
  models?: {
    spread?: {
      model_id?: string;
    };
    [family: string]: unknown;
  };
}

export interface RiskFeatureCollection extends FeatureCollection {
  features: Array<Feature<Polygon | MultiPolygon, { risk_score?: number; [key: string]: unknown }>>;
}
