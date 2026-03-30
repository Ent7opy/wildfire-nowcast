import type { Feature, FeatureCollection, MultiPolygon, Polygon } from "geojson";

export type BBox = [number, number, number, number];

export type GeometrySource = "authoritative" | "estimated";
export type GeometryMethod =
  | "authoritative"
  | "estimated_concave"
  | "estimated_convex"
  | "estimated_point_buffer";

export interface FireEvent {
  event_id?: string;
  lat?: number;
  lon?: number;
  geom_geojson?: unknown;
  geom_source?: GeometrySource;
  geom_method?: GeometryMethod;
  geom_quality?: number;
  authority_profile?: string;
  authoritative_perimeter_id?: number;
  event_score?: number;
  denoiser_decision?: string;
  review_required?: boolean;
  detection_count?: number;
  front_count?: number;
  frp_max?: number;
  frp_mean?: number;
  brightness_max?: number;
  brightness_mean?: number;
  source?: string;
  sensor?: string;
  start_time?: string;
  end_time?: string;
  location_name?: string;
  admin1_name?: string;
  admin0_name?: string;
  country?: string;
  region_name?: string;
  [key: string]: unknown;
}

export interface ReverseGeocodeResponse {
  lat: number;
  lon: number;
  cached_lat?: number;
  cached_lon?: number;
  provider: string;
  cache_hit: boolean;
  status: "resolved" | "unresolved" | "error" | "disabled" | string;
  location_name?: string | null;
  country?: string | null;
  admin1_name?: string | null;
  admin2_name?: string | null;
  display_name?: string | null;
  updated_at?: string | null;
  expires_at?: string | null;
}

export interface FireFront {
  front_id?: string;
  event_id?: string;
  geom_geojson?: unknown;
  geom_source?: GeometrySource;
  geom_method?: GeometryMethod;
  geom_quality?: number;
  authority_profile?: string;
  authoritative_perimeter_id?: number;
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

export interface ForecastGate {
  can_run: boolean;
  would_block_if_fail_closed: boolean;
  policy: "fail_closed" | "best_effort";
  reasons: string[];
  missing_or_stale_sources: string[];
  retry_hint?: string | null;
  as_of?: string | null;
}

export interface DataFreshnessResponse {
  as_of?: string;
  overall_state?: string;
  sources?: Record<string, FreshnessSource>;
  forecast_gate?: ForecastGate;
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

export interface DenoiserReviewPayload {
  event_score?: number;
  frp_max?: number;
  confidence_max?: number;
  fail_closed_hard_bypass?: boolean;
}

export interface DenoiserReviewItem {
  id: number;
  event_id: string;
  fire_detection_id?: number | null;
  reason: string;
  severity: string;
  status: string;
  payload_json?: DenoiserReviewPayload | null;
  resolved_by?: string | null;
  resolved_notes?: string | null;
  resolved_at?: string | null;
  created_at: string;
  updated_at: string;
  centroid_lat?: number | null;
  centroid_lon?: number | null;
  country_code?: string | null;
  region_name?: string | null;
  nearest_place?: string | null;
  terrain_label?: string | null;
}

export interface ReviewQueueResponse {
  as_of: string;
  rows: DenoiserReviewItem[];
}

export interface ResolveReviewResponse {
  as_of: string;
  event_id: string;
  updated: number;
}

export interface AOI {
  id: string;
  name: string;
  description?: string | null;
  tags?: Record<string, unknown> | null;
  owner_id?: string | null;
  geometry: Record<string, unknown>;
  bbox: Record<string, unknown>;
  area_km2: number;
  vertex_count: number;
  created_at: string;
  updated_at: string;
  watch_enabled: boolean;
  watch_interval_minutes?: number | null;
  watch_alert_threshold?: number | null;
  watch_last_checked_at?: string | null;
  watch_last_alerted_at?: string | null;
  watch_last_spread_prob?: number | null;
}

export interface AOIListResponse {
  items: AOI[];
  count: number;
}

export interface WatchlistItem {
  id: string;
  name: string;
  watch_enabled: boolean;
  watch_interval_minutes?: number | null;
  watch_alert_threshold?: number | null;
  watch_last_checked_at?: string | null;
  watch_last_alerted_at?: string | null;
  watch_last_spread_prob?: number | null;
  alert_active: boolean;
}

export interface WatchlistResponse {
  items: WatchlistItem[];
  count: number;
}

export interface WatchConfigRequest {
  enabled: boolean;
  interval_minutes?: number | null;
  alert_threshold?: number | null;
}
