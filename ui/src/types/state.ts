import type { FireEvent } from "./api";

export interface FiltersState {
  hoursStart: number;
  hoursEnd: number;
  minLikelihood: number;
  activeOnly: boolean;
  clusterPoints: boolean;
}

export interface LayersState {
  showFires: boolean;
  showForecast: boolean;
  showRisk: boolean;
}

export interface MapViewState {
  latitude: number;
  longitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
  transitionDuration?: number;
}

export interface ForecastRequestContext {
  eventId?: string;
  eventKey?: string;
  frontId?: string;
  lat: number;
  lon: number;
  locationLabel: string;
  eventSnapshot: FireEvent;
}

export interface ForecastNotification {
  kind: "info" | "ready" | "error";
  message: string;
  runId?: string;
  createdAt: number;
  ttlSeconds: number;
  target?: {
    lat?: number;
    lon?: number;
    eventSnapshot?: FireEvent;
    eventId?: string;
    eventKey?: string;
  };
}

export interface ForecastJobState {
  jobId: string | null;
  pollCount: number;
  lastForecast: ({ run: { id: string } } & Partial<ForecastRequestContext>) | null;
  activeRequest: ForecastRequestContext | null;
  notification: ForecastNotification | null;
}
