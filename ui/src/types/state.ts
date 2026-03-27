import type { FireEvent } from "./api";

export type ViewMode = 'live' | 'archive';
export type ArchiveTimeframe = 'morning' | 'afternoon' | 'evening' | 'night';
export type ArchiveSubMode = 'single' | 'range';

export interface ArchiveModeState {
  viewMode: ViewMode;
  archiveDate: string | null;       // 'YYYY-MM-DD' — used in single-day mode
  archiveTimeframe: ArchiveTimeframe | null;
  // Range-mode fields (active when archiveSubMode === 'range')
  archiveSubMode: ArchiveSubMode;
  rangeStart: string | null;        // 'YYYY-MM-DD'
  rangeEnd: string | null;          // 'YYYY-MM-DD'
  rangeJobId: string | null;
  scrubDate: string | null;         // current day displayed in range playback
}

export interface FiltersState {
  hoursStart: number;
  hoursEnd: number;
  minLikelihood: number;
  activeOnly: boolean;
  clusterPoints: boolean;
}

export interface LayersState {
  showFires: boolean;
  showFronts: boolean;
  showForecast: boolean;
  showRisk: boolean;
  basemap: 'dark' | 'light' | 'satellite';
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

export interface ForecastRunMeta {
  weatherRunId: string | null;
  confidenceLevel: string | null;
  fallbackUsed: boolean;
  weatherBiasApplied: boolean | null;
}

export interface ForecastJobState {
  jobId: string | null;
  pollCount: number;
  lastForecast: ({ run: { id: string }; runMeta?: ForecastRunMeta } & Partial<ForecastRequestContext>) | null;
  activeRequest: ForecastRequestContext | null;
  notification: ForecastNotification | null;
}

export type AssistantConfidenceFilter = "All" | "High";

export interface AssistantViewEventSummary {
  eventId: string;
  locationLabel: string;
  lat: number | null;
  lon: number | null;
  eventScore: number | null;
  detectionCount: number;
  frontCount: number;
  endTime: string | null;
  sensor: string | null;
  source: string | null;
  reviewRequired: boolean;
  denoiserDecision: string | null;
}

export interface AssistantViewContext {
  updatedAt: number;
  searchQuery: string;
  confidenceFilter: AssistantConfidenceFilter;
  visibleEventCount: number;
  filteredEventCount: number;
  topEvents: AssistantViewEventSummary[];
}

export type SafetyTier = 'SAFE' | 'WATCH' | 'WARNING' | 'DANGER';

export interface UserLocationState {
  lat: number;
  lon: number;
  accuracyM: number;
  acquiredAt: number;
}

export interface SafetyModeState {
  enabled: boolean;
  userLocation: UserLocationState | null;
  locationPermission: 'unknown' | 'granted' | 'denied' | 'requesting';
  proximityRadiusKm: number;
  nearestFireDistanceKm: number | null;
  safetyTier: SafetyTier;
  pendingBriefingPrompt: string | null;
}
