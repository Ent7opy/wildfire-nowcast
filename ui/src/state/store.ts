import { create } from "zustand";

import type { FireEvent } from "../types/api";
import type {
  ArchiveModeState,
  ArchiveTimeframe,
  AssistantViewContext,
  FiltersState,
  ForecastJobState,
  ForecastNotification,
  ForecastRequestContext,
  ForecastRunMeta,
  LayersState,
  MapViewState,
  SafetyModeState,
  SafetyTier,
  UserLocationState,
  ViewMode
} from "../types/state";
import { currentTimeframe } from "../utils/time";
import { matchingPreset } from "../utils/presets";

const DEFAULT_FILTERS: FiltersState = {
  hoursStart: 6,
  hoursEnd: 0,
  minLikelihood: 0,
  activeOnly: true,
  clusterPoints: true
};

const DEFAULT_LAYERS: LayersState = {
  showFires: true,
  showFronts: true,
  showForecast: true,
  showRisk: false,
  basemap: 'dark'
};

const DEFAULT_MAP_VIEW: MapViewState = {
  latitude: 20,
  longitude: 0,
  zoom: 1,
  pitch: 0,
  bearing: 0
};

const DEFAULT_FORECAST_STATE: ForecastJobState = {
  jobId: null,
  pollCount: 0,
  lastForecast: null,
  activeRequest: null,
  notification: null
};

const DEFAULT_ARCHIVE_STATE: ArchiveModeState = {
  viewMode: 'live',
  archiveDate: null,
  archiveTimeframe: null,
};

function computeSafetyTier(nearestKm: number | null): SafetyTier {
  if (nearestKm === null) return 'SAFE';
  if (nearestKm <= 5)  return 'DANGER';
  if (nearestKm <= 20) return 'WARNING';
  if (nearestKm <= 50) return 'WATCH';
  return 'SAFE';
}

const DEFAULT_SAFETY_STATE: SafetyModeState = {
  enabled: false,
  userLocation: null,
  locationPermission: 'unknown',
  proximityRadiusKm: 50,
  nearestFireDistanceKm: null,
  safetyTier: 'SAFE',
  pendingBriefingPrompt: null,
};

const DEFAULT_ASSISTANT_VIEW_CONTEXT: AssistantViewContext = {
  updatedAt: Date.now(),
  searchQuery: "",
  confidenceFilter: "All",
  visibleEventCount: 0,
  filteredEventCount: 0,
  topEvents: []
};

interface AppStoreState {
  filters: FiltersState;
  layers: LayersState;
  mapView: MapViewState;
  selectedEvent: FireEvent | null;
  lastClick: { lat: number; lng: number } | null;
  frontIndexByEvent: Record<string, { frontId: string; detectionCount: number }>;
  activePreset: string | null;
  forecast: ForecastJobState;
  assistantViewContext: AssistantViewContext;
  archive: ArchiveModeState;
  safety: SafetyModeState;
  setFilters: (patch: Partial<FiltersState>) => void;
  applyPreset: (preset: { name: string; hoursStart: number; hoursEnd: number; likelihood: number }) => void;
  setLayersState: (patch: Partial<LayersState>) => void;
  setRiskVisibility: (visible: boolean) => void;
  setMapView: (next: MapViewState) => void;
  setSelectedEvent: (event: FireEvent | null) => void;
  setLastClick: (coords: { lat: number; lng: number } | null) => void;
  clearSelection: () => void;
  setFrontIndexByEvent: (index: Record<string, { frontId: string; detectionCount: number }>) => void;
  startForecastJob: (jobId: string, request: ForecastRequestContext) => void;
  incrementForecastPoll: () => void;
  completeForecastJob: (runId: string, runMeta?: ForecastRunMeta) => void;
  clearForecastJob: () => void;
  setForecastNotification: (notification: ForecastNotification | null) => void;
  setAssistantViewContext: (context: AssistantViewContext) => void;
  focusMapOnPoint: (lat: number, lon: number, minZoom: number) => void;
  enterArchiveMode: () => void;
  exitToLiveMode: () => void;
  setArchiveDate: (date: string) => void;
  setArchiveTimeframe: (tf: ArchiveTimeframe) => void;
  setViewMode: (mode: ViewMode) => void;
  enableSafetyMode: () => void;
  disableSafetyMode: () => void;
  setSafetyLocation: (location: UserLocationState | null) => void;
  setSafetyLocationPermission: (status: SafetyModeState['locationPermission']) => void;
  updateSafetyMetrics: (nearestKm: number | null) => void;
  setSafetyProximityRadius: (km: number) => void;
  requestAssistantBriefing: (prompt: string) => void;
  clearAssistantBriefingPrompt: () => void;
}

function updatePreset(filters: FiltersState): string {
  return matchingPreset(filters) || "Custom";
}

export const useAppStore = create<AppStoreState>((set) => ({
  filters: DEFAULT_FILTERS,
  layers: DEFAULT_LAYERS,
  mapView: DEFAULT_MAP_VIEW,
  selectedEvent: null,
  lastClick: null,
  frontIndexByEvent: {},
  activePreset: updatePreset(DEFAULT_FILTERS),
  forecast: DEFAULT_FORECAST_STATE,
  assistantViewContext: DEFAULT_ASSISTANT_VIEW_CONTEXT,
  archive: DEFAULT_ARCHIVE_STATE,
  safety: DEFAULT_SAFETY_STATE,

  setFilters: (patch) => {
    set((state) => {
      const next = { ...state.filters, ...patch };
      if (next.hoursStart <= next.hoursEnd) {
        next.hoursStart = next.hoursEnd + 1;
      }
      if (!next.clusterPoints) {
        return {
          filters: next,
          activePreset: updatePreset(next),
          layers: { ...state.layers, showRisk: false }
        };
      }
      return { filters: next, activePreset: updatePreset(next) };
    });
  },

  applyPreset: (preset) => {
    set((state) => ({
      filters: {
        ...state.filters,
        hoursStart: preset.hoursStart,
        hoursEnd: preset.hoursEnd,
        minLikelihood: preset.likelihood
      },
      activePreset: preset.name
    }));
  },

  setLayersState: (patch) => {
    set((state) => ({
      layers: { ...state.layers, ...patch }
    }));
  },

  setRiskVisibility: (visible) => {
    set((state) => ({
      layers: {
        ...state.layers,
        showRisk: state.filters.clusterPoints ? visible : false
      }
    }));
  },

  setMapView: (next) => set({ mapView: next }),

  setSelectedEvent: (event) => set({ selectedEvent: event }),

  setLastClick: (coords) => set({ lastClick: coords }),

  clearSelection: () => set({ selectedEvent: null, lastClick: null }),

  setFrontIndexByEvent: (index) => set({ frontIndexByEvent: index }),

  startForecastJob: (jobId, request) => {
    set((state) => ({
      forecast: {
        ...state.forecast,
        jobId,
        pollCount: 0,
        activeRequest: request
      }
    }));
  },

  incrementForecastPoll: () => {
    set((state) => ({
      forecast: {
        ...state.forecast,
        pollCount: state.forecast.pollCount + 1
      }
    }));
  },

  completeForecastJob: (runId, runMeta) => {
    set((state) => ({
      forecast: {
        ...state.forecast,
        jobId: null,
        pollCount: 0,
        lastForecast: {
          run: { id: runId },
          ...(runMeta ? { runMeta } : {}),
          ...(state.forecast.activeRequest || {})
        },
        activeRequest: null
      },
      layers: {
        ...state.layers,
        showForecast: true
      }
    }));
  },

  clearForecastJob: () => {
    set((state) => ({
      forecast: {
        ...state.forecast,
        jobId: null,
        pollCount: 0,
        activeRequest: null
      }
    }));
  },

  setForecastNotification: (notification) => {
    set((state) => ({
      forecast: {
        ...state.forecast,
        notification
      }
    }));
  },

  setAssistantViewContext: (context) => {
    set({ assistantViewContext: context });
  },

  enterArchiveMode: () => {
    const today = new Date();
    const date = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    set((state) => ({
      archive: { viewMode: 'archive', archiveDate: date, archiveTimeframe: currentTimeframe() },
      selectedEvent: null,
      lastClick: null,
      forecast: { ...state.forecast, jobId: null, pollCount: 0, activeRequest: null, notification: null },
      layers: { ...state.layers, showForecast: false }
    }));
  },

  exitToLiveMode: () => {
    set((state) => ({
      archive: DEFAULT_ARCHIVE_STATE,
      selectedEvent: null,
      lastClick: null,
      layers: { ...state.layers, showForecast: Boolean(state.forecast.lastForecast) }
    }));
  },

  setArchiveDate: (date) => {
    set((state) => ({ archive: { ...state.archive, archiveDate: date }, selectedEvent: null, lastClick: null }));
  },

  setArchiveTimeframe: (tf) => {
    set((state) => ({ archive: { ...state.archive, archiveTimeframe: tf }, selectedEvent: null, lastClick: null }));
  },

  setViewMode: (mode) => {
    set((state) => ({ archive: { ...state.archive, viewMode: mode } }));
  },

  focusMapOnPoint: (lat, lon, minZoom) => {
    set((state) => ({
      mapView: {
        ...state.mapView,
        latitude: lat,
        longitude: lon,
        zoom: Math.max(state.mapView.zoom, minZoom),
        transitionDuration: 700
      }
    }));
  },

  enableSafetyMode: () => set((state) => ({
    safety: { ...state.safety, enabled: true }
  })),

  disableSafetyMode: () => set({ safety: DEFAULT_SAFETY_STATE }),

  setSafetyLocation: (location) => set((state) => ({
    safety: {
      ...state.safety,
      userLocation: location,
      locationPermission: location ? 'granted' : state.safety.locationPermission
    }
  })),

  setSafetyLocationPermission: (status) => set((state) => ({
    safety: { ...state.safety, locationPermission: status }
  })),

  updateSafetyMetrics: (nearestKm) => set((state) => ({
    safety: {
      ...state.safety,
      nearestFireDistanceKm: nearestKm,
      safetyTier: computeSafetyTier(nearestKm)
    }
  })),

  setSafetyProximityRadius: (km) => set((state) => ({
    safety: { ...state.safety, proximityRadiusKm: km }
  })),

  requestAssistantBriefing: (prompt) => set((state) => ({
    safety: { ...state.safety, pendingBriefingPrompt: prompt }
  })),

  clearAssistantBriefingPrompt: () => set((state) => ({
    safety: { ...state.safety, pendingBriefingPrompt: null }
  })),
}));
