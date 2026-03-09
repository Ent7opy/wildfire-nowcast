import { create } from "zustand";

import type { FireEvent } from "../types/api";
import type {
  FiltersState,
  ForecastJobState,
  ForecastNotification,
  ForecastRequestContext,
  LayersState,
  MapViewState
} from "../types/state";
import { matchingPreset, parseBoolFlag } from "../utils/presets";

const DEFAULT_FILTERS: FiltersState = {
  hoursStart: 24,
  hoursEnd: 0,
  minLikelihood: 0,
  activeOnly: true,
  clusterPoints: true
};

const DEFAULT_LAYERS: LayersState = {
  showFires: true,
  showForecast: true,
  showRisk: false
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

interface AppStoreState {
  initializedFromUrl: boolean;
  filters: FiltersState;
  layers: LayersState;
  mapView: MapViewState;
  selectedEvent: FireEvent | null;
  lastClick: { lat: number; lng: number } | null;
  frontIndexByEvent: Record<string, { frontId: string; detectionCount: number }>;
  activePreset: string | null;
  forecast: ForecastJobState;
  initializeFromUrl: () => void;
  setFilters: (patch: Partial<FiltersState>) => void;
  applyPreset: (preset: { name: string; hoursStart: number; hoursEnd: number; likelihood: number }) => void;
  setRiskVisibility: (visible: boolean) => void;
  setMapView: (next: MapViewState) => void;
  setSelectedEvent: (event: FireEvent | null) => void;
  setLastClick: (coords: { lat: number; lng: number } | null) => void;
  clearSelection: () => void;
  setFrontIndexByEvent: (index: Record<string, { frontId: string; detectionCount: number }>) => void;
  startForecastJob: (jobId: string, request: ForecastRequestContext) => void;
  incrementForecastPoll: () => void;
  completeForecastJob: (runId: string) => void;
  clearForecastJob: () => void;
  setForecastNotification: (notification: ForecastNotification | null) => void;
  focusMapOnPoint: (lat: number, lon: number, minZoom: number) => void;
}

function updatePreset(filters: FiltersState): string {
  return matchingPreset(filters) || "Custom";
}

function parseNumber(value: string | null): number | null {
  if (value === null) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export const useAppStore = create<AppStoreState>((set, get) => ({
  initializedFromUrl: false,
  filters: DEFAULT_FILTERS,
  layers: DEFAULT_LAYERS,
  mapView: DEFAULT_MAP_VIEW,
  selectedEvent: null,
  lastClick: null,
  frontIndexByEvent: {},
  activePreset: updatePreset(DEFAULT_FILTERS),
  forecast: DEFAULT_FORECAST_STATE,

  initializeFromUrl: () => {
    if (get().initializedFromUrl || typeof window === "undefined") {
      return;
    }

    const params = new URLSearchParams(window.location.search);
    const next: FiltersState = { ...get().filters };

    const start = parseNumber(params.get("start"));
    const end = parseNumber(params.get("end"));
    const likelihood = parseNumber(params.get("likelihood"));
    const activeOnly = parseBoolFlag(params.get("active_only"));
    const cluster = parseBoolFlag(params.get("cluster"));

    if (start !== null) {
      next.hoursStart = Math.max(1, Math.min(48, Math.floor(start)));
    }
    if (end !== null) {
      next.hoursEnd = Math.max(0, Math.min(47, Math.floor(end)));
    }
    if (next.hoursStart <= next.hoursEnd) {
      next.hoursStart = Math.min(48, next.hoursEnd + 1);
    }
    if (likelihood !== null) {
      next.minLikelihood = Math.max(0, Math.min(1, likelihood));
    }
    if (activeOnly !== null) {
      next.activeOnly = activeOnly;
    }
    if (cluster !== null) {
      next.clusterPoints = cluster;
    }

    set({
      initializedFromUrl: true,
      filters: next,
      layers: {
        ...get().layers,
        showRisk: cluster === false ? false : get().layers.showRisk
      },
      activePreset: updatePreset(next)
    });
  },

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

  completeForecastJob: (runId) => {
    set((state) => ({
      forecast: {
        ...state.forecast,
        jobId: null,
        pollCount: 0,
        lastForecast: {
          run: { id: runId },
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
  }
}));
