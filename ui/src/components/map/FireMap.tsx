import { useEffect, useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer, ScatterplotLayer } from "@deck.gl/layers";
import { MVTLayer } from "@deck.gl/geo-layers";
import type { PickingInfo } from "@deck.gl/core";
import Map from "react-map-gl/maplibre";
import maplibregl from "maplibre-gl";
import {
  Alert,
  Box,
  Button,
  Divider,
  FormControlLabel,
  IconButton,
  Popover,
  Switch,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography
} from "@mui/material";
import GpsFixedIcon from "@mui/icons-material/GpsFixed";
import ReplayIcon from "@mui/icons-material/Replay";
import MapOutlinedIcon from "@mui/icons-material/MapOutlined";
import PublicIcon from "@mui/icons-material/Public";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import { normalizePickedEvent } from "../../utils/selection";
import { buildUserLocationLayers } from "../../utils/userLocationLayers";
import { useGeolocation } from "../../hooks/useGeolocation";
import { useQuery } from "@tanstack/react-query";
import type { Feature } from "geojson";

import { apiPublicBaseUrl } from "../../config/runtime";
import { getFireEvents, getFireFronts, getRiskGrid } from "../../api/client";
import { getWeatherWarnings } from "../../api/fires";
import { getIgnitionGrid } from "../../api/ignition";
import { ApiError } from "../../api/http";
import { useAppStore } from "../../state/store";
import type { FireEvent } from "../../types/api";
import {
  FORECAST_DEFAULT_HORIZONS,
  FORECAST_DEFAULT_THRESHOLDS,
  RISK_THRESHOLDS,
  buildFrontIndexByEvent,
  clusterEventPoints,
  eventFeature,
  frontFeature,
  geometryBounds,
  isActiveCandidate,
  isForecastContourVisible,
  toRenderEvent
} from "../../map/layerUtils";
import { computeArchiveTimeRange, computeFullDayTimeRange, computeTimeRange } from "../../utils/time";
import { eventLimitForZoom, frontLimitForZoom, shouldLoadFronts, shouldRenderCentroids, viewportBbox } from "../../utils/mapMath";
import { useDebounce } from "../../hooks/useDebounce";
import { selectionViewFromBounds } from "../../utils/mapSelection";
import {
  BASEMAP_DARK,
  BASEMAP_LIGHT,
  BASEMAP_SATELLITE,
  FORECAST_FILL,
  FORECAST_STROKE,
  HIGH_CONFIDENCE_THRESHOLD,
  MIN_SELECTION_ZOOM,
  MAX_SELECTION_ZOOM,
  SELECTION_TARGET_OCCUPANCY,
  SELECTED_FRONT_COLOR,
  SELECTED_EVENT_FILL,
  SELECTED_EVENT_STROKE
} from "./layers/layerConfig";
import { toFiniteNumber } from "../../utils/priorityFeed";
import { geometryProvenanceLabel } from "../fire-details/types";
import type { IgnitionCell } from "../../types/api";

type ConfidenceFilter = "All" | "High";

interface FireMapProps {
  onVisibleEventsChange: (events: FireEvent[]) => void;
  searchQuery?: string;
  confidenceFilter?: ConfidenceFilter;
}

const WARNING_SEVERITY_RGB: Record<string, [number, number, number]> = {
  red: [239, 68, 68],
  orange: [249, 115, 22],
  yellow: [234, 179, 8],
};

function warningSeverityColor(sev: string, alpha: number): [number, number, number, number] {
  const [r, g, b] = WARNING_SEVERITY_RGB[sev] ?? WARNING_SEVERITY_RGB.yellow;
  return [r, g, b, alpha];
}

// Ignition risk colour palette — distinct hue family from the risk grid
// (risk grid: green/gold/crimson; ignition: amber/orange-red/magenta)
const IGNITION_LEVEL_COLOR: Record<string, [number, number, number, number]> = {
  low:      [0, 0, 0, 0],           // invisible
  elevated: [245, 158, 11, 110],    // amber
  high:     [249, 115, 22, 175],    // orange-red
  critical: [220, 38, 127, 210],    // deep magenta
};

function ignitionCellColor(cell: IgnitionCell): [number, number, number, number] {
  return IGNITION_LEVEL_COLOR[cell.level] ?? IGNITION_LEVEL_COLOR.low;
}

function isHighConfidence(event: FireEvent): boolean {
  const score = toFiniteNumber(event.event_score);
  return score !== null && score >= HIGH_CONFIDENCE_THRESHOLD;
}

function eventSearchText(event: FireEvent): string {
  return [
    event.event_id,
    event.location_name,
    event.region_name,
    event.admin1_name,
    event.admin0_name,
    event.country,
    event.source,
    event.sensor,
    event.denoiser_decision
  ]
    .map((value) => String(value || "").toLowerCase())
    .join(" ");
}

export default function FireMap({
  onVisibleEventsChange,
  searchQuery = "",
  confidenceFilter = "All"
}: FireMapProps) {
  const filters = useAppStore((s) => s.filters);
  const layersState = useAppStore((s) => s.layers);
  const mapView = useAppStore((s) => s.mapView);
  const forecast = useAppStore((s) => s.forecast);
  const selectedEvent = useAppStore((s) => s.selectedEvent);
  const archive = useAppStore((s) => s.archive);
  const safety = useAppStore((s) => s.safety);
  const ignitionHorizon = useAppStore((s) => s.ignitionHorizon);
  const { requestLocation } = useGeolocation();
  const setMapView = useAppStore((s) => s.setMapView);
  const setSelectedEvent = useAppStore((s) => s.setSelectedEvent);
  const setLastClick = useAppStore((s) => s.setLastClick);
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);
  const setFrontIndexByEvent = useAppStore((s) => s.setFrontIndexByEvent);
  const setLayersState = useAppStore((s) => s.setLayersState);
  const exitToLiveMode = useAppStore((s) => s.exitToLiveMode);
  const setIgnitionHorizon = useAppStore((s) => s.setIgnitionHorizon);
  const setIgnitionData = useAppStore((s) => s.setIgnitionData);

  const isArchiveMode = archive.viewMode === "archive";

  const [layersPanelAnchor, setLayersPanelAnchor] = useState<HTMLElement | null>(null);

  const debouncedMapView = useDebounce(mapView, 400);
  const bbox = useMemo(() => viewportBbox(debouncedMapView), [debouncedMapView]);
  const timeRange = useMemo(() => {
    if (isArchiveMode) {
      if (archive.archiveSubMode === "range" && archive.scrubDate) {
        return computeFullDayTimeRange(archive.scrubDate);
      }
      if (archive.archiveDate && archive.archiveTimeframe) {
        return computeArchiveTimeRange(archive.archiveDate, archive.archiveTimeframe);
      }
    }
    return computeTimeRange(filters);
  }, [isArchiveMode, archive.archiveSubMode, archive.scrubDate, archive.archiveDate, archive.archiveTimeframe, filters]);
  const normalizedSearch = searchQuery.trim().toLowerCase();

  const eventsQuery = useQuery({
    queryKey: ["fire-events", bbox, timeRange.startTime.toISOString(), timeRange.endTime.toISOString(), filters.minLikelihood, debouncedMapView.zoom, archive.viewMode, archive.archiveDate, archive.archiveTimeframe],
    queryFn: () =>
      getFireEvents({
        bbox,
        startTime: timeRange.startTime,
        endTime: timeRange.endTime,
        minEventScore: filters.minLikelihood,
        includeReviewRequired: true,
        limit: eventLimitForZoom(debouncedMapView.zoom)
      }),
    placeholderData: (prev) => prev,
    // In live mode, refresh data every 60s. Archive queries are static — no interval needed.
    refetchInterval: isArchiveMode ? false : 60_000,
    staleTime: isArchiveMode ? Infinity : 50_000,
  });

  const frontsQuery = useQuery({
    queryKey: ["fire-fronts", bbox, timeRange.startTime.toISOString(), timeRange.endTime.toISOString(), filters.minLikelihood, debouncedMapView.zoom, archive.viewMode, archive.archiveDate, archive.archiveTimeframe],
    queryFn: () =>
      getFireFronts({
        bbox,
        startTime: timeRange.startTime,
        endTime: timeRange.endTime,
        minEventScore: filters.minLikelihood,
        includeReviewRequired: true,
        limit: frontLimitForZoom(debouncedMapView.zoom)
      }),
    enabled: shouldLoadFronts(debouncedMapView.zoom),
    placeholderData: (prev) => prev,
    // In live mode, refresh data every 60s. Archive queries are static — no interval needed.
    refetchInterval: isArchiveMode ? false : 60_000,
    staleTime: isArchiveMode ? Infinity : 50_000,
  });

  const riskQuery = useQuery({
    queryKey: ["risk-grid", bbox, layersState.showRisk],
    queryFn: () => getRiskGrid({ bbox }),
    enabled: layersState.showRisk,
    placeholderData: (prev) => prev
  });

  const warningsQuery = useQuery({
    queryKey: ["weather-warnings", bbox, layersState.showWarnings],
    queryFn: () => getWeatherWarnings({ bbox }),
    enabled: layersState.showWarnings,
    staleTime: 15 * 60 * 1000,  // MeteoAlarm feed updates every ~15 min
    placeholderData: (prev) => prev
  });

  const ignitionQuery = useQuery({
    queryKey: ["ignition", bbox, ignitionHorizon],
    queryFn: () => getIgnitionGrid({ bbox, horizon: ignitionHorizon }),
    enabled: layersState.showIgnition,
    staleTime: 6 * 60 * 60 * 1000,  // 6h — matches server cache cadence
    placeholderData: (prev) => prev,
    retry: (failureCount, error) => {
      // Don't retry 503 — model is explicitly unavailable
      if (error instanceof ApiError && error.statusCode === 503) return false;
      return failureCount < 2;
    }
  });

  useEffect(() => {
    setIgnitionData(ignitionQuery.data ?? null);
  }, [ignitionQuery.data, setIgnitionData]);

  const normalizedEvents = useMemo(() => {
    const source = eventsQuery.data?.events || [];
    const renderable = source.filter((event) => !filters.activeOnly || isActiveCandidate(event));

    const searchFiltered = normalizedSearch
      ? renderable.filter((event) => eventSearchText(event).includes(normalizedSearch))
      : renderable;

    const confidenceFiltered =
      confidenceFilter === "High"
        ? searchFiltered.filter((event) => isHighConfidence(event))
        : searchFiltered;

    const mapped = confidenceFiltered
      .map(toRenderEvent)
      .filter((event): event is NonNullable<ReturnType<typeof toRenderEvent>> => Boolean(event));

    if (selectedEvent) {
      const fallback = toRenderEvent(selectedEvent);
      if (fallback) {
        const fallbackId = String(fallback.event_id || "");
        const alreadyIncluded = mapped.some((item) => String(item.event_id || "") === fallbackId);
        if (!alreadyIncluded) {
          return [fallback, ...mapped];
        }
      }
    }

    return mapped;
  }, [
    confidenceFilter,
    eventsQuery.data?.events,
    filters.activeOnly,
    normalizedSearch,
    selectedEvent
  ]);

  useEffect(() => {
    onVisibleEventsChange(normalizedEvents);
  }, [onVisibleEventsChange, normalizedEvents]);

  const visibleFronts = useMemo(() => {
    const source = frontsQuery.data?.fronts || [];
    return source.filter((front) => !filters.activeOnly || isActiveCandidate(front));
  }, [frontsQuery.data?.fronts, filters.activeOnly]);

  useEffect(() => {
    setFrontIndexByEvent(buildFrontIndexByEvent(visibleFronts));
  }, [visibleFronts, setFrontIndexByEvent]);

  const selectedEventId = selectedEvent?.event_id ? String(selectedEvent.event_id) : "";

  const selectedEventFeature = useMemo(() => {
    if (!selectedEvent) {
      return null;
    }
    const selectedRenderEvent = toRenderEvent(selectedEvent);
    if (!selectedRenderEvent) {
      return null;
    }
    return eventFeature({
      ...selectedRenderEvent,
      fill_r: SELECTED_EVENT_FILL[0],
      fill_g: SELECTED_EVENT_FILL[1],
      fill_b: SELECTED_EVENT_FILL[2],
      fill_a: SELECTED_EVENT_FILL[3],
      line_r: SELECTED_EVENT_STROKE[0],
      line_g: SELECTED_EVENT_STROKE[1],
      line_b: SELECTED_EVENT_STROKE[2],
      line_a: SELECTED_EVENT_STROKE[3]
    });
  }, [selectedEvent]);

  const selectedFrontFeatures = useMemo(() => {
    if (!selectedEventId) {
      return [];
    }
    return visibleFronts
      .filter((front) => front.event_id && String(front.event_id) === selectedEventId)
      .map(frontFeature)
      .filter((feature): feature is Feature => Boolean(feature));
  }, [selectedEventId, visibleFronts]);

  const layers = useMemo(() => {
    const deckLayers: Array<GeoJsonLayer | ScatterplotLayer | MVTLayer> = [];

    const markerPoints = filters.clusterPoints ? clusterEventPoints(normalizedEvents, mapView.zoom) : normalizedEvents;

    const polygonPoints = filters.clusterPoints
      ? [
          ...normalizedEvents.filter((point) => Boolean(point.geom_geojson)),
          ...clusterEventPoints(normalizedEvents.filter((point) => !point.geom_geojson), mapView.zoom)
        ]
      : normalizedEvents;

    if (layersState.showFires) {
      const fireFeatures = polygonPoints.map(eventFeature);

      deckLayers.push(
        new GeoJsonLayer({
          id: `events-${mapView.zoom.toFixed(2)}-${fireFeatures.length}`,
          data: {
            type: "FeatureCollection",
            features: fireFeatures
          },
          pickable: true,
          autoHighlight: true,
          filled: true,
          stroked: true,
          getLineWidth: 3,
          lineWidthMinPixels: 1,
          lineWidthMaxPixels: 4,
          getFillColor: (feature) => {
            const properties = feature.properties as Record<string, number>;
            return [properties.fill_r, properties.fill_g, properties.fill_b, properties.fill_a] as [number, number, number, number];
          },
          getLineColor: (feature) => {
            const properties = feature.properties as Record<string, number>;
            return [properties.line_r, properties.line_g, properties.line_b, properties.line_a] as [number, number, number, number];
          }
        })
      );

      if (selectedEventFeature) {
        deckLayers.push(
          new GeoJsonLayer({
            id: `selected-event-${String(selectedEventFeature.properties?.event_id || "event")}`,
            data: {
              type: "FeatureCollection",
              features: [selectedEventFeature]
            },
            pickable: false,
            filled: true,
            stroked: true,
            getFillColor: (feature) => {
              const properties = feature.properties as Record<string, number>;
              return [properties.fill_r, properties.fill_g, properties.fill_b, properties.fill_a] as [number, number, number, number];
            },
            getLineColor: (feature) => {
              const properties = feature.properties as Record<string, number>;
              return [properties.line_r, properties.line_g, properties.line_b, properties.line_a] as [number, number, number, number];
            },
            getLineWidth: 5,
            lineWidthMinPixels: 2,
            lineWidthMaxPixels: 8
          })
        );
      }

      if (markerPoints.length > 0 && shouldRenderCentroids(mapView.zoom)) {
        deckLayers.push(
          new ScatterplotLayer({
            id: `events-centroids-${mapView.zoom.toFixed(2)}-${markerPoints.length}`,
            data: markerPoints,
            pickable: true,
            autoHighlight: true,
            filled: true,
            stroked: true,
            getPosition: (d) => [d.lon, d.lat],
            getFillColor: (d) => [d.fill_r, d.fill_g, d.fill_b, 220],
            getLineColor: (d) => [d.line_r, d.line_g, d.line_b, 240],
            getRadius: 8,
            radiusUnits: "pixels",
            radiusMinPixels: 4,
            radiusMaxPixels: 16,
            lineWidthMinPixels: 1
          })
        );
      }
    }

    if (layersState.showFronts) {
      const frontFeatures = visibleFronts.map(frontFeature).filter((feature): feature is Feature => Boolean(feature));
      if (frontFeatures.length > 0) {
        deckLayers.push(
          new GeoJsonLayer({
            id: `fronts-${mapView.zoom.toFixed(2)}-${frontFeatures.length}`,
            data: {
              type: "FeatureCollection",
              features: frontFeatures
            },
            pickable: false,
            stroked: true,
            filled: false,
            getLineColor: (feature) => {
              const properties = feature.properties as Record<string, number>;
              return [properties.line_r, properties.line_g, properties.line_b, properties.line_a];
            },
            getLineWidth: (feature) => Number((feature.properties as Record<string, number>).line_width || 2),
            lineWidthMinPixels: 1,
            lineWidthMaxPixels: 6
          })
        );
      }

      if (selectedFrontFeatures.length > 0) {
        deckLayers.push(
          new GeoJsonLayer({
            id: `selected-fronts-${selectedEventId}-${selectedFrontFeatures.length}`,
            data: {
              type: "FeatureCollection",
              features: selectedFrontFeatures
            },
            pickable: false,
            stroked: true,
            filled: false,
            getLineColor: SELECTED_FRONT_COLOR,
            getLineWidth: (feature) => Math.max(4, Number((feature.properties as Record<string, number>).line_width || 2) + 2),
            lineWidthMinPixels: 2,
            lineWidthMaxPixels: 10
          })
        );
      }
    }

    if (layersState.showForecast) {
      const runId = forecast.lastForecast?.run.id;
      const contourUrl = `${apiPublicBaseUrl()}/tiles/forecast_contours/{z}/{x}/{y}.pbf${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`;

      deckLayers.push(
        new MVTLayer({
          id: "forecast-contours",
          data: contourUrl,
          pickable: false,
          getFillColor: (feature) =>
            isForecastContourVisible(
              feature.properties as { horizon_hours?: number; threshold?: number },
              FORECAST_DEFAULT_HORIZONS,
              FORECAST_DEFAULT_THRESHOLDS
            )
              ? (FORECAST_FILL as [number, number, number, number])
              : ([0, 0, 0, 0] as [number, number, number, number]),
          getLineColor: (feature) =>
            isForecastContourVisible(
              feature.properties as { horizon_hours?: number; threshold?: number },
              FORECAST_DEFAULT_HORIZONS,
              FORECAST_DEFAULT_THRESHOLDS
            )
              ? (FORECAST_STROKE as [number, number, number, number])
              : ([0, 0, 0, 0] as [number, number, number, number]),
          getLineWidth: 2,
          lineWidthMinPixels: 1
        })
      );
    }

    if (layersState.showRisk && riskQuery.data) {
      deckLayers.push(
        new GeoJsonLayer({
          id: `risk-${riskQuery.data.features.length}`,
          data: riskQuery.data,
          pickable: false,
          stroked: true,
          filled: true,
          getFillColor: (feature) => {
            const risk = Number((feature.properties as Record<string, unknown>).risk_score || 0);
            if (risk < RISK_THRESHOLDS.medium) return [34, 139, 34, 80];
            if (risk < RISK_THRESHOLDS.high) return [255, 215, 0, 100];
            return [220, 20, 60, 120];
          },
          getLineColor: (feature) => {
            const risk = Number((feature.properties as Record<string, unknown>).risk_score || 0);
            if (risk < RISK_THRESHOLDS.medium) return [34, 139, 34, 180];
            if (risk < RISK_THRESHOLDS.high) return [255, 215, 0, 180];
            return [220, 20, 60, 180];
          },
          lineWidthMinPixels: 1
        })
      );
    }

    if (layersState.showWarnings && warningsQuery.data) {
      deckLayers.push(
        new GeoJsonLayer({
          id: `weather-warnings-${warningsQuery.data.features.length}`,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          data: warningsQuery.data as any,
          pickable: false,
          stroked: true,
          filled: true,
          getFillColor: (feature) => {
            const sev = String((feature.properties as Record<string, unknown>).severity || "yellow");
            return warningSeverityColor(sev, sev === "yellow" ? 45 : 55);
          },
          getLineColor: (feature) => {
            const sev = String((feature.properties as Record<string, unknown>).severity || "yellow");
            return warningSeverityColor(sev, sev === "yellow" ? 180 : 200);
          },
          lineWidthMinPixels: 1.5,
        })
      );
    }

    // Ignition probability layer
    if (layersState.showIgnition && ignitionQuery.data) {
      const visibleCells = ignitionQuery.data.cells.filter((c) => c.level !== 'low');
      if (visibleCells.length > 0) {
        deckLayers.push(
          new ScatterplotLayer<IgnitionCell>({
            id: `ignition-${ignitionHorizon}-${visibleCells.length}`,
            data: visibleCells,
            pickable: false,
            filled: true,
            stroked: false,
            getPosition: (c) => [c.lon, c.lat],
            getFillColor: ignitionCellColor,
            getRadius: 10000,  // ~10 km radius per cell; distinct at zoom 5+
            radiusMinPixels: 3,
            radiusMaxPixels: 20
          })
        );
      }
    }

    // User location layers (safety mode)
    if (safety.enabled && safety.userLocation) {
      const locationLayers = buildUserLocationLayers(safety.userLocation, safety.proximityRadiusKm);
      deckLayers.push(...locationLayers);
    }

    return deckLayers;
  }, [
    filters.clusterPoints,
    forecast.lastForecast?.run.id,
    ignitionHorizon,
    ignitionQuery.data,
    layersState.showFires,
    layersState.showFronts,
    layersState.showForecast,
    layersState.showIgnition,
    layersState.showRisk,
    layersState.showWarnings,
    mapView.zoom,
    normalizedEvents,
    riskQuery.data,
    warningsQuery.data,
    safety.enabled,
    safety.userLocation,
    safety.proximityRadiusKm,
    selectedEventId,
    selectedEventFeature,
    selectedFrontFeatures,
    visibleFronts
  ]);

  const onClick = (info: PickingInfo): void => {
    const selected = normalizePickedEvent(info.object);
    if (!selected) {
      return;
    }

    const lat = Number(selected.lat);
    const lon = Number(selected.lon);
    setSelectedEvent(selected);
    setLastClick({ lat, lng: lon });
    const selectedBounds = geometryBounds(selected.geom_geojson);
    if (selectedBounds) {
      const next = selectionViewFromBounds(selectedBounds, {
        minZoom: MIN_SELECTION_ZOOM,
        maxZoom: MAX_SELECTION_ZOOM,
        targetOccupancy: SELECTION_TARGET_OCCUPANCY
      });
      setMapView({
        ...mapView,
        latitude: next.latitude,
        longitude: next.longitude,
        zoom: next.zoom,
        transitionDuration: 700
      });
      return;
    }
    focusMapOnPoint(lat, lon, MIN_SELECTION_ZOOM);
  };

  const tooltip = (info: PickingInfo): { html: string } | null => {
    const selected = normalizePickedEvent(info.object);
    if (!selected) {
      return null;
    }

    return {
      html: `
        <div style="font-family:Inter,sans-serif;padding:2px;">
          <div style="font-size:13px;font-weight:700;color:#f97316;margin-bottom:4px;">Fire Event</div>
          <div style="font-size:12px;color:#e5e7eb;line-height:1.45;">
            <b>Event ID:</b> ${String(selected.event_id || "unknown")}<br/>
            <b>Cluster events:</b> ${String(selected.cluster_event_count || 1)}<br/>
            <b>Window:</b> ${String(selected.start_time || "n/a")} → ${String(selected.end_time || "n/a")}<br/>
            <b>Sensor:</b> ${String(selected.sensor || "unknown")}<br/>
            <b>Detections:</b> ${String(selected.detection_count || 0)}<br/>
            <b>Event score:</b> ${String(selected.event_score || "n/a")}<br/>
            <b>Decision:</b> ${String(selected.denoiser_decision || "unknown")}<br/>
            <b>Review required:</b> ${String(Boolean(selected.review_required))}<br/>
            <b>Perimeter:</b> ${geometryProvenanceLabel(selected)}
          </div>
        </div>
      `
    };
  };

  return (
    <Box
      sx={{
        position: "relative",
        height: "100%",
        borderRadius: 3,
        overflow: "hidden",
        border: "1px solid rgba(255,255,255,0.06)",
        bgcolor: "#0a0c10",
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.03)"
      }}
    >
      {(eventsQuery.isError || frontsQuery.isError || riskQuery.isError) && (
        <Alert severity="warning" sx={{ position: "absolute", top: 12, left: 12, right: 12, zIndex: 20 }}>
          Live map data is partially unavailable; showing last successful snapshot where possible.
        </Alert>
      )}

      {layersState.showIgnition && ignitionQuery.isError && (
        <Alert severity="info" sx={{ position: "absolute", top: 12, left: 12, right: 260, zIndex: 20 }}>
          Ignition model unavailable — layer hidden.
        </Alert>
      )}

      {layersState.showIgnition && !ignitionQuery.isError && ignitionQuery.data?.coverage_warnings && ignitionQuery.data.coverage_warnings.length > 0 && (
        <Alert severity="warning" icon={false} sx={{ position: "absolute", bottom: 52, left: 12, right: 12, zIndex: 20, py: 0.5, fontSize: 11 }}>
          {ignitionQuery.data.coverage_warnings.join(' · ')}
        </Alert>
      )}

      <DeckGL
        layers={layers}
        viewState={mapView}
        controller
        onViewStateChange={({ viewState }) => {
          const next = viewState as {
            latitude: number;
            longitude: number;
            zoom: number;
            pitch: number;
            bearing: number;
          };
          setMapView({
            latitude: next.latitude,
            longitude: next.longitude,
            zoom: next.zoom,
            pitch: next.pitch,
            bearing: next.bearing
          });
        }}
        onClick={onClick}
        getTooltip={tooltip}
      >
        <Map
          mapLib={maplibregl}
          mapStyle={
            layersState.basemap === 'light' ? BASEMAP_LIGHT :
            layersState.basemap === 'satellite' ? BASEMAP_SATELLITE :
            BASEMAP_DARK
          }
          maxZoom={22}
          minZoom={1}
          reuseMaps
        />
      </DeckGL>

      <Box sx={{ position: "absolute", inset: 0, pointerEvents: "none", backgroundImage: "radial-gradient(circle, rgba(255,255,255,0.06) 1px, transparent 1px)", backgroundSize: "30px 30px", opacity: 0.18 }} />

      <Box sx={{ position: "absolute", top: 12, left: 12, px: 1.5, py: 1, bgcolor: "rgba(13,17,23,0.86)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 2, backdropFilter: "blur(8px)", zIndex: 11 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <PublicIcon sx={{ fontSize: 16, color: "rgba(255,255,255,0.45)" }} />
          <Box>
            <Typography sx={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.14em", textTransform: "uppercase", color: "#9ca3af" }}>
              Global Thermal View
            </Typography>
            <Typography sx={{ fontSize: 11, color: "#6b7280" }}>
              Interactive map layer active
            </Typography>
          </Box>
        </Box>
      </Box>

      <Box sx={{ position: "absolute", top: 12, right: 12, display: "flex", flexDirection: "column", gap: 1, zIndex: 11 }}>
        {safety.enabled && (
          <Tooltip title={safety.locationPermission === 'granted' ? "Location active" : "Find my location"}>
            <IconButton
              size="small"
              onClick={requestLocation}
              sx={{
                bgcolor: safety.userLocation ? "rgba(59,130,246,0.18)" : "#161b22",
                border: `1px solid ${safety.userLocation ? "rgba(59,130,246,0.5)" : "rgba(255,255,255,0.1)"}`,
                color: safety.userLocation ? "#60a5fa" : "#9ca3af",
                pointerEvents: "auto"
              }}
            >
              <GpsFixedIcon sx={{ fontSize: 18 }} />
            </IconButton>
          </Tooltip>
        )}
        <Tooltip title="Settings">
          <IconButton
            size="small"
            sx={{
              bgcolor: "#161b22",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "#9ca3af",
              pointerEvents: "auto"
            }}
          >
            <SettingsOutlinedIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Tooltip>
        <Tooltip title="Map layers">
          <IconButton
            size="small"
            onClick={(e) => setLayersPanelAnchor(e.currentTarget)}
            sx={{
              bgcolor: layersPanelAnchor ? "rgba(249,115,22,0.18)" : "#161b22",
              border: `1px solid ${layersPanelAnchor ? "rgba(249,115,22,0.5)" : "rgba(255,255,255,0.1)"}`,
              color: layersPanelAnchor ? "#f97316" : "#9ca3af",
              pointerEvents: "auto"
            }}
          >
            <MapOutlinedIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Tooltip>
      </Box>

      <Popover
        open={Boolean(layersPanelAnchor)}
        anchorEl={layersPanelAnchor}
        onClose={() => setLayersPanelAnchor(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        slotProps={{
          paper: {
            sx: {
              mt: 0.5,
              bgcolor: "rgba(13,17,23,0.96)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 2,
              backdropFilter: "blur(12px)",
              minWidth: 220,
              p: 1.5
            }
          }
        }}
      >
        <Typography sx={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "#6b7280", mb: 1 }}>
          Data layers
        </Typography>
        <FormControlLabel
          control={<Switch size="small" checked={layersState.showFires} onChange={(e) => setLayersState({ showFires: e.target.checked })} />}
          label={<Typography sx={{ fontSize: 13, color: "#d1d5db" }}>Fire Events</Typography>}
          sx={{ display: "flex", mx: 0, mb: 0.25 }}
        />
        <FormControlLabel
          control={<Switch size="small" checked={layersState.showFronts} onChange={(e) => setLayersState({ showFronts: e.target.checked })} />}
          label={<Typography sx={{ fontSize: 13, color: "#d1d5db" }}>Fire Fronts</Typography>}
          sx={{ display: "flex", mx: 0, mb: 0.25 }}
        />
        <FormControlLabel
          control={<Switch size="small" checked={layersState.showForecast} onChange={(e) => setLayersState({ showForecast: e.target.checked })} />}
          label={<Typography sx={{ fontSize: 13, color: "#d1d5db" }}>Forecast Overlay</Typography>}
          sx={{ display: "flex", mx: 0, mb: 0.25 }}
        />
        <FormControlLabel
          control={<Switch size="small" checked={layersState.showRisk} onChange={(e) => setLayersState({ showRisk: e.target.checked })} disabled={!filters.clusterPoints} />}
          label={<Typography sx={{ fontSize: 13, color: "#d1d5db" }}>Risk Index</Typography>}
          sx={{ display: "flex", mx: 0, mb: 0.25 }}
        />
        <FormControlLabel
          control={<Switch size="small" checked={layersState.showWarnings} onChange={(e) => setLayersState({ showWarnings: e.target.checked })} />}
          label={<Typography sx={{ fontSize: 13, color: "#d1d5db" }}>Weather Warnings <Typography component="span" sx={{ fontSize: 10, color: "#6b7280" }}>(Europe)</Typography></Typography>}
          sx={{ display: "flex", mx: 0, mb: 0.25 }}
        />
        <FormControlLabel
          control={<Switch size="small" checked={layersState.showIgnition} onChange={(e) => setLayersState({ showIgnition: e.target.checked })} />}
          label={<Typography sx={{ fontSize: 13, color: "#d1d5db" }}>Ignition Risk</Typography>}
          sx={{ display: "flex", mx: 0 }}
        />

        {layersState.showIgnition && (
          <Box sx={{ mt: 1, pl: 0.5 }}>
            <Typography sx={{ fontSize: 11, color: "#6b7280", mb: 0.5 }}>Horizon</Typography>
            <ToggleButtonGroup
              exclusive
              size="small"
              value={ignitionHorizon}
              onChange={(_, v) => { if (v) setIgnitionHorizon(v); }}
              sx={{ width: "100%" }}
            >
              {(['now', '+24h', '+48h'] as const).map((h) => (
                <ToggleButton
                  key={h}
                  value={h}
                  sx={{
                    flex: 1,
                    fontSize: 11,
                    fontWeight: 600,
                    color: "#9ca3af",
                    borderColor: "rgba(255,255,255,0.12)",
                    "&.Mui-selected": { bgcolor: "rgba(245,158,11,0.18)", color: "#f59e0b", borderColor: "rgba(245,158,11,0.4)" }
                  }}
                >
                  {h}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
            {ignitionHorizon === '+48h' && (
              <Typography sx={{ fontSize: 10, color: "#9ca3af", mt: 0.5, fontStyle: "italic" }}>
                Lower confidence — 48h forecast
              </Typography>
            )}
          </Box>
        )}

        {/* Ignition legend — only shown when layer is active */}
        {layersState.showIgnition && (
          <Box sx={{ mt: 1, p: 1, bgcolor: "rgba(0,0,0,0.2)", borderRadius: 1.5, border: "1px solid rgba(255,255,255,0.06)" }}>
            <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#6b7280", letterSpacing: "0.08em", textTransform: "uppercase", mb: 0.75 }}>
              Ignition Risk
            </Typography>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
              {[
                { label: "Critical", color: "#dc2680" },
                { label: "High",     color: "#f97316" },
                { label: "Elevated", color: "#f59e0b" },
              ].map(({ label, color }) => (
                <Box key={label} sx={{ display: "flex", alignItems: "center", gap: 0.7 }}>
                  <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: color }} />
                  <Typography sx={{ fontSize: 10, color: "#d1d5db" }}>{label}</Typography>
                </Box>
              ))}
            </Box>
          </Box>
        )}

        <Divider sx={{ my: 1.5, borderColor: "rgba(255,255,255,0.08)" }} />

        <Typography sx={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "#6b7280", mb: 1 }}>
          Basemap
        </Typography>
        <ToggleButtonGroup
          exclusive
          size="small"
          value={layersState.basemap}
          onChange={(_, value) => { if (value) setLayersState({ basemap: value }); }}
          sx={{ width: "100%" }}
        >
          {(["dark", "light", "satellite"] as const).map((bm) => (
            <ToggleButton
              key={bm}
              value={bm}
              sx={{
                flex: 1,
                fontSize: 11,
                fontWeight: 600,
                textTransform: "capitalize",
                color: "#9ca3af",
                borderColor: "rgba(255,255,255,0.12)",
                "&.Mui-selected": { bgcolor: "rgba(249,115,22,0.18)", color: "#f97316", borderColor: "rgba(249,115,22,0.4)" }
              }}
            >
              {bm}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Popover>

      {isArchiveMode && (
        <Box
          sx={{
            position: "absolute",
            top: 12,
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 20,
            pointerEvents: "auto"
          }}
        >
          <Button
            variant="contained"
            startIcon={<ReplayIcon />}
            onClick={exitToLiveMode}
            sx={{
              bgcolor: "#f97316",
              color: "#fff",
              fontWeight: 800,
              fontSize: 11,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              px: 2.5,
              py: 1.2,
              borderRadius: 2,
              boxShadow: "0 4px 20px rgba(249,115,22,0.4)",
              "&:hover": { bgcolor: "#ea6f10" }
            }}
          >
            Return to Live Feed
          </Button>
        </Box>
      )}

      <Box
        sx={{
          position: "absolute",
          left: 12,
          bottom: 12,
          px: 1.5,
          py: 1.2,
          bgcolor: "rgba(13,17,23,0.88)",
          border: "1px solid rgba(255,255,255,0.12)",
          borderRadius: 2,
          backdropFilter: "blur(8px)",
          zIndex: 11
        }}
      >
        <Typography sx={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.12em", textTransform: "uppercase", color: "#6b7280", mb: 0.8 }}>
          Confidence Legend
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.75 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.7 }}>
            <Box sx={{ width: 10, height: 10, borderRadius: "50%", bgcolor: "#f97316", boxShadow: "0 0 8px rgba(249,115,22,0.55)" }} />
            <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#d1d5db" }}>High</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.7 }}>
            <Box sx={{ width: 10, height: 10, borderRadius: "50%", bgcolor: "rgba(251,146,60,0.35)" }} />
            <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#d1d5db" }}>Nominal</Typography>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
