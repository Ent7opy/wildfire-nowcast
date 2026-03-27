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
import { normalizePickedEvent } from "../utils/selection";
import { buildUserLocationLayers } from "../utils/userLocationLayers";
import { useGeolocation } from "../hooks/useGeolocation";
import { useQuery } from "@tanstack/react-query";
import type { Feature } from "geojson";

import { apiPublicBaseUrl } from "../config/runtime";
import { getFireEvents, getFireFronts, getRiskGrid } from "../api/client";
import { useAppStore } from "../state/store";
import type { FireEvent } from "../types/api";
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
} from "../map/layerUtils";
import { computeArchiveTimeRange, computeFullDayTimeRange, computeTimeRange } from "../utils/time";
import { eventLimitForZoom, frontLimitForZoom, shouldLoadFronts, shouldRenderCentroids, viewportBbox } from "../utils/mapMath";
import { useDebounce } from "../hooks/useDebounce";
import { selectionViewFromBounds } from "../utils/mapSelection";

const BASEMAP_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const BASEMAP_LIGHT = "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json";
const BASEMAP_SATELLITE = {
  version: 8 as const,
  sources: {
    "esri-satellite": {
      type: "raster" as const,
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      tileSize: 256,
      attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
    }
  },
  layers: [{ id: "esri-satellite", type: "raster" as const, source: "esri-satellite" }]
};
const FORECAST_FILL = [255, 165, 0, 40];
const FORECAST_STROKE = [255, 165, 0, 200];
const HIGH_CONFIDENCE_THRESHOLD = 0.6;
const MIN_SELECTION_ZOOM = 6;
const MAX_SELECTION_ZOOM = 14;
const SELECTION_TARGET_OCCUPANCY = 0.3;
const SELECTED_FRONT_COLOR: [number, number, number, number] = [59, 130, 246, 255];
const SELECTED_EVENT_FILL: [number, number, number, number] = [59, 130, 246, 88];
const SELECTED_EVENT_STROKE: [number, number, number, number] = [96, 165, 250, 255];

type ConfidenceFilter = "All" | "High";

interface FireMapProps {
  onVisibleEventsChange: (events: FireEvent[]) => void;
  searchQuery?: string;
  confidenceFilter?: ConfidenceFilter;
}

function toFiniteNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
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

function geometryProvenanceLabel(event: FireEvent): string {
  return String(event.geom_source || "").toLowerCase() === "authoritative"
    ? "Authoritative perimeter"
    : "Estimated perimeter";
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
  const { requestLocation } = useGeolocation();
  const setMapView = useAppStore((s) => s.setMapView);
  const setSelectedEvent = useAppStore((s) => s.setSelectedEvent);
  const setLastClick = useAppStore((s) => s.setLastClick);
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);
  const setFrontIndexByEvent = useAppStore((s) => s.setFrontIndexByEvent);
  const setLayersState = useAppStore((s) => s.setLayersState);
  const exitToLiveMode = useAppStore((s) => s.exitToLiveMode);

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
    queryKey: ["fire-events", bbox, timeRange.startTime.toISOString(), timeRange.endTime.toISOString(), filters.minLikelihood, filters.activeOnly, filters.clusterPoints, debouncedMapView.zoom, archive.viewMode, archive.archiveDate, archive.archiveTimeframe],
    queryFn: () =>
      getFireEvents({
        bbox,
        startTime: timeRange.startTime,
        endTime: timeRange.endTime,
        minEventScore: filters.minLikelihood,
        includeReviewRequired: true,
        limit: eventLimitForZoom(debouncedMapView.zoom)
      }),
    placeholderData: (prev) => prev
  });

  const frontsQuery = useQuery({
    queryKey: ["fire-fronts", bbox, timeRange.startTime.toISOString(), timeRange.endTime.toISOString(), filters.minLikelihood, filters.activeOnly, debouncedMapView.zoom, archive.viewMode, archive.archiveDate, archive.archiveTimeframe],
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
    placeholderData: (prev) => prev
  });

  const riskQuery = useQuery({
    queryKey: ["risk-grid", bbox, layersState.showRisk],
    queryFn: () => getRiskGrid({ bbox }),
    enabled: layersState.showRisk,
    placeholderData: (prev) => prev
  });

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

  const visibleEvents = useMemo(() => {
    return normalizedEvents.map((event) => ({ ...event }));
  }, [normalizedEvents]);

  useEffect(() => {
    onVisibleEventsChange(visibleEvents);
  }, [onVisibleEventsChange, visibleEvents]);

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

    // User location layers (safety mode)
    if (safety.enabled && safety.userLocation) {
      const locationLayers = buildUserLocationLayers(safety.userLocation, safety.proximityRadiusKm);
      deckLayers.push(...locationLayers);
    }

    return deckLayers;
  }, [
    filters.clusterPoints,
    forecast.lastForecast?.run.id,
    layersState.showFires,
    layersState.showFronts,
    layersState.showForecast,
    layersState.showRisk,
    mapView.zoom,
    normalizedEvents,
    riskQuery.data,
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
          sx={{ display: "flex", mx: 0 }}
        />

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
