import { useEffect, useMemo } from "react";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer, ScatterplotLayer } from "@deck.gl/layers";
import { MVTLayer } from "@deck.gl/geo-layers";
import type { PickingInfo } from "@deck.gl/core";
import Map from "react-map-gl/maplibre";
import maplibregl from "maplibre-gl";
import { Alert, Box } from "@mui/material";
import { normalizePickedEvent } from "../utils/selection";
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
  isActiveCandidate,
  isForecastContourVisible,
  toRenderEvent
} from "../map/layerUtils";
import { computeTimeRange } from "../utils/time";
import { eventLimitForZoom, frontLimitForZoom, shouldLoadFronts, shouldRenderCentroids, viewportBbox } from "../utils/mapMath";

const BASEMAP_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const FORECAST_FILL = [255, 165, 0, 40];
const FORECAST_STROKE = [255, 165, 0, 200];

interface FireMapProps {
  onVisibleEventsChange: (events: FireEvent[]) => void;
}

export default function FireMap({ onVisibleEventsChange }: FireMapProps) {
  const filters = useAppStore((s) => s.filters);
  const layersState = useAppStore((s) => s.layers);
  const mapView = useAppStore((s) => s.mapView);
  const forecast = useAppStore((s) => s.forecast);
  const selectedEvent = useAppStore((s) => s.selectedEvent);
  const setMapView = useAppStore((s) => s.setMapView);
  const setSelectedEvent = useAppStore((s) => s.setSelectedEvent);
  const setLastClick = useAppStore((s) => s.setLastClick);
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);
  const setFrontIndexByEvent = useAppStore((s) => s.setFrontIndexByEvent);

  const bbox = useMemo(() => viewportBbox(mapView), [mapView]);
  const timeRange = useMemo(() => computeTimeRange(filters), [filters]);

  const eventsQuery = useQuery({
    queryKey: ["fire-events", bbox, timeRange.startTime.toISOString(), timeRange.endTime.toISOString(), filters.minLikelihood, filters.activeOnly, filters.clusterPoints, mapView.zoom],
    queryFn: () =>
      getFireEvents({
        bbox,
        startTime: timeRange.startTime,
        endTime: timeRange.endTime,
        minEventScore: filters.minLikelihood,
        includeReviewRequired: true,
        limit: eventLimitForZoom(mapView.zoom)
      }),
    placeholderData: (prev) => prev
  });

  const frontsQuery = useQuery({
    queryKey: ["fire-fronts", bbox, timeRange.startTime.toISOString(), timeRange.endTime.toISOString(), filters.minLikelihood, filters.activeOnly, mapView.zoom],
    queryFn: () =>
      getFireFronts({
        bbox,
        startTime: timeRange.startTime,
        endTime: timeRange.endTime,
        minEventScore: filters.minLikelihood,
        includeReviewRequired: true,
        limit: frontLimitForZoom(mapView.zoom)
      }),
    enabled: shouldLoadFronts(mapView.zoom),
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
    const mapped = renderable.map(toRenderEvent).filter((event): event is NonNullable<ReturnType<typeof toRenderEvent>> => Boolean(event));

    if (mapped.length === 0 && selectedEvent) {
      const fallback = toRenderEvent(selectedEvent);
      if (fallback) {
        return [fallback];
      }
    }

    return mapped;
  }, [eventsQuery.data?.events, filters.activeOnly, selectedEvent]);

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

  const layers = useMemo(() => {
    const deckLayers: Array<GeoJsonLayer | ScatterplotLayer | MVTLayer> = [];

    const markerPoints = filters.clusterPoints ? clusterEventPoints(normalizedEvents, mapView.zoom) : normalizedEvents;

    const polygonPoints = filters.clusterPoints
      ? [
          ...normalizedEvents.filter((point) => Boolean(point.geom_geojson)),
          ...clusterEventPoints(normalizedEvents.filter((point) => !point.geom_geojson), mapView.zoom)
        ]
      : normalizedEvents;

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
          getRadius: 5,
          radiusUnits: "pixels",
          radiusMinPixels: 3,
          radiusMaxPixels: 8,
          lineWidthMinPixels: 1
        })
      );
    }

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

    return deckLayers;
  }, [
    filters.clusterPoints,
    forecast.lastForecast?.run.id,
    layersState.showRisk,
    mapView.zoom,
    normalizedEvents,
    riskQuery.data,
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
    focusMapOnPoint(lat, lon, 6);
  };

  const tooltip = (info: PickingInfo): { html: string } | null => {
    const selected = normalizePickedEvent(info.object);
    if (!selected) {
      return null;
    }

    return {
      html: `
        <div style="font-family:Inter,sans-serif;padding:2px;">
          <div style="font-size:13px;font-weight:600;color:#ff6b35;margin-bottom:4px;">Fire Event</div>
          <div style="font-size:12px;color:#e0e0e0;">
            <b>Event ID:</b> ${String(selected.event_id || "unknown")}<br/>
            <b>Cluster events:</b> ${String(selected.cluster_event_count || 1)}<br/>
            <b>Window:</b> ${String(selected.start_time || "n/a")} → ${String(selected.end_time || "n/a")}<br/>
            <b>Sensor:</b> ${String(selected.sensor || "unknown")}<br/>
            <b>Detections:</b> ${String(selected.detection_count || 0)}<br/>
            <b>Event score:</b> ${String(selected.event_score || "n/a")}<br/>
            <b>Decision:</b> ${String(selected.denoiser_decision || "unknown")}<br/>
            <b>Review required:</b> ${String(Boolean(selected.review_required))}
          </div>
        </div>
      `
    };
  };

  return (
    <Box sx={{ position: "relative", height: "100%", borderRadius: 1, overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)" }}>
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
          mapStyle={BASEMAP_DARK}
          maxZoom={22}
          minZoom={1}
          reuseMaps
        />
      </DeckGL>
    </Box>
  );
}
