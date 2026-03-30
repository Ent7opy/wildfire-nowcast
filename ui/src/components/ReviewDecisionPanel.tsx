import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import Map, { Marker, Source, Layer } from "react-map-gl/maplibre";
import type { CircleLayerSpecification, LineLayerSpecification } from "maplibre-gl";
import { Box, Typography, Button, CircularProgress, Alert } from "@mui/material";
import MapIcon from "@mui/icons-material/Map";
import { getReviewEventDetail } from "../api/review";
import { windCompassLabel } from "./fire-details/WeatherBlock";
import { BASEMAP_SATELLITE } from "./map/layers/layerConfig";
import { eventRingCoords } from "../map/layerUtils";
import { useAppStore } from "../state/store";

const RADIUS_SOURCE_ID = "review-radius-circle";
const POINT_SOURCE_ID = "review-detection-point";

const CIRCLE_LINE_LAYER: LineLayerSpecification = {
  id: "review-radius-line",
  type: "line",
  source: RADIUS_SOURCE_ID,
  paint: { "line-color": "#60a5fa", "line-width": 1.5, "line-opacity": 0.6, "line-dasharray": [4, 3] },
};

const CIRCLE_FILL_LAYER: CircleLayerSpecification = {
  id: "review-detection-point",
  type: "circle",
  source: POINT_SOURCE_ID,
  paint: {
    "circle-radius": 7,
    "circle-color": "#ef4444",
    "circle-stroke-width": 2,
    "circle-stroke-color": "#ffffff",
  },
};

function WindArrow({ directionDeg }: { directionDeg: number }): JSX.Element {
  return (
    <svg
      width={28}
      height={28}
      viewBox="0 0 28 28"
      style={{ transform: `rotate(${directionDeg}deg)`, display: "block" }}
    >
      <line x1="14" y1="22" x2="14" y2="6" stroke="#60a5fa" strokeWidth={2.5} strokeLinecap="round" />
      <polyline points="9,11 14,5 19,11" fill="none" stroke="#60a5fa" strokeWidth={2.5} strokeLinejoin="round" />
    </svg>
  );
}

interface ReviewDecisionPanelProps {
  eventId: string;
  borderColor: string;
  onViewOnMap: () => void;
}

export function ReviewDecisionPanel({
  eventId,
  borderColor,
  onViewOnMap,
}: ReviewDecisionPanelProps): JSX.Element {
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["review-detail", eventId],
    queryFn: () => getReviewEventDetail(eventId),
    staleTime: 60_000,
  });

  const handleViewOnMap = (): void => {
    if (data) {
      focusMapOnPoint(data.centroid_lat, data.centroid_lon, 12);
    }
    onViewOnMap();
  };

  if (isLoading) {
    return (
      <Box sx={{ p: 1.5, display: "flex", alignItems: "center", gap: 1 }}>
        <CircularProgress size={14} sx={{ color: "#6b7280" }} />
        <Typography sx={{ fontSize: 11, color: "#6b7280" }}>Loading decision context…</Typography>
      </Box>
    );
  }

  if (isError || !data) {
    return (
      <Box sx={{ px: 1.5, pb: 1.5 }}>
        <Alert severity="error" sx={{ fontSize: 11, py: 0.4 }}>
          Could not load decision context. Action buttons above still work.
        </Alert>
      </Box>
    );
  }

  return <DecisionPanelContent data={data} borderColor={borderColor} onViewOnMap={handleViewOnMap} />;
}

// Split out so hooks (useMemo) run after data guard
function DecisionPanelContent({
  data,
  borderColor,
  onViewOnMap,
}: {
  data: NonNullable<Awaited<ReturnType<typeof getReviewEventDetail>>>;
  borderColor: string;
  onViewOnMap: () => void;
}): JSX.Element {
  const circleGeoJson = useMemo<GeoJSON.Feature<GeoJSON.Polygon>>(
    () => ({
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [eventRingCoords(data.centroid_lon, data.centroid_lat, 10_000)],
      },
      properties: {},
    }),
    [data.centroid_lat, data.centroid_lon],
  );

  const pointGeoJson = useMemo<GeoJSON.FeatureCollection>(
    () => ({
      type: "FeatureCollection",
      features: [{
        type: "Feature",
        geometry: { type: "Point", coordinates: [data.centroid_lon, data.centroid_lat] },
        properties: {},
      }],
    }),
    [data.centroid_lat, data.centroid_lon],
  );

  let weatherLine: string | null = null;
  if (
    data.wind_speed_kmh != null &&
    data.wind_direction_deg != null &&
    data.relative_humidity_pct != null &&
    data.temperature_c != null
  ) {
    const compass = windCompassLabel(data.wind_direction_deg);
    weatherLine = `Wind: ${data.wind_speed_kmh.toFixed(0)} km/h ${compass} · RH: ${data.relative_humidity_pct.toFixed(0)}% · Temp: ${data.temperature_c.toFixed(1)}°C`;
  }

  let nearbyText: string;
  if (data.nearby_fires_count === 0) {
    nearbyText = "No confirmed fires within 100 km in the last 48 hours.";
  } else {
    const parts: string[] = [`${data.nearby_fires_count} active fire${data.nearby_fires_count !== 1 ? "s" : ""} within 100 km`];
    if (data.nearby_fires_max_frp_mw != null) {
      parts.push(`largest: ${data.nearby_fires_max_frp_mw.toFixed(0)} MW`);
    }
    if (data.nearby_fires_nearest_km != null) {
      parts.push(`nearest: ${data.nearby_fires_nearest_km.toFixed(0)} km`);
    }
    nearbyText = parts.join(" — ") + ".";
  }

  let historyText: string;
  if (data.location_history_flagged === 0) {
    historyText = "First time this location has been flagged.";
  } else {
    const parts: string[] = [`flagged ${data.location_history_flagged}× in past 30 days`];
    const outcomes: string[] = [];
    if (data.location_history_confirmed > 0) {
      outcomes.push(`${data.location_history_confirmed} confirmed fire${data.location_history_confirmed !== 1 ? "s" : ""}`);
    }
    if (data.location_history_noise > 0) {
      outcomes.push(`${data.location_history_noise} marked noise`);
    }
    if (outcomes.length > 0) parts.push(outcomes.join(", "));
    historyText = "This location: " + parts.join(" — ") + ".";
  }

  return (
    <Box
      sx={{
        borderLeft: `3px solid ${borderColor}`,
        ml: 1,
        mr: 0.5,
        mb: 1.5,
        pl: 1.2,
        display: "flex",
        flexDirection: "column",
        gap: 1.2,
      }}
    >
      <Typography sx={{ fontSize: 11.5, color: "#d1d5db", lineHeight: 1.55, fontStyle: "italic" }}>
        {data.reason_summary}
      </Typography>

      <Box sx={{ position: "relative", width: 300, height: 200, borderRadius: 1.5, overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)", flexShrink: 0 }}>
        <Map
          initialViewState={{ latitude: data.centroid_lat, longitude: data.centroid_lon, zoom: 10 }}
          style={{ width: "100%", height: "100%" }}
          mapStyle={BASEMAP_SATELLITE}
          interactive={false}
          attributionControl={false}
        >
          <Source id={RADIUS_SOURCE_ID} type="geojson" data={circleGeoJson}>
            <Layer {...CIRCLE_LINE_LAYER} />
          </Source>
          <Source id={POINT_SOURCE_ID} type="geojson" data={pointGeoJson}>
            <Layer {...CIRCLE_FILL_LAYER} />
          </Source>
          {data.wind_direction_deg != null && (
            <Marker latitude={data.centroid_lat} longitude={data.centroid_lon} anchor="center" offset={[30, -30]}>
              <WindArrow directionDeg={data.wind_direction_deg} />
            </Marker>
          )}
        </Map>
        <Button
          size="small"
          startIcon={<MapIcon sx={{ fontSize: 13 }} />}
          onClick={onViewOnMap}
          sx={{
            position: "absolute",
            bottom: 6,
            right: 6,
            fontSize: 10,
            py: 0.3,
            px: 0.8,
            bgcolor: "rgba(0,0,0,0.72)",
            color: "#e5e7eb",
            border: "1px solid rgba(255,255,255,0.15)",
            backdropFilter: "blur(4px)",
            textTransform: "none",
            lineHeight: 1.4,
            "&:hover": { bgcolor: "rgba(0,0,0,0.85)" },
          }}
        >
          View on map
        </Button>
      </Box>

      <Box sx={{ display: "flex", alignItems: "center", gap: 0.8 }}>
        <Typography sx={{ fontSize: 10, color: "#4b5563", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Weather
        </Typography>
        <Typography sx={{ fontSize: 11, color: "#9ca3af" }}>
          {weatherLine ?? "Weather data not available for this location."}
        </Typography>
      </Box>

      <Box sx={{ display: "flex", alignItems: "baseline", gap: 0.8 }}>
        <Typography sx={{ fontSize: 10, color: "#4b5563", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", whiteSpace: "nowrap" }}>
          Nearby
        </Typography>
        <Typography sx={{ fontSize: 11, color: "#9ca3af" }}>{nearbyText}</Typography>
      </Box>

      <Box sx={{ display: "flex", alignItems: "baseline", gap: 0.8 }}>
        <Typography sx={{ fontSize: 10, color: "#4b5563", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", whiteSpace: "nowrap" }}>
          History
        </Typography>
        <Typography sx={{ fontSize: 11, color: data.location_history_confirmed > 0 ? "#fca5a5" : "#9ca3af" }}>
          {historyText}
        </Typography>
      </Box>
    </Box>
  );
}
