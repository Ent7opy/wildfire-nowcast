import { useEffect, useMemo, useState } from "react";
import {
  Box,
  Button,
  CircularProgress,
  Paper,
  TextField,
  Tooltip,
  Typography
} from "@mui/material";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import Brightness3Icon from "@mui/icons-material/Brightness3";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import PublicIcon from "@mui/icons-material/Public";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import ShowChartIcon from "@mui/icons-material/ShowChart";
import SunriseIcon from "@mui/icons-material/WbTwilight";
import SunsetIcon from "@mui/icons-material/Nightlight";
import SunIcon from "@mui/icons-material/WbSunny";
import VerifiedIcon from "@mui/icons-material/Verified";

import AIChatAssistant from "./components/AIChatAssistant";
import DataFreshnessBanner from "./components/DataFreshnessBanner";
import FireDetailsPanel from "./components/FireDetailsPanel";
import FireMap from "./components/FireMap";
import ForecastNotification from "./components/ForecastNotification";
import RegionFilter from "./components/RegionFilter";
import SafetyStatusBar from "./components/SafetyStatusBar";
import { useArchiveData } from "./hooks/useArchiveData";
import { useForecastPolling } from "./hooks/useForecastPolling";
import { useGeolocation } from "./hooks/useGeolocation";
import { useSafetyMetrics } from "./hooks/useSafetyMetrics";
import { useAppStore } from "./state/store";
import type { FireEvent } from "./types/api";
import type {
  ArchiveTimeframe,
  AssistantConfidenceFilter,
  AssistantViewEventSummary
} from "./types/state";
import {
  CONTINENT_VIEWPORTS,
  EMPTY_REGION_FILTER,
  formatRegionFilter,
  matchesRegionFilter,
  type RegionFilterValue,
} from "./utils/continents";
import { eventsWithinRadius } from "./utils/geo";
import { TIMEFRAME_DEFS } from "./utils/time";

const HIGH_CONFIDENCE_THRESHOLD = 0.6;

function toFiniteNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function locationLabel(event: FireEvent): string {
  const candidates = [
    event.location_name,
    event.region_name,
    event.admin1_name,
    event.admin0_name,
    event.country
  ];

  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim();
    }
  }

  const lat = toFiniteNumber(event.lat);
  const lon = toFiniteNumber(event.lon);
  if (lat !== null && lon !== null) {
    return `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
  }

  return "Unknown region";
}

function isHighConfidence(event: FireEvent): boolean {
  const score = toFiniteNumber(event.event_score);
  return score !== null && score >= HIGH_CONFIDENCE_THRESHOLD;
}


function formatStatValue(value: number, digits = 0): string {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
}

function toTextOrNull(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function toEventSummary(event: FireEvent): AssistantViewEventSummary {
  const score = toFiniteNumber(event.event_score);
  return {
    eventId: String(event.event_id || "unknown"),
    locationLabel: locationLabel(event),
    lat: toFiniteNumber(event.lat),
    lon: toFiniteNumber(event.lon),
    eventScore: score,
    detectionCount: Number(toFiniteNumber(event.detection_count) || 0),
    frontCount: Number(toFiniteNumber(event.front_count) || 0),
    endTime: toTextOrNull(event.end_time),
    sensor: toTextOrNull(event.sensor),
    source: toTextOrNull(event.source),
    reviewRequired: Boolean(event.review_required),
    denoiserDecision: toTextOrNull(event.denoiser_decision)
  };
}

const TIMEFRAME_ICONS: Record<ArchiveTimeframe, React.ElementType> = {
  morning: SunriseIcon,
  afternoon: SunIcon,
  evening: SunsetIcon,
  night: Brightness3Icon,
};

export default function App(): JSX.Element {
  const [visibleEvents, setVisibleEvents] = useState<FireEvent[]>([]);
  const [regionFilter, setRegionFilter] = useState<RegionFilterValue>(EMPTY_REGION_FILTER);
  const [confidenceFilter, setConfidenceFilter] = useState<AssistantConfidenceFilter>("All");
  const setAssistantViewContext = useAppStore((s) => s.setAssistantViewContext);
  const archive = useAppStore((s) => s.archive);
  const enterArchiveMode = useAppStore((s) => s.enterArchiveMode);
  const exitToLiveMode = useAppStore((s) => s.exitToLiveMode);
  const setArchiveDate = useAppStore((s) => s.setArchiveDate);
  const setArchiveTimeframe = useAppStore((s) => s.setArchiveTimeframe);
  const safety = useAppStore((s) => s.safety);
  const enableSafetyMode = useAppStore((s) => s.enableSafetyMode);
  const disableSafetyMode = useAppStore((s) => s.disableSafetyMode);
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);
  const setMapView = useAppStore((s) => s.setMapView);
  const mapView = useAppStore((s) => s.mapView);

  const isArchiveMode = archive.viewMode === "archive";
  const isSafetyMode = safety.enabled;
  const archiveData = useArchiveData();
  const { requestLocation } = useGeolocation();

  useForecastPolling();
  useSafetyMetrics(visibleEvents);

  const filteredEvents = useMemo(() => {
    return visibleEvents.filter((event) => {
      if (confidenceFilter === "High" && !isHighConfidence(event)) {
        return false;
      }
      return matchesRegionFilter(event, regionFilter);
    });
  }, [confidenceFilter, regionFilter, visibleEvents]);

  const totalDetections = useMemo(() => {
    const aggregate = filteredEvents.reduce((sum, event) => sum + (toFiniteNumber(event.detection_count) || 0), 0);
    return aggregate > 0 ? aggregate : filteredEvents.length;
  }, [filteredEvents]);

  const activePerimeters = useMemo(() => {
    return filteredEvents.reduce((sum, event) => sum + (toFiniteNumber(event.front_count) || 0), 0);
  }, [filteredEvents]);

  const averageScore = useMemo(() => {
    const scores = filteredEvents
      .map((event) => toFiniteNumber(event.event_score))
      .filter((score): score is number => score !== null);

    if (scores.length === 0) return null;
    const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    return mean;
  }, [filteredEvents]);

  const confidencePercent = useMemo(() => {
    if (filteredEvents.length === 0) return null;
    const highCount = filteredEvents.filter((event) => isHighConfidence(event)).length;
    return (highCount / filteredEvents.length) * 100;
  }, [filteredEvents]);

  const topEventsForAssistant = useMemo(() => {
    return [...filteredEvents]
      .sort((a, b) => {
        const scoreDiff = (toFiniteNumber(b.event_score) || 0) - (toFiniteNumber(a.event_score) || 0);
        if (scoreDiff !== 0) return scoreDiff;
        return (toFiniteNumber(b.detection_count) || 0) - (toFiniteNumber(a.detection_count) || 0);
      })
      .slice(0, 20)
      .map(toEventSummary);
  }, [filteredEvents]);

  useEffect(() => {
    setAssistantViewContext({
      updatedAt: Date.now(),
      searchQuery: formatRegionFilter(regionFilter) ?? "",
      confidenceFilter,
      visibleEventCount: visibleEvents.length,
      filteredEventCount: filteredEvents.length,
      topEvents: topEventsForAssistant
    });
  }, [
    confidenceFilter,
    filteredEvents.length,
    regionFilter,
    setAssistantViewContext,
    topEventsForAssistant,
    visibleEvents.length
  ]);

  // Clear region filter when entering archive mode
  useEffect(() => {
    if (isArchiveMode) {
      setRegionFilter(EMPTY_REGION_FILTER);
    }
  }, [isArchiveMode]);

  // Auto-pan to user location when safety mode is enabled and GPS acquired
  useEffect(() => {
    if (isSafetyMode && safety.userLocation) {
      focusMapOnPoint(safety.userLocation.lat, safety.userLocation.lon, 7);
    }
  }, [isSafetyMode, safety.userLocation?.lat, safety.userLocation?.lon, focusMapOnPoint]);

  // Auto-request location when safety mode is first enabled
  useEffect(() => {
    if (isSafetyMode && safety.locationPermission === 'unknown') {
      requestLocation();
    }
  }, [isSafetyMode, safety.locationPermission, requestLocation]);

  // Fly to selected region when the region filter changes
  useEffect(() => {
    const { continent, country, admin1 } = regionFilter;
    if (!continent && !country && !admin1) return;

    // Try to centre on the actual events in the region first
    const matching = visibleEvents.filter((e) => matchesRegionFilter(e, regionFilter));
    const targetZoom = admin1 ? 6 : country ? 5 : 3;
    if (matching.length > 0) {
      const avgLat = matching.reduce((s, e) => s + (e.lat ?? 0), 0) / matching.length;
      const avgLon = matching.reduce((s, e) => s + (e.lon ?? 0), 0) / matching.length;
      setMapView({ ...mapView, latitude: avgLat, longitude: avgLon, zoom: targetZoom, transitionDuration: 700 });
    } else if (continent && CONTINENT_VIEWPORTS[continent]) {
      const vp = CONTINENT_VIEWPORTS[continent];
      setMapView({ ...mapView, latitude: vp.lat, longitude: vp.lon, zoom: vp.zoom, transitionDuration: 700 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [regionFilter]);

  const nearbyEvents = useMemo(() => {
    if (!isSafetyMode || !safety.userLocation) return [];
    return eventsWithinRadius(safety.userLocation.lat, safety.userLocation.lon, safety.proximityRadiusKm, filteredEvents);
  }, [isSafetyMode, safety.userLocation, safety.proximityRadiusKm, filteredEvents]);

  const maxIntensityFrp = useMemo(() => {
    if (filteredEvents.length === 0) return null;
    const frpValues = filteredEvents
      .map((e) => toFiniteNumber(e.frp_max))
      .filter((v): v is number => v !== null);
    return frpValues.length > 0 ? Math.max(...frpValues) : null;
  }, [filteredEvents]);

  const activeTimeframeDef = isArchiveMode
    ? TIMEFRAME_DEFS.find((d) => d.id === archive.archiveTimeframe) ?? TIMEFRAME_DEFS[1]
    : null;

  type StatCard = { label: string; value: string; unit: string; icon: React.ElementType; color: string };

  const liveStats: StatCard[] = [
    { label: "Total Detections", value: formatStatValue(totalDetections), unit: "", icon: QueryStatsIcon, color: "#f97316" },
    { label: "Active Perimeters", value: formatStatValue(activePerimeters), unit: "", icon: LocalFireDepartmentIcon, color: "#ef4444" },
    { label: "Avg Event Score", value: averageScore === null ? "n/a" : formatStatValue(averageScore, 3), unit: "0-1", icon: PublicIcon, color: "#60a5fa" },
    { label: "High Confidence", value: confidencePercent === null ? "n/a" : formatStatValue(confidencePercent, 1), unit: "%", icon: VerifiedIcon, color: "#4ade80" }
  ];

  const archiveStats: StatCard[] = [
    { label: "Time Filter", value: activeTimeframeDef?.label ?? "—", unit: "", icon: activeTimeframeDef ? TIMEFRAME_ICONS[activeTimeframeDef.id] : AccessTimeIcon, color: "#60a5fa" },
    { label: "Active Events", value: formatStatValue(filteredEvents.length), unit: "", icon: ShowChartIcon, color: "#f97316" },
    { label: "Max Intensity", value: maxIntensityFrp === null ? "n/a" : formatStatValue(maxIntensityFrp), unit: "MW", icon: LocalFireDepartmentIcon, color: "#ef4444" },
    { label: "Context", value: "ARCHIVE", unit: "", icon: PublicIcon, color: "#4ade80" }
  ];

  const SAFETY_TIER_COLORS: Record<string, string> = {
    SAFE: "#22c55e", WATCH: "#eab308", WARNING: "#f97316", DANGER: "#ef4444"
  };
  const tierColor = SAFETY_TIER_COLORS[safety.safetyTier] ?? "#6b7280";

  const safetyStats: StatCard[] = [
    {
      label: "Nearest Fire",
      value: safety.nearestFireDistanceKm !== null ? formatStatValue(safety.nearestFireDistanceKm, 1) : "—",
      unit: "km",
      icon: LocalFireDepartmentIcon,
      color: tierColor
    },
    {
      label: "Fires in Radius",
      value: formatStatValue(nearbyEvents.length),
      unit: "",
      icon: QueryStatsIcon,
      color: "#f97316"
    },
    {
      label: "Risk Level",
      value: safety.safetyTier,
      unit: "",
      icon: PublicIcon,
      color: tierColor
    },
    {
      label: "Radius",
      value: formatStatValue(safety.proximityRadiusKm),
      unit: "km",
      icon: ShowChartIcon,
      color: "#60a5fa"
    }
  ];

  const stats: StatCard[] = isSafetyMode ? safetyStats : isArchiveMode ? archiveStats : liveStats;

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "#010409", color: "#d1d5db" }}>
      <Box
        sx={{
          width: "100%",
          maxWidth: "1600px",
          mx: "auto",
          px: { xs: 2, md: 3, lg: 4 },
          py: { xs: 2, md: 3 },
          display: "flex",
          flexDirection: "column",
          gap: 2.5,
          minHeight: "100vh"
        }}
      >
        <Box
          sx={{
            display: "flex",
            flexDirection: { xs: "column", md: "row" },
            justifyContent: "space-between",
            alignItems: { xs: "stretch", md: "flex-end" },
            gap: 2
          }}
        >
          <Box>
            <Box sx={{ display: "flex", gap: 1, mb: 1.25, flexWrap: "wrap", alignItems: "center" }}>
              {/* Live Nowcast badge — clickable */}
              <Box
                component="button"
                onClick={exitToLiveMode}
                sx={{
                  px: 1,
                  py: 0.4,
                  borderRadius: 1,
                  cursor: "pointer",
                  border: isArchiveMode
                    ? "1px dashed rgba(249,115,22,0.25)"
                    : "1px solid rgba(249,115,22,0.5)",
                  bgcolor: isArchiveMode ? "transparent" : "rgba(249,115,22,0.12)",
                  color: isArchiveMode ? "rgba(249,115,22,0.45)" : "#f97316",
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: "0.14em",
                  textTransform: "uppercase",
                  transition: "all 0.15s",
                  "&:hover": { opacity: 0.8 }
                }}
              >
                Live Nowcast
              </Box>
              {/* Historical Archive badge — clickable */}
              <Box
                component="button"
                onClick={enterArchiveMode}
                sx={{
                  px: 1,
                  py: 0.4,
                  borderRadius: 1,
                  cursor: "pointer",
                  border: isArchiveMode
                    ? "1px solid rgba(59,130,246,0.5)"
                    : "1px dashed rgba(59,130,246,0.25)",
                  bgcolor: isArchiveMode ? "rgba(59,130,246,0.12)" : "transparent",
                  color: isArchiveMode ? "#60a5fa" : "rgba(96,165,250,0.45)",
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: "0.14em",
                  textTransform: "uppercase",
                  transition: "all 0.15s",
                  "&:hover": { opacity: 0.8 }
                }}
              >
                Historical Archive
              </Box>
              {/* Personal Safety Mode badge — clickable */}
              <Box
                component="button"
                onClick={() => isSafetyMode ? disableSafetyMode() : enableSafetyMode()}
                sx={{
                  px: 1,
                  py: 0.4,
                  borderRadius: 1,
                  cursor: "pointer",
                  border: isSafetyMode
                    ? "1px solid rgba(239,68,68,0.5)"
                    : "1px dashed rgba(239,68,68,0.25)",
                  bgcolor: isSafetyMode ? "rgba(239,68,68,0.12)" : "transparent",
                  color: isSafetyMode ? "#f87171" : "rgba(248,113,113,0.45)",
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: "0.14em",
                  textTransform: "uppercase",
                  transition: "all 0.15s",
                  "&:hover": { opacity: 0.8 }
                }}
              >
                Personal Safety
              </Box>
              <Box
                sx={{
                  px: 1,
                  py: 0.4,
                  borderRadius: 1,
                  bgcolor: "rgba(59,130,246,0.12)",
                  border: "1px solid rgba(59,130,246,0.25)",
                  color: "#60a5fa",
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: "0.14em",
                  textTransform: "uppercase"
                }}
              >
                Earth Tools Beta
              </Box>
            </Box>

            {/* Title */}
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.8 }}>
              {isArchiveMode && <AccessTimeIcon sx={{ color: "#60a5fa", fontSize: 28 }} />}
              <Typography variant="h3" sx={{ color: "#fff", fontWeight: 800, lineHeight: 1.05, letterSpacing: "-0.02em" }}>
                {isArchiveMode ? "Wildfire Snapshot" : "Wildfire Nowcast"}
              </Typography>
            </Box>

            {/* Subtitle */}
            {isArchiveMode ? (
              <Typography sx={{ color: "#6b7280", maxWidth: 650, fontSize: 14, lineHeight: 1.65 }}>
                Viewing Archive for {archive.archiveDate ?? "—"}. Quadrant:{" "}
                {activeTimeframeDef?.label ?? "—"}.
              </Typography>
            ) : (
              <Typography sx={{ color: "#6b7280", maxWidth: 650, fontSize: 14, lineHeight: 1.65 }}>
                Global thermal anomalies from NASA FIRMS with denoiser-scored event confidence and live spread context.
              </Typography>
            )}

            {/* Archive date + timeframe controls */}
            {isArchiveMode && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mt: 1.5, flexWrap: "wrap" }}>
                <TextField
                  type="date"
                  size="small"
                  value={archive.archiveDate ?? ""}
                  onChange={(e) => setArchiveDate(e.target.value)}
                  inputProps={{ max: new Date().toISOString().slice(0, 10) }}
                  sx={{
                    "& .MuiOutlinedInput-root": {
                      bgcolor: "#0d1117",
                      borderRadius: 2,
                      fontSize: 12,
                      color: "#e5e7eb"
                    },
                    "& input": { colorScheme: "light" }
                  }}
                />
                <Box sx={{ display: "flex", gap: 0.5 }}>
                  {TIMEFRAME_DEFS.map((def) => {
                    const Icon = TIMEFRAME_ICONS[def.id];
                    const isSelected = archive.archiveTimeframe === def.id;
                    return (
                      <Tooltip key={def.id} title={def.label}>
                        <Box
                          component="button"
                          onClick={() => setArchiveTimeframe(def.id)}
                          sx={{
                            p: 0.8,
                            borderRadius: 1.5,
                            border: isSelected ? "1px solid rgba(59,130,246,0.6)" : "1px solid rgba(255,255,255,0.1)",
                            bgcolor: isSelected ? "rgba(59,130,246,0.15)" : "#0d1117",
                            color: isSelected ? "#60a5fa" : "#6b7280",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                            transition: "all 0.15s",
                            "&:hover": { borderColor: "rgba(59,130,246,0.4)", color: "#93c5fd" }
                          }}
                        >
                          <Icon sx={{ fontSize: 16 }} />
                        </Box>
                      </Tooltip>
                    );
                  })}
                </Box>
                {/* Ingestion status indicator */}
                {archiveData.status === "checking" && (
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                    <CircularProgress size={14} sx={{ color: "#60a5fa" }} />
                    <Typography sx={{ fontSize: 11, color: "#6b7280" }}>Checking data…</Typography>
                  </Box>
                )}
                {archiveData.status === "ingesting" && (
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                    <CircularProgress size={14} sx={{ color: "#f97316" }} />
                    <Typography sx={{ fontSize: 11, color: "#f97316" }}>
                      {archiveData.message ?? "Ingesting data…"}
                    </Typography>
                  </Box>
                )}
                {archiveData.status === "unavailable" && (
                  <Typography sx={{ fontSize: 11, color: "#ef4444" }}>
                    {archiveData.message ?? "Data unavailable for this date."}
                  </Typography>
                )}
              </Box>
            )}
          </Box>

          <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, flexWrap: "wrap" }}>
            <RegionFilter
              events={visibleEvents}
              value={regionFilter}
              onChange={setRegionFilter}
            />

            <Box
              sx={{
                p: 0.5,
                bgcolor: "#0d1117",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 2,
                display: "flex",
                alignItems: "center",
                gap: 0.5
              }}
            >
              <Button
                size="small"
                onClick={() => setConfidenceFilter("All")}
                sx={{
                  px: 1.6,
                  py: 0.7,
                  fontSize: 10,
                  fontWeight: 800,
                  letterSpacing: "0.08em",
                  borderRadius: 1.2,
                  color: confidenceFilter === "All" ? "#fff" : "#9ca3af",
                  bgcolor: confidenceFilter === "All" ? "#21262d" : "transparent"
                }}
              >
                All
              </Button>
              <Button
                size="small"
                onClick={() => setConfidenceFilter("High")}
                sx={{
                  px: 1.6,
                  py: 0.7,
                  fontSize: 10,
                  fontWeight: 800,
                  letterSpacing: "0.08em",
                  borderRadius: 1.2,
                  color: confidenceFilter === "High" ? "#f97316" : "#9ca3af",
                  bgcolor: confidenceFilter === "High" ? "#21262d" : "transparent"
                }}
              >
                High Confidence
              </Button>
            </Box>
          </Box>
        </Box>

        {!isArchiveMode && <DataFreshnessBanner />}
        <ForecastNotification />
        {isSafetyMode && (
          <SafetyStatusBar
            onDisable={disableSafetyMode}
            onLocate={requestLocation}
            nearbyCount={nearbyEvents.length}
          />
        )}

        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", lg: "minmax(0,8fr) minmax(360px,4fr)" },
            gap: 2.5,
            flex: 1,
            minHeight: 0
          }}
        >
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5, minHeight: 0 }}>
            <Box sx={{ minHeight: { xs: 420, md: 520 }, flex: 1 }}>
              <FireMap
                onVisibleEventsChange={setVisibleEvents}
                confidenceFilter={confidenceFilter}
              />
            </Box>

            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "repeat(2, minmax(0,1fr))", md: "repeat(4, minmax(0,1fr))" },
                gap: 1.5
              }}
            >
              {stats.map((stat) => (
                <Paper key={stat.label} sx={{ p: 2, bgcolor: "#0d1117", borderColor: "rgba(255,255,255,0.08)", borderRadius: 2.5 }}>
                  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.4 }}>
                    <Typography sx={{ fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: "#6b7280", fontWeight: 800 }}>
                      {stat.label}
                    </Typography>
                    <stat.icon sx={{ fontSize: 16, color: stat.color }} />
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "baseline", gap: 0.75 }}>
                    <Typography sx={{ color: "#fff", fontSize: 28, lineHeight: 1, fontWeight: 800 }}>{stat.value}</Typography>
                    {stat.unit && (
                      <Typography sx={{ color: "#4b5563", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 700 }}>
                        {stat.unit}
                      </Typography>
                    )}
                  </Box>
                </Paper>
              ))}
            </Box>
          </Box>

          <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5, minHeight: 0 }}>
            <Box sx={{ flex: "0 0 auto", minHeight: { xs: 360, lg: 0 } }}>
              <FireDetailsPanel visibleEvents={filteredEvents} />
            </Box>
            {!isArchiveMode && (
              <Box sx={{ flex: 1, minHeight: 320 }}>
                <AIChatAssistant />
              </Box>
            )}
          </Box>
        </Box>
      </Box>

      <Box sx={{ borderTop: "1px solid rgba(255,255,255,0.05)", py: 4 }}>
        <Box
          sx={{
            maxWidth: "1600px",
            mx: "auto",
            px: { xs: 2, md: 4 },
            display: "flex",
            flexDirection: { xs: "column", md: "row" },
            alignItems: "center",
            justifyContent: "space-between",
            gap: 1.5
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <PublicIcon sx={{ fontSize: 16, color: "#f97316" }} />
            <Typography sx={{ fontSize: 10, color: "#fff", fontWeight: 900, letterSpacing: "0.2em", textTransform: "uppercase" }}>
              Earth Tools Ecosystem
            </Typography>
          </Box>
          <Typography sx={{ fontSize: 10, color: "#4b5563", fontWeight: 800, letterSpacing: "0.12em", textTransform: "uppercase" }}>
            Open Ecological Intelligence • 2026
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}
