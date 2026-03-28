import { useEffect, useMemo, useState } from "react";
import {
  Box,
  Paper,
  Typography
} from "@mui/material";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import PublicIcon from "@mui/icons-material/Public";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import ShowChartIcon from "@mui/icons-material/ShowChart";
import VerifiedIcon from "@mui/icons-material/Verified";

import AIChatAssistant from "./components/AIChatAssistant";
import WatchlistDashboard from "./components/WatchlistDashboard";
import DataFreshnessBanner from "./components/DataFreshnessBanner";
import FireDetailsPanel from "./components/FireDetailsPanel";
import FireMap from "./components/FireMap";
import ForecastNotification from "./components/ForecastNotification";
import RegionFilter from "./components/RegionFilter";
import ReviewQueuePanel from "./components/ReviewQueuePanel";
import SafetyStatusBar from "./components/SafetyStatusBar";
import ArchiveRangeScrubber from "./components/ArchiveRangeScrubber";
import { ArchiveControls, TIMEFRAME_ICONS } from "./components/archive/ArchiveControls";
import { AppLayout } from "./components/layout/AppLayout";
import { useArchiveData } from "./hooks/useArchiveData";
import { useArchiveRangeData } from "./hooks/useArchiveRangeData";
import { useAppDerivedState } from "./hooks/useAppDerivedState";
import { useForecastPolling } from "./hooks/useForecastPolling";
import { useGeolocation } from "./hooks/useGeolocation";
import { useSafetyMetrics } from "./hooks/useSafetyMetrics";
import { useAppStore } from "./state/store";
import type { FireEvent } from "./types/api";
import type { AssistantConfidenceFilter } from "./types/state";
import {
  CONTINENT_VIEWPORTS,
  EMPTY_REGION_FILTER,
  formatRegionFilter,
  matchesRegionFilter,
  type RegionFilterValue,
} from "./utils/continents";
import { eventsWithinRadius } from "./utils/geo";
import { TIMEFRAME_DEFS } from "./utils/time";

function formatStatValue(value: number, digits = 0): string {
  if (!Number.isFinite(value)) return "n/a";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
}

const SAFETY_TIER_COLORS: Record<string, string> = {
  SAFE: "#22c55e", WATCH: "#eab308", WARNING: "#f97316", DANGER: "#ef4444"
};

type StatCard = { label: string; value: string; unit: string; icon: React.ElementType; color: string };

export default function App(): JSX.Element {
  const [visibleEvents, setVisibleEvents] = useState<FireEvent[]>([]);
  const [regionFilter, setRegionFilter] = useState<RegionFilterValue>(EMPTY_REGION_FILTER);
  const [confidenceFilter, setConfidenceFilter] = useState<AssistantConfidenceFilter>("All");
  const setAssistantViewContext = useAppStore((s) => s.setAssistantViewContext);
  const archive = useAppStore((s) => s.archive);
  const enterArchiveMode = useAppStore((s) => s.enterArchiveMode);
  const exitToLiveMode = useAppStore((s) => s.exitToLiveMode);
  const setScrubDate = useAppStore((s) => s.setScrubDate);
  const safety = useAppStore((s) => s.safety);
  const enableSafetyMode = useAppStore((s) => s.enableSafetyMode);
  const disableSafetyMode = useAppStore((s) => s.disableSafetyMode);
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);
  const setMapView = useAppStore((s) => s.setMapView);
  const mapView = useAppStore((s) => s.mapView);

  const isArchiveMode = archive.viewMode === "archive";
  const isRangeMode = isArchiveMode && archive.archiveSubMode === "range";
  const isSafetyMode = safety.enabled;
  const archiveData = useArchiveData();
  const archiveRangeData = useArchiveRangeData();
  const { requestLocation } = useGeolocation();

  useForecastPolling();

  const {
    filteredEvents,
    safetyEvents,
    totalDetections,
    activePerimeters,
    averageScore,
    confidencePercent,
    maxIntensityFrp,
    topEventsForAssistant
  } = useAppDerivedState({
    visibleEvents,
    confidenceFilter,
    regionFilter,
    isArchiveMode,
    safetyProximityRadiusKm: safety.proximityRadiusKm,
    safetyUserLocation: safety.userLocation ?? null
  });

  useSafetyMetrics(safetyEvents);

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
    const matching = visibleEvents.filter((e) => matchesRegionFilter(e, regionFilter));
    const targetZoom = admin1 ? 6 : country ? 5 : 3;
    const geoEvents = matching.filter((e) => e.lat != null && e.lon != null);
    if (geoEvents.length > 0) {
      const avgLat = geoEvents.reduce((s, e) => s + e.lat!, 0) / geoEvents.length;
      const avgLon = geoEvents.reduce((s, e) => s + e.lon!, 0) / geoEvents.length;
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

  const activeTimeframeDef = isArchiveMode
    ? TIMEFRAME_DEFS.find((d) => d.id === archive.archiveTimeframe) ?? TIMEFRAME_DEFS[1]
    : null;

  const tierColor = SAFETY_TIER_COLORS[safety.safetyTier] ?? "#6b7280";

  const liveStats: StatCard[] = [
    { label: "Total Detections", value: formatStatValue(totalDetections), unit: "", icon: QueryStatsIcon, color: "#f97316" },
    { label: "Active Perimeters", value: formatStatValue(activePerimeters), unit: "", icon: LocalFireDepartmentIcon, color: "#ef4444" },
    { label: "Avg Event Score", value: averageScore === null ? "n/a" : formatStatValue(averageScore, 3), unit: "0-1", icon: PublicIcon, color: "#60a5fa" },
    { label: "High Confidence", value: confidencePercent === null ? "n/a" : formatStatValue(confidencePercent, 1), unit: "%", icon: VerifiedIcon, color: "#4ade80" }
  ];

  const archiveStats: StatCard[] = isRangeMode
    ? [
      { label: "Viewing Date", value: archive.scrubDate ?? "—", unit: "", icon: AccessTimeIcon, color: "#60a5fa" },
      { label: "Active Events", value: formatStatValue(filteredEvents.length), unit: "", icon: ShowChartIcon, color: "#f97316" },
      { label: "Max Intensity", value: maxIntensityFrp === null ? "n/a" : formatStatValue(maxIntensityFrp), unit: "MW", icon: LocalFireDepartmentIcon, color: "#ef4444" },
      { label: "Context", value: "REPLAY", unit: "", icon: PublicIcon, color: "#4ade80" }
    ]
    : [
      { label: "Time Filter", value: activeTimeframeDef?.label ?? "—", unit: "", icon: activeTimeframeDef ? TIMEFRAME_ICONS[activeTimeframeDef.id] : AccessTimeIcon, color: "#60a5fa" },
      { label: "Active Events", value: formatStatValue(filteredEvents.length), unit: "", icon: ShowChartIcon, color: "#f97316" },
      { label: "Max Intensity", value: maxIntensityFrp === null ? "n/a" : formatStatValue(maxIntensityFrp), unit: "MW", icon: LocalFireDepartmentIcon, color: "#ef4444" },
      { label: "Context", value: "ARCHIVE", unit: "", icon: PublicIcon, color: "#4ade80" }
    ];

  const safetyStats: StatCard[] = [
    { label: "Nearest Fire", value: safety.nearestFireDistanceKm !== null ? formatStatValue(safety.nearestFireDistanceKm, 1) : "—", unit: "km", icon: LocalFireDepartmentIcon, color: tierColor },
    { label: "Fires in Radius", value: formatStatValue(nearbyEvents.length), unit: "", icon: QueryStatsIcon, color: "#f97316" },
    { label: "Risk Level", value: safety.safetyTier, unit: "", icon: PublicIcon, color: tierColor },
    { label: "Radius", value: formatStatValue(safety.proximityRadiusKm), unit: "km", icon: ShowChartIcon, color: "#60a5fa" }
  ];

  const stats: StatCard[] = isSafetyMode ? safetyStats : isArchiveMode ? archiveStats : liveStats;

  // ── Toolbar ──────────────────────────────────────────────────────────────
  const toolbar = (
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
          <Box component="button" onClick={exitToLiveMode} sx={{ px: 1, py: 0.4, borderRadius: 1, cursor: "pointer", border: isArchiveMode ? "1px dashed rgba(249,115,22,0.25)" : "1px solid rgba(249,115,22,0.5)", bgcolor: isArchiveMode ? "transparent" : "rgba(249,115,22,0.12)", color: isArchiveMode ? "rgba(249,115,22,0.45)" : "#f97316", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase", transition: "all 0.15s", "&:hover": { opacity: 0.8 } }}>Live Nowcast</Box>
          <Box component="button" onClick={enterArchiveMode} sx={{ px: 1, py: 0.4, borderRadius: 1, cursor: "pointer", border: isArchiveMode ? "1px solid rgba(59,130,246,0.5)" : "1px dashed rgba(59,130,246,0.25)", bgcolor: isArchiveMode ? "rgba(59,130,246,0.12)" : "transparent", color: isArchiveMode ? "#60a5fa" : "rgba(96,165,250,0.45)", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase", transition: "all 0.15s", "&:hover": { opacity: 0.8 } }}>Historical Archive</Box>
          <Box component="button" onClick={() => isSafetyMode ? disableSafetyMode() : enableSafetyMode()} sx={{ px: 1, py: 0.4, borderRadius: 1, cursor: "pointer", border: isSafetyMode ? "1px solid rgba(239,68,68,0.5)" : "1px dashed rgba(239,68,68,0.25)", bgcolor: isSafetyMode ? "rgba(239,68,68,0.12)" : "transparent", color: isSafetyMode ? "#f87171" : "rgba(248,113,113,0.45)", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase", transition: "all 0.15s", "&:hover": { opacity: 0.8 } }}>Personal Safety</Box>
          <Box sx={{ px: 1, py: 0.4, borderRadius: 1, bgcolor: "rgba(59,130,246,0.12)", border: "1px solid rgba(59,130,246,0.25)", color: "#60a5fa", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>ALPHA</Box>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.8 }}>
          {isArchiveMode && <AccessTimeIcon sx={{ color: "#60a5fa", fontSize: 28 }} />}
          <Typography variant="h3" sx={{ color: "#fff", fontWeight: 800, lineHeight: 1.05, letterSpacing: "-0.02em" }}>
            {isRangeMode ? "Wildfire Replay" : isArchiveMode ? "Wildfire Snapshot" : "Wildfire Nowcast"}
          </Typography>
        </Box>
        {isRangeMode ? (
          <Typography sx={{ color: "#6b7280", maxWidth: 650, fontSize: 14, lineHeight: 1.65 }}>
            Replaying {archive.rangeStart ?? "—"} → {archive.rangeEnd ?? "—"}.{archive.scrubDate ? ` Viewing ${archive.scrubDate}.` : " Select a day on the scrubber below."}
          </Typography>
        ) : isArchiveMode ? (
          <Typography sx={{ color: "#6b7280", maxWidth: 650, fontSize: 14, lineHeight: 1.65 }}>
            Viewing Archive for {archive.archiveDate ?? "—"}. Quadrant: {activeTimeframeDef?.label ?? "—"}.
          </Typography>
        ) : (
          <Typography sx={{ color: "#6b7280", maxWidth: 650, fontSize: 14, lineHeight: 1.65 }}>
            Global thermal anomalies from NASA FIRMS with denoiser-scored event confidence and live spread context.
          </Typography>
        )}
        {isArchiveMode && (
          <ArchiveControls archiveData={archiveData} archiveRangeData={archiveRangeData} />
        )}
      </Box>

      <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, flexWrap: "wrap" }}>
        <RegionFilter events={visibleEvents} value={regionFilter} onChange={setRegionFilter} />
        <Box sx={{ p: 0.5, bgcolor: "#0d1117", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 2, display: "flex", alignItems: "center", gap: 0.5 }}>
          <Box component="button" onClick={() => setConfidenceFilter("All")} sx={{ px: 1.6, py: 0.7, fontSize: 10, fontWeight: 800, letterSpacing: "0.08em", borderRadius: 1.2, border: "none", cursor: "pointer", color: confidenceFilter === "All" ? "#fff" : "#9ca3af", bgcolor: confidenceFilter === "All" ? "#21262d" : "transparent" }}>All</Box>
          <Box component="button" onClick={() => setConfidenceFilter("High")} sx={{ px: 1.6, py: 0.7, fontSize: 10, fontWeight: 800, letterSpacing: "0.08em", borderRadius: 1.2, border: "none", cursor: "pointer", color: confidenceFilter === "High" ? "#f97316" : "#9ca3af", bgcolor: confidenceFilter === "High" ? "#21262d" : "transparent" }}>High Confidence</Box>
        </Box>
      </Box>
    </Box>
  );

  // ── Stats row ────────────────────────────────────────────────────────────
  const statsRow = (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "repeat(2, minmax(0,1fr))", md: "repeat(4, minmax(0,1fr))" }, gap: 1.5 }}>
      {stats.map((stat) => (
        <Paper key={stat.label} sx={{ p: 2, bgcolor: "#0d1117", borderColor: "rgba(255,255,255,0.08)", borderRadius: 2.5 }}>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.4 }}>
            <Typography sx={{ fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: "#6b7280", fontWeight: 800 }}>{stat.label}</Typography>
            <stat.icon sx={{ fontSize: 16, color: stat.color }} />
          </Box>
          <Box sx={{ display: "flex", alignItems: "baseline", gap: 0.75 }}>
            <Typography sx={{ color: "#fff", fontSize: 28, lineHeight: 1, fontWeight: 800 }}>{stat.value}</Typography>
            {stat.unit && <Typography sx={{ color: "#4b5563", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 700 }}>{stat.unit}</Typography>}
          </Box>
        </Paper>
      ))}
    </Box>
  );

  // ── Scrubber ─────────────────────────────────────────────────────────────
  const scrubber = isRangeMode && archive.rangeStart && archive.rangeEnd && archiveRangeData.dayStatuses.length > 0 ? (
    <ArchiveRangeScrubber
      startDate={archive.rangeStart}
      endDate={archive.rangeEnd}
      scrubDate={archive.scrubDate}
      dayStatuses={archiveRangeData.dayStatuses}
      onScrub={setScrubDate}
    />
  ) : undefined;

  // ── Sidebar ──────────────────────────────────────────────────────────────
  const sidebar = (
    <>
      {!isArchiveMode && <DataFreshnessBanner />}
      <ForecastNotification />
      {isSafetyMode && (
        <SafetyStatusBar onDisable={disableSafetyMode} onLocate={requestLocation} nearbyCount={nearbyEvents.length} />
      )}
      <Box sx={{ flex: "0 0 auto", minHeight: { xs: 360, lg: 0 } }}>
        <FireDetailsPanel visibleEvents={filteredEvents} />
      </Box>
      {!isArchiveMode && (
        <Box sx={{ flex: "0 0 auto" }}>
          <ReviewQueuePanel visibleEvents={filteredEvents} />
        </Box>
      )}
      {!isArchiveMode && (
        <Box sx={{ flex: "0 0 auto" }}>
          <WatchlistDashboard />
        </Box>
      )}
      {!isArchiveMode && (
        <Box sx={{ flex: 1, minHeight: 320 }}>
          <AIChatAssistant />
        </Box>
      )}
    </>
  );

  return (
    <AppLayout
      toolbar={toolbar}
      mainContent={<FireMap onVisibleEventsChange={setVisibleEvents} confidenceFilter={confidenceFilter} />}
      sidebar={sidebar}
      statsRow={statsRow}
      scrubber={scrubber}
    />
  );
}
