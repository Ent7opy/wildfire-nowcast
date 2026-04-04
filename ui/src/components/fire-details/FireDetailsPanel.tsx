import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, IconButton, Typography } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import NewspaperIcon from "@mui/icons-material/Newspaper";
import ShowChartIcon from "@mui/icons-material/ShowChart";
import { useMutation, useQuery } from "@tanstack/react-query";

import {
  ApiError,
  ApiUnavailableError,
  buildEventKey,
  createJitForecast,
  createJitForecastFromFront,
  getActiveSpreadModelId,
  getReverseGeocode
} from "../../api/client";
import { useAppStore } from "../../state/store";
import type { FireEvent, ReverseGeocodeResponse } from "../../types/api";
import { comparePriorityFeedEvents } from "../../utils/priorityFeed";
import { computeArchiveTimeRange } from "../../utils/time";
import {
  safeNumber,
  coordinateKey,
  hasDirectLocation,
  isWildfireArticle,
  type GdeltArticle
} from "./types";
import { FireOverviewTab } from "./FireOverviewTab";
import { FireDetectionsTab } from "./FireDetectionsTab";
import { FireFrontsTab } from "./FireFrontsTab";
import { NewsExpandedModal } from "./NewsExpandedModal";

interface FireDetailsPanelProps {
  visibleEvents: FireEvent[];
}

export function FireDetailsPanel({ visibleEvents }: FireDetailsPanelProps): JSX.Element {
  const selectedEvent = useAppStore((s) => s.selectedEvent);
  const setSelectedEvent = useAppStore((s) => s.setSelectedEvent);
  const frontIndex = useAppStore((s) => s.frontIndexByEvent);
  const startForecastJob = useAppStore((s) => s.startForecastJob);
  const setForecastNotification = useAppStore((s) => s.setForecastNotification);
  const safety = useAppStore((s) => s.safety);
  const archive = useAppStore((s) => s.archive);

  const isArchiveMode = archive.viewMode === "archive";
  const isSafetyMode = safety.enabled;

  const [submitError, setSubmitError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"telemetry" | "news">("telemetry");
  const [newsExpanded, setNewsExpanded] = useState(false);
  const [resolvedGeocodes, setResolvedGeocodes] = useState<Record<string, ReverseGeocodeResponse>>({});
  const resolvedGeocodesRef = useRef<Record<string, ReverseGeocodeResponse>>({});
  const geocodeInFlightRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    setSubmitError(null);
    setNewsExpanded(false);
  }, [selectedEvent?.event_id]);

  useEffect(() => {
    resolvedGeocodesRef.current = resolvedGeocodes;
  }, [resolvedGeocodes]);

  const topFires = useMemo(() => {
    return [...visibleEvents]
      .sort(comparePriorityFeedEvents)
      .slice(0, 5);
  }, [visibleEvents]);

  const gdeltTimeParam = useMemo(() => {
    if (isArchiveMode && archive.archiveDate && archive.archiveTimeframe) {
      const { startTime, endTime } = computeArchiveTimeRange(archive.archiveDate, archive.archiveTimeframe);
      const fmt = (d: Date) =>
        `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}${String(d.getHours()).padStart(2, "0")}${String(d.getMinutes()).padStart(2, "0")}${String(d.getSeconds()).padStart(2, "0")}`;
      return { startdatetime: fmt(startTime), enddatetime: fmt(endTime) };
    }
    return null;
  }, [isArchiveMode, archive.archiveDate, archive.archiveTimeframe]);

  const { data: newsData, isLoading: newsLoading, isError: newsError } = useQuery({
    queryKey: ["gdelt-news", gdeltTimeParam],
    queryFn: async ({ signal }) => {
      const query = encodeURIComponent(
        "(wildfire OR bushfire OR \"forest fire\" OR \"brush fire\" OR \"grass fire\" OR firefighter OR \"fire evacuation\" OR \"fire season\" OR \"fire crews\" OR \"prescribed burn\" OR \"controlled burn\" OR \"fire weather\" OR \"red flag warning\") sourcelang:english"
      );
      const timeQuery = gdeltTimeParam
        ? `&startdatetime=${gdeltTimeParam.startdatetime}&enddatetime=${gdeltTimeParam.enddatetime}`
        : `&timespan=12h`;
      const url = `https://api.gdeltproject.org/api/v2/doc/doc?query=${query}&mode=artlist&format=json${timeQuery}&sort=datedesc&maxrecords=75`;
      const res = await fetch(url, { signal });
      if (!res.ok) throw new Error("Failed to fetch news");
      const json = await res.json() as { articles?: GdeltArticle[] };
      return (json.articles ?? []).filter((a) => isWildfireArticle(a.title));
    },
    staleTime: 5 * 60 * 1000,
    refetchInterval: isArchiveMode ? false : 10 * 60 * 1000,
    enabled: activeTab === "news"
  });

  const resolveLocation = useCallback(async (event: FireEvent): Promise<void> => {
    if (hasDirectLocation(event)) {
      return;
    }
    const lat = safeNumber(event.lat);
    const lon = safeNumber(event.lon);
    const key = coordinateKey(lat, lon);
    if (lat === null || lon === null || key === null) {
      return;
    }
    if (resolvedGeocodesRef.current[key] || geocodeInFlightRef.current.has(key)) {
      return;
    }

    geocodeInFlightRef.current.add(key);
    try {
      const geocode = await getReverseGeocode({ lat, lon });
      setResolvedGeocodes((prev) => {
        if (prev[key]) {
          return prev;
        }
        return { ...prev, [key]: geocode };
      });
    } catch {
      // Ignore transient reverse geocode errors in UI and keep fallback label.
    } finally {
      geocodeInFlightRef.current.delete(key);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const hydrateTopFires = async (): Promise<void> => {
      for (const event of topFires) {
        if (cancelled) return;
        await resolveLocation(event);
      }
    };
    void hydrateTopFires();
    return () => { cancelled = true; };
  }, [resolveLocation, topFires]);

  useEffect(() => {
    if (!selectedEvent) return;
    void resolveLocation(selectedEvent);
  }, [resolveLocation, selectedEvent]);

  const forecastMutation = useMutation({
    mutationFn: async (event: FireEvent) => {
      const lat = safeNumber(event.lat);
      const lon = safeNumber(event.lon);
      if (lat === null || lon === null) {
        throw new ApiError("Selected event is missing coordinates.");
      }

      const eventKey = buildEventKey(event, lat, lon);
      const modelId = await getActiveSpreadModelId();

      const refTime = event.end_time ? new Date(event.end_time) : new Date();
      const forecastReferenceTime = Number.isNaN(refTime.getTime()) ? new Date() : refTime;

      const front = event.event_id ? frontIndex[String(event.event_id)] : undefined;
      const cacheKey = coordinateKey(lat, lon);
      const geocoded = cacheKey ? resolvedGeocodesRef.current[cacheKey] : null;
      const location = (() => {
        const candidates = [
          geocoded?.location_name,
          event.location_name,
          event.region_name,
          geocoded?.admin1_name,
          event.admin1_name,
          event.admin0_name,
          geocoded?.country,
          event.country,
          geocoded?.display_name
        ];
        for (const candidate of candidates) {
          if (typeof candidate === "string" && candidate.trim().length > 0) {
            return candidate.trim();
          }
        }
        return "Unresolved location";
      })();
      const requestContext = {
        eventId: event.event_id ? String(event.event_id) : undefined,
        eventKey,
        frontId: front?.frontId,
        lat,
        lon,
        locationLabel: location,
        eventSnapshot: { ...event, lat, lon }
      };

      let response;
      if (front?.frontId) {
        response = await createJitForecastFromFront({
          frontId: front.frontId,
          bufferKm: 3,
          horizonsHours: [24, 48, 72],
          forecastReferenceTime,
          modelId
        });
      } else {
        const radiusKm = 20;
        const latDelta = radiusKm / 111;
        const lonScale = Math.max(Math.cos((Math.PI / 180) * lat), 0.1);
        const lonDelta = radiusKm / (111 * lonScale);
        const bbox: [number, number, number, number] = [lon - lonDelta, lat - latDelta, lon + lonDelta, lat + latDelta];

        response = await createJitForecast({
          bbox,
          horizonsHours: [24, 48, 72],
          forecastReferenceTime,
          modelId
        });
      }

      return { response, requestContext };
    },
    onSuccess: (payload) => {
      startForecastJob(payload.response.job_id, payload.requestContext);
      setForecastNotification({
        kind: "info",
        message: "Spread forecast is being generated and may take around a minute.",
        createdAt: Date.now(),
        ttlSeconds: 20
      });
      setSubmitError(null);
    },
    onError: (error: unknown) => {
      if (error instanceof ApiUnavailableError) {
        setSubmitError("Data service is unavailable right now. Please try again in a moment.");
        return;
      }
      if (error instanceof ApiError) {
        setSubmitError(error.message);
        return;
      }
      setSubmitError("Failed to start forecast.");
    }
  });

  // ── No event selected: show priority feed + ground reports ──────────────
  if (!selectedEvent) {
    return (
      <Box
        sx={{
          bgcolor: "#0d1117",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 3,
          overflow: "hidden",
          boxShadow: "0 24px 80px rgba(0,0,0,0.35)",
          display: "flex",
          flexDirection: "column",
          height: 560
        }}
      >
        {/* Tab switcher */}
        <Box sx={{ display: "flex", borderBottom: "1px solid rgba(255,255,255,0.05)", bgcolor: "#161b22" }}>
          <Box
            component="button"
            onClick={() => setActiveTab("telemetry")}
            sx={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 0.75,
              py: 1.5,
              fontSize: 10,
              fontWeight: 800,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              border: "none",
              cursor: "pointer",
              transition: "all 160ms ease",
              ...(activeTab === "telemetry"
                ? { bgcolor: "#0d1117", color: "#fff" }
                : { bgcolor: "transparent", color: "#4b5563", "&:hover": { color: "#9ca3af" } })
            }}
          >
            <ShowChartIcon sx={{ fontSize: 12 }} />
            Telemetry
          </Box>
          <Box
            component="button"
            onClick={() => setActiveTab("news")}
            sx={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 0.75,
              py: 1.5,
              fontSize: 10,
              fontWeight: 800,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              border: "none",
              borderLeft: "1px solid rgba(255,255,255,0.05)",
              cursor: "pointer",
              transition: "all 160ms ease",
              ...(activeTab === "news"
                ? { bgcolor: "#0d1117", color: "#f97316" }
                : { bgcolor: "transparent", color: "#4b5563", "&:hover": { color: "#9ca3af" } })
            }}
          >
            <NewspaperIcon sx={{ fontSize: 12 }} />
            Ground Reports
          </Box>
        </Box>

        {activeTab === "telemetry" && (
          <FireOverviewTab
            topFires={topFires}
            resolvedGeocodes={resolvedGeocodes}
          />
        )}

        {activeTab === "news" && (
          <FireDetectionsTab
            newsData={newsData}
            newsLoading={newsLoading}
            newsError={newsError}
            onExpandNews={() => setNewsExpanded(true)}
          />
        )}

        <NewsExpandedModal
          open={newsExpanded}
          articles={newsData ?? []}
          onClose={() => setNewsExpanded(false)}
        />
      </Box>
    );
  }

  // ── Event selected: show Fire Inspector ─────────────────────────────────
  return (
    <Box
      sx={{
        bgcolor: "#0d1117",
        border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 3,
        overflow: "hidden",
        boxShadow: "0 24px 80px rgba(0,0,0,0.35)",
        display: "flex",
        flexDirection: "column",
        minHeight: 360
      }}
    >
      {/* Panel header */}
      <Box sx={{ px: 2.25, py: 1.6, borderBottom: "1px solid rgba(255,255,255,0.05)", bgcolor: "#161b22", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Typography sx={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.14em", textTransform: "uppercase", color: "#fff", display: "flex", alignItems: "center", gap: 0.75 }}>
          <LocalFireDepartmentIcon sx={{ fontSize: 14, color: isSafetyMode ? "#ef4444" : "#f97316" }} />
          {isSafetyMode ? "Immediate Threat Assessment" : "Fire Inspector"}
        </Typography>
        <IconButton
          size="small"
          onClick={() => setSelectedEvent(null)}
          sx={{ color: "#6b7280", '&:hover': { color: "#fff", bgcolor: "rgba(255,255,255,0.08)" } }}
        >
          <CloseIcon sx={{ fontSize: 16 }} />
        </IconButton>
      </Box>

      <FireFrontsTab
        selectedEvent={selectedEvent}
        resolvedGeocodes={resolvedGeocodes}
        submitError={submitError}
        forecastMutation={{
          mutate: forecastMutation.mutate,
          isPending: forecastMutation.isPending
        }}
      />
    </Box>
  );
}

export default FireDetailsPanel;
