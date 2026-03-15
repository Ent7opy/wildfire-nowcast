import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
  Stack,
  Typography
} from "@mui/material";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import CloseIcon from "@mui/icons-material/Close";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import NewspaperIcon from "@mui/icons-material/Newspaper";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import ShowChartIcon from "@mui/icons-material/ShowChart";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import ZoomOutMapIcon from "@mui/icons-material/ZoomOutMap";
import { useMutation, useQuery } from "@tanstack/react-query";

import {
  ApiError,
  ApiUnavailableError,
  buildEventKey,
  createJitForecast,
  createJitForecastFromFront,
  getActiveSpreadModelId,
  getReverseGeocode
} from "../api/client";
import { useAppStore } from "../state/store";
import type { FireEvent, ReverseGeocodeResponse } from "../types/api";
import { haversineKm } from "../utils/geo";
import { forecastButtonState } from "../utils/forecast";
import { comparePriorityFeedEvents } from "../utils/priorityFeed";

interface FireDetailsPanelProps {
  visibleEvents: FireEvent[];
}

interface GdeltArticle {
  title: string;
  url: string;
  socialimage?: string;
  seendate: string;
  sourcecountry?: string;
}

const STRONG_WILDFIRE_TERMS = [
  "wildfire", "wildfires", "bushfire", "bushfires", "forest fire", "forest fires",
  "brush fire", "brush fires", "grass fire", "grass fires",
  "fire evacuation", "fire evacuations", "fire season", "acres burned",
  "fire containment", "fire crews", "firefighter", "firefighters",
  "prescribed burn", "prescribed fire", "controlled burn",
  "fire weather", "red flag warning", "structure fire", "fire behavior",
  "fire perimeter", "fire spread", "fire retardant", "air tanker"
];

const FIRE_CONTEXT_TERMS = [
  "evacuate", "evacuation", "blaze", "flames", "contained", "containment",
  "smoke", "acres", "crews", "perimeter", "hotspot", "embers",
  "arson", "drought", "fire line", "backfire", "torching", "spotting"
];

const EXCLUDE_TERMS = [
  "gunfire", "ceasefire", "cease-fire", "opens fire", "open fire",
  "fired on", "fired at", "under fire", "crossfire", "hail of fire",
  "fire sale", "fired from", "firing squad", "return fire", "friendly fire",
  "rapid fire", "spitfire", "fire someone", "fired over", "drew fire",
  "facing fire", "political fire", "israel", "gaza", "ukraine", "russia",
  "shooting", "gunman", "military", "soldier", "missile", "bomb"
];

function isWildfireArticle(title: string): boolean {
  const lower = title.toLowerCase();
  if (EXCLUDE_TERMS.some((term) => lower.includes(term))) return false;
  if (STRONG_WILDFIRE_TERMS.some((term) => lower.includes(term))) return true;
  if (lower.includes("fire") && FIRE_CONTEXT_TERMS.some((term) => lower.includes(term))) return true;
  return false;
}

const HIGH_CONFIDENCE_THRESHOLD = 0.6;

function safeNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function severity(event: FireEvent): number {
  const score = safeNumber(event.event_score);
  if (score === null) return 0;
  return Math.max(0, Math.min(score, 1));
}

function coordinateKey(lat: number | null, lon: number | null): string | null {
  if (lat === null || lon === null) {
    return null;
  }
  return `${lat.toFixed(4)},${lon.toFixed(4)}`;
}

function hasDirectLocation(event: FireEvent): boolean {
  const candidates = [event.location_name, event.region_name, event.admin1_name, event.admin0_name, event.country];
  return candidates.some((candidate) => typeof candidate === "string" && candidate.trim().length > 0);
}

function locationLabel(event: FireEvent, resolved?: ReverseGeocodeResponse | null): string {
  const candidates = [
    resolved?.location_name,
    event.location_name,
    event.region_name,
    resolved?.admin1_name,
    event.admin1_name,
    event.admin0_name,
    resolved?.country,
    event.country,
    resolved?.display_name
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim();
    }
  }
  return "Unresolved location";
}

function confidenceLabel(event: FireEvent): "High" | "Nominal" {
  return severity(event) >= HIGH_CONFIDENCE_THRESHOLD ? "High" : "Nominal";
}

function formattedTime(value: unknown): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    return "n/a";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "n/a";
  }
  return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

interface IntensityDescriptor {
  label: string;
  value: number;
  unit: "MW" | "K";
}

function primaryIntensity(event: FireEvent): IntensityDescriptor | null {
  const frpMax = safeNumber(event.frp_max);
  if (frpMax !== null) {
    return { label: "Peak FRP", value: frpMax, unit: "MW" };
  }

  const frpMean = safeNumber(event.frp_mean);
  if (frpMean !== null) {
    return { label: "Mean FRP", value: frpMean, unit: "MW" };
  }

  const brightnessMax = safeNumber(event.brightness_max);
  if (brightnessMax !== null) {
    return { label: "Peak Brightness", value: brightnessMax, unit: "K" };
  }

  const brightnessMean = safeNumber(event.brightness_mean);
  if (brightnessMean !== null) {
    return { label: "Mean Brightness", value: brightnessMean, unit: "K" };
  }

  return null;
}

function formatIntensity(value: number, unit: "MW" | "K"): string {
  if (!Number.isFinite(value)) {
    return `n/a ${unit}`;
  }
  if (unit === "MW") {
    return `${value.toFixed(2)} ${unit}`;
  }
  return `${value.toFixed(1)} ${unit}`;
}

function frpHumanLabel(frpMw: number): string {
  if (frpMw >= 500) return "Extreme Intensity / Rapid Spread";
  if (frpMw >= 100) return "Intense Fire Activity";
  if (frpMw >= 10)  return "Moderate Activity";
  return "Smoldering / Low Intensity";
}

function riskTierFromScore(score: number): { label: string; color: string } {
  if (score >= 0.75) return { label: "Critical", color: "#ef4444" };
  if (score >= 0.5)  return { label: "High",     color: "#f97316" };
  if (score >= 0.25) return { label: "Moderate", color: "#eab308" };
  return { label: "Low", color: "#22c55e" };
}

function observationSummary(event: FireEvent): string {
  if (event.review_required) {
    return "This event is flagged for analyst review. Treat the perimeter and intensity as provisional until verified.";
  }
  const time = typeof event.start_time === "string" && event.start_time.trim().length > 0
    ? ` at ${formattedTime(event.start_time)}`
    : "";
  const provenance = String(event.geom_source || "").toLowerCase() === "authoritative"
    ? "Authoritative perimeter from official source."
    : "Perimeter is estimated from detection cluster.";
  const fronts = Number(event.front_count || 0);
  const frontStr = fronts === 1 ? "1 active front tracked." : fronts > 1 ? `${fronts} active fronts tracked.` : "No fronts tracked yet.";
  return `Satellite thermal anomaly detected${time}. ${provenance} ${frontStr}`;
}

function satelliteLabel(source?: string | null, sensor?: string | null): string {
  const s = `${source || ""} ${sensor || ""}`.toUpperCase();
  if (s.includes("VIIRS") && (s.includes("NOAA20") || s.includes("NOAA-20"))) return "VIIRS · NOAA-20";
  if (s.includes("VIIRS") && (s.includes("SNPP") || s.includes("NPP"))) return "VIIRS · Suomi-NPP";
  if (s.includes("MODIS") && s.includes("TERRA")) return "MODIS · Terra";
  if (s.includes("MODIS") && s.includes("AQUA")) return "MODIS · Aqua";
  if (s.includes("CLUSTER") || s.includes("AGGREGATED")) return "Multi-sensor cluster";
  return [source, sensor].filter(Boolean).join(" · ") || "Unknown sensor";
}

function geometryProvenanceLabel(event: FireEvent): "Authoritative perimeter" | "Estimated perimeter" {
  return String(event.geom_source || "").toLowerCase() === "authoritative"
    ? "Authoritative perimeter"
    : "Estimated perimeter";
}

function formatSeenDate(seendate: string): string {
  return new Date(
    seendate.replace(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z/, "$1-$2-$3T$4:$5:$6Z")
  ).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function NewsCard({ item, expanded = false }: { item: GdeltArticle; expanded?: boolean }): JSX.Element {
  return (
    <Box
      component="a"
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      sx={{
        display: "block",
        flexShrink: 0,
        border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 2.5,
        overflow: "hidden",
        textDecoration: "none",
        transition: "all 160ms ease",
        bgcolor: "rgba(22,27,34,0.5)",
        "&:hover": { borderColor: "rgba(249,115,22,0.3)", bgcolor: "#1c2128" }
      }}
    >
      {item.socialimage && (
        <Box sx={{ position: "relative", height: expanded ? 140 : 110, overflow: "hidden" }}>
          <Box
            component="img"
            src={item.socialimage}
            alt=""
            sx={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              filter: "grayscale(100%)",
              transition: "filter 500ms ease",
              "&:hover": { filter: "grayscale(0%)" }
            }}
          />
          {item.sourcecountry && (
            <Box sx={{
              position: "absolute", top: 6, left: 6, px: 0.75, py: 0.25,
              bgcolor: "rgba(0,0,0,0.65)", backdropFilter: "blur(8px)",
              borderRadius: 0.75, border: "1px solid rgba(255,255,255,0.1)"
            }}>
              <Typography sx={{ fontSize: 8, fontWeight: 700, color: "#fff", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                {item.sourcecountry}
              </Typography>
            </Box>
          )}
        </Box>
      )}
      <Box sx={{ p: 1.5 }}>
        <Typography sx={{
          fontSize: expanded ? 12 : 11, fontWeight: 700, color: "#d1d5db", lineHeight: 1.4,
          display: "-webkit-box", WebkitLineClamp: expanded ? 3 : 2,
          WebkitBoxOrient: "vertical", overflow: "hidden"
        }}>
          {item.title}
        </Typography>
        <Box sx={{ mt: 1, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <Typography sx={{ fontSize: 9, fontWeight: 700, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            {formatSeenDate(item.seendate)}
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.4 }}>
            <Typography sx={{ fontSize: 9, fontWeight: 700, color: "rgba(249,115,22,0.5)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
              Read
            </Typography>
            <OpenInNewIcon sx={{ fontSize: 9, color: "rgba(249,115,22,0.5)" }} />
          </Box>
        </Box>
      </Box>
    </Box>
  );
}

function NewsExpandedModal({ open, articles, onClose }: { open: boolean; articles: GdeltArticle[]; onClose: () => void }): JSX.Element {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: "#0d1117",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 3,
          boxShadow: "0 32px 100px rgba(0,0,0,0.6)",
          maxHeight: "85vh"
        }
      }}
    >
      <DialogTitle sx={{ px: 3, py: 2, borderBottom: "1px solid rgba(255,255,255,0.05)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <NewspaperIcon sx={{ fontSize: 16, color: "#f97316" }} />
          <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#fff", letterSpacing: "0.1em", textTransform: "uppercase" }}>
            Ground Reports
          </Typography>
          <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#6b7280", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            · {articles.length} reports · last 12h
          </Typography>
        </Box>
        <IconButton onClick={onClose} size="small" sx={{ color: "#6b7280", "&:hover": { color: "#fff" } }}>
          <CloseIcon sx={{ fontSize: 16 }} />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ p: 3 }}>
        <Box sx={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
          gap: 2,
          pt: 0.5
        }}>
          {articles.map((item, idx) => (
            <NewsCard key={idx} item={item} expanded />
          ))}
          {articles.length === 0 && (
            <Typography sx={{ fontSize: 13, color: "#6b7280", gridColumn: "1 / -1" }}>
              No wildfire reports found.
            </Typography>
          )}
        </Box>
      </DialogContent>
    </Dialog>
  );
}

export default function FireDetailsPanel({ visibleEvents }: FireDetailsPanelProps): JSX.Element {
  const selectedEvent = useAppStore((s) => s.selectedEvent);
  const setSelectedEvent = useAppStore((s) => s.setSelectedEvent);
  const setLastClick = useAppStore((s) => s.setLastClick);
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);
  const frontIndex = useAppStore((s) => s.frontIndexByEvent);
  const forecast = useAppStore((s) => s.forecast);
  const startForecastJob = useAppStore((s) => s.startForecastJob);
  const setForecastNotification = useAppStore((s) => s.setForecastNotification);
  const safety = useAppStore((s) => s.safety);
  const requestAssistantBriefing = useAppStore((s) => s.requestAssistantBriefing);
  const isSafetyMode = safety.enabled;

  const [submitError, setSubmitError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"telemetry" | "news">("telemetry");
  const [newsExpanded, setNewsExpanded] = useState(false);
  const [resolvedGeocodes, setResolvedGeocodes] = useState<Record<string, ReverseGeocodeResponse>>({});
  const resolvedGeocodesRef = useRef<Record<string, ReverseGeocodeResponse>>({});
  const geocodeInFlightRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    resolvedGeocodesRef.current = resolvedGeocodes;
  }, [resolvedGeocodes]);

  const topFires = useMemo(() => {
    return [...visibleEvents]
      .sort(comparePriorityFeedEvents)
      .slice(0, 5);
  }, [visibleEvents]);

  const { data: newsData, isLoading: newsLoading, isError: newsError } = useQuery({
    queryKey: ["gdelt-news"],
    queryFn: async () => {
      const query = encodeURIComponent(
        "(wildfire OR bushfire OR \"forest fire\" OR \"brush fire\" OR \"grass fire\" OR firefighter OR \"fire evacuation\" OR \"fire season\" OR \"fire crews\" OR \"prescribed burn\" OR \"controlled burn\" OR \"fire weather\" OR \"red flag warning\") sourcelang:english"
      );
      const url = `https://api.gdeltproject.org/api/v2/doc/doc?query=${query}&mode=artlist&format=json&timespan=12h&sort=datedesc&maxrecords=75`;
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to fetch news");
      const json = await res.json() as { articles?: GdeltArticle[] };
      return (json.articles ?? []).filter((a) => isWildfireArticle(a.title));
    },
    staleTime: 5 * 60 * 1000,
    refetchInterval: 10 * 60 * 1000,
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
        if (cancelled) {
          return;
        }
        await resolveLocation(event);
      }
    };

    void hydrateTopFires();
    return () => {
      cancelled = true;
    };
  }, [resolveLocation, topFires]);

  useEffect(() => {
    if (!selectedEvent) {
      return;
    }
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
      const location = locationLabel(event, geocoded);
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
          <>
            <Box sx={{ px: 2.25, py: 1.2, borderBottom: "1px solid rgba(255,255,255,0.05)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#6b7280", letterSpacing: "0.1em", textTransform: "uppercase" }}>
                {topFires.length} in view
              </Typography>
            </Box>
            <Box sx={{ p: 2.25, overflowY: "auto", display: "flex", flexDirection: "column", gap: 1.2 }}>
              {topFires.map((event, index) => {
                const lat = safeNumber(event.lat);
                const lon = safeNumber(event.lon);
                const key = coordinateKey(lat, lon);
                const geocoded = key ? resolvedGeocodes[key] : null;
                const loc = locationLabel(event, geocoded);
                const intensity = primaryIntensity(event);
                const canSelect = lat !== null && lon !== null;

                return (
                  <Box
                    key={`${String(event.event_id || "event")}-${index}`}
                    component="button"
                    disabled={!canSelect}
                    onClick={() => {
                      if (lat === null || lon === null) {
                        return;
                      }
                      setSelectedEvent({ ...event, lat, lon });
                      setLastClick({ lat, lng: lon });
                      focusMapOnPoint(lat, lon, 5.5);
                    }}
                    sx={{
                      textAlign: "left",
                      border: "1px solid rgba(255,255,255,0.06)",
                      borderRadius: 2.5,
                      background: "rgba(22,27,34,0.5)",
                      color: "inherit",
                      p: 1.6,
                      cursor: canSelect ? "pointer" : "not-allowed",
                      transition: "all 160ms ease",
                      opacity: canSelect ? 1 : 0.55,
                      '&:hover': canSelect
                        ? {
                            borderColor: "rgba(249,115,22,0.34)",
                            backgroundColor: "#1c2128"
                          }
                        : undefined
                    }}
                  >
                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 0.75 }}>
                      <Typography sx={{ fontSize: 10, color: "#6b7280", letterSpacing: "0.12em", textTransform: "uppercase", fontWeight: 700, pr: 1 }}>
                        {loc}
                      </Typography>
                      <Typography sx={{ fontSize: 10, color: "#4b5563", fontWeight: 700 }}>
                        {formattedTime(event.end_time)}
                      </Typography>
                    </Box>

                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.9 }}>
                        <Box
                          sx={{
                            width: 7,
                            height: 7,
                            borderRadius: "50%",
                            bgcolor: confidenceLabel(event) === "High" ? "#f97316" : "#6b7280",
                            boxShadow: confidenceLabel(event) === "High" ? "0 0 8px rgba(249,115,22,0.45)" : undefined
                          }}
                        />
                        <Typography sx={{ fontSize: 12, color: "#fff", fontWeight: 700 }}>
                          {intensity ? `${intensity.label.toUpperCase()}: ${formatIntensity(intensity.value, intensity.unit)}` : "INTENSITY: n/a"}
                        </Typography>
                      </Box>
                      <ChevronRightIcon sx={{ fontSize: 16, color: "#4b5563" }} />
                    </Box>
                  </Box>
                );
              })}

              {topFires.length === 0 && (
                <Typography sx={{ fontSize: 13, color: "#6b7280" }}>
                  No events match the current viewport and filter settings.
                </Typography>
              )}
            </Box>
          </>
        )}

        {activeTab === "news" && (
          <Box sx={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
            {/* News header */}
            <Box sx={{ px: 2.25, py: 1.2, borderBottom: "1px solid rgba(255,255,255,0.05)", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
              <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#6b7280", letterSpacing: "0.1em", textTransform: "uppercase" }}>
                {newsLoading ? "Loading…" : `${(newsData ?? []).length} reports`}
              </Typography>
              <IconButton
                size="small"
                onClick={() => setNewsExpanded(true)}
                disabled={newsLoading || (newsData ?? []).length === 0}
                sx={{ color: "#4b5563", "&:hover": { color: "#9ca3af" }, p: 0.5 }}
              >
                <ZoomOutMapIcon sx={{ fontSize: 13 }} />
              </IconButton>
            </Box>
            {/* Scrollable list */}
            <Box sx={{ flex: 1, overflowY: "auto", p: 2.25, display: "flex", flexDirection: "column", gap: 1.5 }}>
              {newsLoading && (
                <Typography sx={{ fontSize: 12, color: "#6b7280" }}>Loading ground reports...</Typography>
              )}
              {newsError && (
                <Typography sx={{ fontSize: 12, color: "#ef4444" }}>Failed to load news. Check your connection.</Typography>
              )}
              {!newsLoading && !newsError && (newsData ?? []).length === 0 && (
                <Typography sx={{ fontSize: 12, color: "#6b7280" }}>No wildfire reports in the last 12 hours.</Typography>
              )}
              {(newsData ?? []).map((item, idx) => (
                <NewsCard key={idx} item={item} />
              ))}
            </Box>
          </Box>
        )}

        {/* Expanded news modal */}
        <NewsExpandedModal
          open={newsExpanded}
          articles={newsData ?? []}
          onClose={() => setNewsExpanded(false)}
        />
      </Box>
    );
  }

  const lat = safeNumber(selectedEvent.lat);
  const lon = safeNumber(selectedEvent.lon);
  const eventKey = lat !== null && lon !== null ? buildEventKey(selectedEvent, lat, lon) : "";
  const sameEventCompleted = Boolean(forecast.lastForecast?.run.id && forecast.lastForecast?.eventKey === eventKey);
  const button = forecastButtonState({
    forecastRunning: Boolean(forecast.jobId),
    sameEventCompleted
  });

  const selectedKey = coordinateKey(lat, lon);
  const selectedGeocoded = selectedKey ? resolvedGeocodes[selectedKey] : null;
  const loc = locationLabel(selectedEvent, selectedGeocoded);
  const intensity = primaryIntensity(selectedEvent);
  const score = severity(selectedEvent);
  const provenance = geometryProvenanceLabel(selectedEvent);

  // Safety-mode derived values
  const riskTier = riskTierFromScore(score);
  const distanceToFireKm = isSafetyMode && safety.userLocation && lat !== null && lon !== null
    ? haversineKm(safety.userLocation.lat, safety.userLocation.lon, lat, lon)
    : null;
  const isNearby = isSafetyMode && (safety.safetyTier === 'DANGER' || safety.safetyTier === 'WARNING');
  const frpHuman = intensity?.unit === "MW" ? frpHumanLabel(intensity.value) : null;

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

      <Box sx={{ p: 2.25, overflowY: "auto", display: "flex", flexDirection: "column", gap: 1.5 }}>
        {/* Safety Mode: threat zone banner */}
        {isNearby && (
          <Box sx={{ mx: -2.25, mt: -2.25, mb: 0, px: 2.25, py: 1, bgcolor: "rgba(239,68,68,0.14)", borderBottom: "2px solid #ef4444" }}>
            <Typography sx={{ fontSize: 11, fontWeight: 900, color: "#fca5a5", letterSpacing: "0.14em" }}>
              {safety.safetyTier === 'DANGER' ? "⚠ DANGER ZONE" : "⚠ WATCH ZONE"}
              {distanceToFireKm !== null ? ` — ${distanceToFireKm.toFixed(1)} km away` : ""}
            </Typography>
          </Box>
        )}

        {submitError && <Alert severity="error">{submitError}</Alert>}

        {/* Safety Mode: "Get Safety Info" is the primary action */}
        {isSafetyMode && (
          <Button
            variant="contained"
            onClick={() => requestAssistantBriefing(
              `Give a safety briefing for this fire. Focus on: risk level, distance from user (${distanceToFireKm !== null ? distanceToFireKm.toFixed(1) + ' km' : 'unknown distance'}), and immediate actions. 2-3 sentences, plain language only.`
            )}
            sx={{
              alignSelf: "flex-start",
              bgcolor: "#ef4444",
              color: "#fff",
              '&:hover': { bgcolor: "#dc2626" }
            }}
          >
            Get Safety Info
          </Button>
        )}

        {/* Analyst Mode / Safety Mode: forecast button (demoted in safety mode) */}
        {!isSafetyMode && (
          <Button
            variant="contained"
            disabled={button.disabled || forecastMutation.isPending || lat === null || lon === null}
            onClick={() => forecastMutation.mutate(selectedEvent)}
            sx={{
              alignSelf: "flex-start",
              bgcolor: "#f97316",
              color: "#fff",
              '&:hover': { bgcolor: "#ea580c" },
              '&.Mui-disabled': { bgcolor: "rgba(249,115,22,0.4)", color: "rgba(255,255,255,0.8)" }
            }}
          >
            {button.label}
          </Button>
        )}

        {!isSafetyMode && button.reason && (
          <Typography sx={{ fontSize: 11, color: "#6b7280" }}>
            {button.reason}
          </Typography>
        )}

        <Box sx={{ p: 2, bgcolor: "#161b22", borderRadius: 2.5, border: `1px solid ${isSafetyMode ? "rgba(239,68,68,0.16)" : "rgba(249,115,22,0.16)"}` }}>
          <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 1.5, mb: 1.5 }}>
            <Box>
              <Typography sx={{ fontSize: 10, color: isSafetyMode ? "#ef4444" : "#f97316", fontWeight: 800, letterSpacing: "0.14em", textTransform: "uppercase", mb: 0.5 }}>
                {isSafetyMode ? "Threat Assessment" : "Event Selected"}
              </Typography>
              <Typography sx={{ fontSize: 21, fontWeight: 800, color: "#fff", lineHeight: 1.1 }}>{loc}</Typography>
              {/* Safety Mode: risk tier chip shown prominently */}
              {isSafetyMode && (
                <Box
                  sx={{
                    mt: 0.75,
                    display: "inline-flex",
                    alignItems: "center",
                    px: 0.9,
                    py: 0.3,
                    borderRadius: 999,
                    bgcolor: `${riskTier.color}20`,
                    border: `1px solid ${riskTier.color}60`,
                    color: riskTier.color,
                    fontSize: 10,
                    fontWeight: 900,
                    letterSpacing: "0.1em",
                    textTransform: "uppercase"
                  }}
                >
                  {riskTier.label} Risk
                </Box>
              )}
              {!isSafetyMode && (
                <Typography
                  sx={{
                    mt: 0.75,
                    display: "inline-flex",
                    alignItems: "center",
                    px: 0.9,
                    py: 0.3,
                    borderRadius: 999,
                    bgcolor: provenance === "Authoritative perimeter" ? "rgba(34,197,94,0.18)" : "rgba(59,130,246,0.2)",
                    border: provenance === "Authoritative perimeter" ? "1px solid rgba(34,197,94,0.45)" : "1px solid rgba(59,130,246,0.5)",
                    color: provenance === "Authoritative perimeter" ? "#86efac" : "#93c5fd",
                    fontSize: 10,
                    fontWeight: 800,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase"
                  }}
                >
                  {provenance}
                </Typography>
              )}
            </Box>
            <Box sx={{ textAlign: "right" }}>
              {isSafetyMode && frpHuman ? (
                <>
                  <Typography sx={{ fontSize: 13, fontWeight: 900, color: "#fff", lineHeight: 1.2, maxWidth: 130, textAlign: "right" }}>
                    {frpHuman}
                  </Typography>
                  <Box
                    component="details"
                    sx={{ mt: 0.5, cursor: "pointer", "& summary": { fontSize: 9, color: "#6b7280", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", userSelect: "none", listStyle: "none", "&::-webkit-details-marker": { display: "none" } } }}
                  >
                    <Box component="summary">Technical Details ▾</Box>
                    <Typography sx={{ fontSize: 11, color: "#9ca3af", mt: 0.3 }}>
                      {formatIntensity(intensity!.value, intensity!.unit)}
                    </Typography>
                  </Box>
                </>
              ) : (
                <>
                  <Typography sx={{ fontSize: 30, fontWeight: 900, color: "#fff", lineHeight: 0.95 }}>
                    {intensity ? formatIntensity(intensity.value, intensity.unit) : "n/a"}
                  </Typography>
                  <Typography sx={{ fontSize: 10, color: "#4b5563", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", mt: 0.4 }}>
                    {intensity ? intensity.label : "Fire Intensity"}
                  </Typography>
                </>
              )}
            </Box>
          </Box>

          <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, mb: 1.35 }}>
            <Box sx={{ p: 1.2, bgcolor: "#0d1117", borderRadius: 1.7, border: "1px solid rgba(255,255,255,0.08)" }}>
              <Typography sx={{ fontSize: 10, color: "#4b5563", fontWeight: 800, letterSpacing: "0.1em", textTransform: "uppercase", mb: 0.4 }}>
                Confidence
              </Typography>
              <Typography sx={{ fontSize: 12, fontWeight: 800, color: confidenceLabel(selectedEvent) === "High" ? "#4ade80" : "#facc15" }}>
                {confidenceLabel(selectedEvent).toUpperCase()}
              </Typography>
            </Box>
            <Box sx={{ p: 1.2, bgcolor: "#0d1117", borderRadius: 1.7, border: "1px solid rgba(255,255,255,0.08)" }}>
              <Typography sx={{ fontSize: 10, color: "#4b5563", fontWeight: 800, letterSpacing: "0.1em", textTransform: "uppercase", mb: 0.4 }}>
                Event Score
              </Typography>
              <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#fff" }}>
                {score.toFixed(3)}
              </Typography>
            </Box>
          </Box>

          <Box sx={{ display: "flex", alignItems: "center", gap: 1, color: "#6b7280", fontFamily: "monospace", fontSize: 11 }}>
            <QueryStatsIcon sx={{ fontSize: 12 }} />
            <Typography component="span" sx={{ fontSize: 11, fontFamily: "inherit", color: "inherit" }}>
              LOC: {lat !== null ? lat.toFixed(4) : "n/a"}, {lon !== null ? lon.toFixed(4) : "n/a"}
            </Typography>
          </Box>
        </Box>

        <Stack spacing={1.1}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.85 }}>
            <InfoOutlinedIcon sx={{ fontSize: 14, color: "#60a5fa" }} />
            <Typography sx={{ fontSize: 10, color: "#fff", fontWeight: 800, letterSpacing: "0.12em", textTransform: "uppercase" }}>
              Observation
            </Typography>
          </Box>
          <Typography sx={{ fontSize: 12, color: "#9ca3af", lineHeight: 1.65, p: 1.6, borderRadius: 2.4, border: "1px solid rgba(255,255,255,0.06)", bgcolor: "rgba(22,27,34,0.55)", fontStyle: "italic" }}>
            {observationSummary(selectedEvent)}
          </Typography>
        </Stack>

        <Divider sx={{ borderColor: "rgba(255,255,255,0.08)" }} />

        <Stack spacing={1}>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <Typography sx={{ fontSize: 11, color: "#6b7280", fontWeight: 700 }}>First detected</Typography>
            <Typography sx={{ fontSize: 11, color: "#e5e7eb" }}>{formattedTime(selectedEvent.start_time)}</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <Typography sx={{ fontSize: 11, color: "#6b7280", fontWeight: 700 }}>Satellite</Typography>
            <Typography sx={{ fontSize: 11, color: "#e5e7eb" }}>
              {satelliteLabel(selectedEvent.source, selectedEvent.sensor)}
            </Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <Typography sx={{ fontSize: 11, color: "#6b7280", fontWeight: 700 }}>Active fronts</Typography>
            <Typography sx={{ fontSize: 11, color: "#e5e7eb" }}>{Number(selectedEvent.front_count || 0)}</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <Typography sx={{ fontSize: 11, color: "#6b7280", fontWeight: 700 }}>Perimeter</Typography>
            <Typography sx={{ fontSize: 11, color: "#e5e7eb" }}>{geometryProvenanceLabel(selectedEvent)}</Typography>
          </Box>

          {selectedEvent.review_required && (
            <Box sx={{ mt: 0.8, px: 1.25, py: 1, border: "1px solid rgba(250,204,21,0.3)", bgcolor: "rgba(250,204,21,0.08)", borderRadius: 1.6, display: "flex", alignItems: "center", gap: 0.85 }}>
              <WarningAmberIcon sx={{ fontSize: 14, color: "#facc15" }} />
              <Typography sx={{ fontSize: 11, color: "#fcd34d", fontWeight: 700 }}>
                Analyst review required — perimeter and intensity are provisional.
              </Typography>
            </Box>
          )}

          {forecast.lastForecast?.runMeta && forecast.lastForecast.runMeta.weatherRunId === null && (
            <Box sx={{ mt: 0.8, px: 1.25, py: 1, border: "1px solid rgba(251,146,60,0.3)", bgcolor: "rgba(251,146,60,0.08)", borderRadius: 1.6, display: "flex", alignItems: "flex-start", gap: 0.85 }}>
              <WarningAmberIcon sx={{ fontSize: 14, color: "#fb923c", mt: 0.1 }} />
              <Typography sx={{ fontSize: 11, color: "#fdba74", fontWeight: 600, lineHeight: 1.5 }}>
                Spread forecast assumed calm conditions — no weather data was available for this area and time. The symmetric shape reflects this, not actual wind direction.
              </Typography>
            </Box>
          )}
        </Stack>

        {/* Safety Mode: forecast button demoted to bottom */}
        {isSafetyMode && (
          <Button
            variant="outlined"
            size="small"
            disabled={button.disabled || forecastMutation.isPending || lat === null || lon === null}
            onClick={() => forecastMutation.mutate(selectedEvent)}
            sx={{
              alignSelf: "flex-start",
              borderColor: "rgba(249,115,22,0.4)",
              color: "#f97316",
              fontSize: 10,
              '&:hover': { borderColor: "#f97316", bgcolor: "rgba(249,115,22,0.08)" },
              '&.Mui-disabled': { borderColor: "rgba(249,115,22,0.2)", color: "rgba(249,115,22,0.4)" }
            }}
          >
            {button.label}
          </Button>
        )}
      </Box>
    </Box>
  );
}
