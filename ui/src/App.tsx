import { useEffect, useMemo, useState } from "react";
import {
  Box,
  Button,
  InputAdornment,
  Paper,
  TextField,
  Typography
} from "@mui/material";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import PublicIcon from "@mui/icons-material/Public";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import SearchIcon from "@mui/icons-material/Search";
import VerifiedIcon from "@mui/icons-material/Verified";

import AIChatAssistant from "./components/AIChatAssistant";
import DataFreshnessBanner from "./components/DataFreshnessBanner";
import FireDetailsPanel from "./components/FireDetailsPanel";
import FireMap from "./components/FireMap";
import ForecastNotification from "./components/ForecastNotification";
import { useForecastPolling } from "./hooks/useForecastPolling";
import { useUrlStateSync } from "./hooks/useUrlStateSync";
import { useAppStore } from "./state/store";
import type { FireEvent } from "./types/api";
import type {
  AssistantConfidenceFilter,
  AssistantViewEventSummary
} from "./types/state";

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

function matchesSearch(event: FireEvent, normalizedQuery: string): boolean {
  if (!normalizedQuery) return true;

  const haystack = [
    locationLabel(event),
    event.event_id,
    event.source,
    event.sensor,
    event.denoiser_decision
  ]
    .map((value) => String(value || "").toLowerCase())
    .join(" ");

  return haystack.includes(normalizedQuery);
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

export default function App(): JSX.Element {
  const [visibleEvents, setVisibleEvents] = useState<FireEvent[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [confidenceFilter, setConfidenceFilter] = useState<AssistantConfidenceFilter>("All");
  const setAssistantViewContext = useAppStore((s) => s.setAssistantViewContext);

  useUrlStateSync();
  useForecastPolling();

  const filteredEvents = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase();
    return visibleEvents.filter((event) => {
      if (confidenceFilter === "High" && !isHighConfidence(event)) {
        return false;
      }
      return matchesSearch(event, normalizedQuery);
    });
  }, [confidenceFilter, searchQuery, visibleEvents]);

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
      searchQuery,
      confidenceFilter,
      visibleEventCount: visibleEvents.length,
      filteredEventCount: filteredEvents.length,
      topEvents: topEventsForAssistant
    });
  }, [
    confidenceFilter,
    filteredEvents.length,
    searchQuery,
    setAssistantViewContext,
    topEventsForAssistant,
    visibleEvents.length
  ]);

  const stats = [
    {
      label: "Total Detections",
      value: formatStatValue(totalDetections),
      unit: "",
      icon: QueryStatsIcon,
      color: "#f97316"
    },
    {
      label: "Active Perimeters",
      value: formatStatValue(activePerimeters),
      unit: "",
      icon: LocalFireDepartmentIcon,
      color: "#ef4444"
    },
    {
      label: "Avg Event Score",
      value: averageScore === null ? "n/a" : formatStatValue(averageScore, 3),
      unit: "0-1",
      icon: PublicIcon,
      color: "#60a5fa"
    },
    {
      label: "High Confidence",
      value: confidencePercent === null ? "n/a" : formatStatValue(confidencePercent, 1),
      unit: "%",
      icon: VerifiedIcon,
      color: "#4ade80"
    }
  ] as const;

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
            <Box sx={{ display: "flex", gap: 1, mb: 1.25, flexWrap: "wrap" }}>
              <Box
                sx={{
                  px: 1,
                  py: 0.4,
                  borderRadius: 1,
                  bgcolor: "rgba(249,115,22,0.12)",
                  border: "1px solid rgba(249,115,22,0.25)",
                  color: "#f97316",
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: "0.14em",
                  textTransform: "uppercase"
                }}
              >
                Live Nowcast
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

            <Typography variant="h3" sx={{ color: "#fff", fontWeight: 800, lineHeight: 1.05, letterSpacing: "-0.02em", mb: 0.8 }}>
              Wildfire Nowcast
            </Typography>
            <Typography sx={{ color: "#6b7280", maxWidth: 650, fontSize: 14, lineHeight: 1.65 }}>
              Global thermal anomalies from NASA FIRMS with denoiser-scored event confidence and live spread context.
            </Typography>
          </Box>

          <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, flexWrap: "wrap" }}>
            <TextField
              size="small"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Filter regions..."
              sx={{
                minWidth: { xs: "100%", sm: 240 },
                "& .MuiOutlinedInput-root": {
                  bgcolor: "#0d1117",
                  borderRadius: 2,
                  fontSize: 12,
                  color: "#e5e7eb"
                }
              }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon sx={{ fontSize: 16, color: "#6b7280" }} />
                  </InputAdornment>
                )
              }}
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

        <DataFreshnessBanner />
        <ForecastNotification />

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
                searchQuery={searchQuery}
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
            <Box sx={{ flex: 1, minHeight: 320 }}>
              <AIChatAssistant />
            </Box>
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
