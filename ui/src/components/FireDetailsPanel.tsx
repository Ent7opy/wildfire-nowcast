import { useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Divider,
  IconButton,
  Stack,
  Typography
} from "@mui/material";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import CloseIcon from "@mui/icons-material/Close";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { useMutation } from "@tanstack/react-query";

import {
  ApiError,
  ApiUnavailableError,
  buildEventKey,
  createJitForecast,
  createJitForecastFromFront,
  getActiveSpreadModelId
} from "../api/client";
import { useAppStore } from "../state/store";
import type { FireEvent } from "../types/api";
import { forecastButtonState } from "../utils/forecast";

interface FireDetailsPanelProps {
  visibleEvents: FireEvent[];
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

function significance(event: FireEvent): number {
  const sev = severity(event);
  const detections = safeNumber(event.detection_count) ?? 0;
  const detectionsComponent = Math.max(0, Math.min(1, Math.log1p(detections) / Math.log1p(100)));
  let recency = 0;
  if (event.end_time) {
    const end = new Date(event.end_time);
    if (!Number.isNaN(end.getTime())) {
      const ageHours = Math.max((Date.now() - end.getTime()) / 3_600_000, 0);
      recency = Math.max(0, 1 - Math.min(ageHours, 24) / 24);
    }
  }
  return 0.65 * sev + 0.25 * detectionsComponent + 0.1 * recency;
}

function locationLabel(event: FireEvent, lat: number | null, lon: number | null): string {
  const candidates = [event.location_name, event.region_name, event.admin1_name, event.admin0_name, event.country];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim();
    }
  }
  if (lat !== null && lon !== null) {
    return `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
  }
  return "Unknown region";
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

function insightText(event: FireEvent): string {
  if (event.review_required) {
    return "This event is marked for analyst review. Treat the perimeter as provisional until verified.";
  }
  if (typeof event.denoiser_decision === "string" && event.denoiser_decision.trim().length > 0) {
    return `Denoiser decision is ${event.denoiser_decision}. Continue monitoring event score and fronts for escalation.`;
  }
  return "No denoiser decision is attached to this event snapshot.";
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

  const [submitError, setSubmitError] = useState<string | null>(null);

  const topFires = useMemo(() => {
    return [...visibleEvents]
      .sort((a, b) => significance(b) - significance(a))
      .slice(0, 10);
  }, [visibleEvents]);

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
      const location = locationLabel(event, lat, lon);
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
          minHeight: 360
        }}
      >
        <Box sx={{ px: 2.25, py: 1.6, borderBottom: "1px solid rgba(255,255,255,0.05)", bgcolor: "#161b22", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <Typography sx={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.14em", textTransform: "uppercase", color: "#fff", display: "flex", alignItems: "center", gap: 0.75 }}>
            <LocalFireDepartmentIcon sx={{ fontSize: 14, color: "#f97316" }} />
            Priority Feed
          </Typography>
          <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#6b7280", letterSpacing: "0.1em", textTransform: "uppercase" }}>
            {topFires.length} in view
          </Typography>
        </Box>

        <Box sx={{ p: 2.25, overflowY: "auto", display: "flex", flexDirection: "column", gap: 1.2 }}>
          {topFires.map((event, index) => {
            const lat = safeNumber(event.lat);
            const lon = safeNumber(event.lon);
            const loc = locationLabel(event, lat, lon);
            const score = severity(event);
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
                      SCORE: {score.toFixed(3)} • DETECTIONS: {Number(event.detection_count || 0)}
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

  const loc = locationLabel(selectedEvent, lat, lon);
  const score = severity(selectedEvent);

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
          <LocalFireDepartmentIcon sx={{ fontSize: 14, color: "#f97316" }} />
          Fire Inspector
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
        {submitError && <Alert severity="error">{submitError}</Alert>}

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

        {button.reason && (
          <Typography sx={{ fontSize: 11, color: "#6b7280" }}>
            {button.reason}
          </Typography>
        )}

        <Box sx={{ p: 2, bgcolor: "#161b22", borderRadius: 2.5, border: "1px solid rgba(249,115,22,0.16)" }}>
          <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 1.5, mb: 1.5 }}>
            <Box>
              <Typography sx={{ fontSize: 10, color: "#f97316", fontWeight: 800, letterSpacing: "0.14em", textTransform: "uppercase", mb: 0.5 }}>
                Event Selected
              </Typography>
              <Typography sx={{ fontSize: 21, fontWeight: 800, color: "#fff", lineHeight: 1.1 }}>{loc}</Typography>
            </Box>
            <Box sx={{ textAlign: "right" }}>
              <Typography sx={{ fontSize: 30, fontWeight: 900, color: "#fff", lineHeight: 0.95 }}>{score.toFixed(3)}</Typography>
              <Typography sx={{ fontSize: 10, color: "#4b5563", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", mt: 0.4 }}>
                Event Score
              </Typography>
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
                Detections
              </Typography>
              <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#fff" }}>
                {Number(selectedEvent.detection_count || 0)}
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
              Environment Insight
            </Typography>
          </Box>
          <Typography sx={{ fontSize: 12, color: "#9ca3af", lineHeight: 1.65, p: 1.6, borderRadius: 2.4, border: "1px solid rgba(255,255,255,0.06)", bgcolor: "rgba(22,27,34,0.55)", fontStyle: "italic" }}>
            {insightText(selectedEvent)}
          </Typography>
        </Stack>

        <Divider sx={{ borderColor: "rgba(255,255,255,0.08)" }} />

        <Stack spacing={1}>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <Typography sx={{ fontSize: 11, color: "#6b7280", fontWeight: 700 }}>Event ID</Typography>
            <Typography sx={{ fontSize: 11, color: "#e5e7eb", fontFamily: "monospace" }}>{String(selectedEvent.event_id || "unknown")}</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <Typography sx={{ fontSize: 11, color: "#6b7280", fontWeight: 700 }}>Source / Sensor</Typography>
            <Typography sx={{ fontSize: 11, color: "#e5e7eb" }}>
              {String(selectedEvent.source || "unknown")} • {String(selectedEvent.sensor || "unknown")}
            </Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <Typography sx={{ fontSize: 11, color: "#6b7280", fontWeight: 700 }}>Window</Typography>
            <Typography sx={{ fontSize: 11, color: "#e5e7eb" }}>
              {formattedTime(selectedEvent.start_time)} → {formattedTime(selectedEvent.end_time)}
            </Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <Typography sx={{ fontSize: 11, color: "#6b7280", fontWeight: 700 }}>Fronts</Typography>
            <Typography sx={{ fontSize: 11, color: "#e5e7eb" }}>{Number(selectedEvent.front_count || 0)}</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <Typography sx={{ fontSize: 11, color: "#6b7280", fontWeight: 700 }}>Decision</Typography>
            <Typography sx={{ fontSize: 11, color: "#e5e7eb" }}>{String(selectedEvent.denoiser_decision || "unknown")}</Typography>
          </Box>

          {selectedEvent.review_required && (
            <Box sx={{ mt: 0.8, px: 1.25, py: 1, border: "1px solid rgba(250,204,21,0.3)", bgcolor: "rgba(250,204,21,0.08)", borderRadius: 1.6, display: "flex", alignItems: "center", gap: 0.85 }}>
              <WarningAmberIcon sx={{ fontSize: 14, color: "#facc15" }} />
              <Typography sx={{ fontSize: 11, color: "#fcd34d", fontWeight: 700 }}>
                Review required flag is active for this event.
              </Typography>
            </Box>
          )}
        </Stack>
      </Box>
    </Box>
  );
}
