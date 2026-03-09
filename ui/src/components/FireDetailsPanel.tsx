import { useMemo, useState } from "react";
import { Alert, Box, Button, Chip, Divider, List, ListItem, ListItemText, Paper, Stack, Typography } from "@mui/material";
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

function locationLabel(event: FireEvent, lat: number, lon: number): string {
  const candidates = [event.country, event.admin0_name, event.admin1_name, event.region_name, event.location_name];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim();
    }
  }
  return `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
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

  const majorFires = useMemo(() => {
    return [...visibleEvents]
      .sort((a, b) => significance(b) - significance(a))
      .slice(0, 5);
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
      <Paper sx={{ p: 2, height: "100%", overflow: "auto" }}>
        <Typography variant="h6" gutterBottom>
          Overview
        </Typography>
        <Typography variant="h3" color="primary.main" lineHeight={1}>
          {visibleEvents.length}
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", letterSpacing: 0.8 }}>
          Active events
        </Typography>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle2" gutterBottom>
          Major fires in view
        </Typography>
        <Stack spacing={1}>
          {majorFires.map((event, index) => {
            const lat = safeNumber(event.lat);
            const lon = safeNumber(event.lon);
            const score = severity(event);
            const eventId = String(event.event_id || "unknown");

            return (
              <Button
                key={`${eventId}-${index}`}
                variant="outlined"
                onClick={() => {
                  if (lat === null || lon === null) {
                    return;
                  }
                  setSelectedEvent({ ...event, lat, lon });
                  setLastClick({ lat, lng: lon });
                  focusMapOnPoint(lat, lon, 5.5);
                }}
              >
                #{index + 1} {eventId} · score {score.toFixed(2)} · detections {Number(event.detection_count || 0)}
              </Button>
            );
          })}
          {majorFires.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              No events in the current viewport.
            </Typography>
          )}
        </Stack>
      </Paper>
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

  return (
    <Paper sx={{ p: 2, height: "100%", overflow: "auto" }}>
      <Stack spacing={1.5}>
        {submitError && <Alert severity="error">{submitError}</Alert>}

        <Button
          variant="contained"
          disabled={button.disabled || forecastMutation.isPending || lat === null || lon === null}
          onClick={() => forecastMutation.mutate(selectedEvent)}
        >
          {button.label}
        </Button>
        {button.reason && (
          <Typography variant="caption" color="text.secondary">
            {button.reason}
          </Typography>
        )}

        <Typography variant="h6">Fire details</Typography>
        <Typography variant="body2">Selection: Fire event</Typography>
        {lat !== null && lon !== null && (
          <Typography variant="body2" color="text.secondary">
            Location: {lat.toFixed(4)}, {lon.toFixed(4)}
          </Typography>
        )}

        <List dense disablePadding>
          <ListItem disableGutters>
            <ListItemText primary="Event ID" secondary={String(selectedEvent.event_id || "unknown")} />
          </ListItem>
          <ListItem disableGutters>
            <ListItemText primary="Source" secondary={String(selectedEvent.source || "unknown")} />
          </ListItem>
          <ListItem disableGutters>
            <ListItemText primary="Satellite" secondary={String(selectedEvent.sensor || "unknown")} />
          </ListItem>
          <ListItem disableGutters>
            <ListItemText primary="Window" secondary={`${String(selectedEvent.start_time || "n/a")} → ${String(selectedEvent.end_time || "n/a")}`} />
          </ListItem>
          <ListItem disableGutters>
            <ListItemText primary="Detections" secondary={String(selectedEvent.detection_count || 0)} />
          </ListItem>
          <ListItem disableGutters>
            <ListItemText primary="Fronts" secondary={String(selectedEvent.front_count || 0)} />
          </ListItem>
          <ListItem disableGutters>
            <ListItemText primary="Decision" secondary={String(selectedEvent.denoiser_decision || "unknown")} />
          </ListItem>
          <ListItem disableGutters>
            <ListItemText primary="Review required" secondary={selectedEvent.review_required ? "true" : "false"} />
          </ListItem>
        </List>

        <Divider />

        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Event score
          </Typography>
          <Chip
            color={severity(selectedEvent) >= 0.6 ? "error" : severity(selectedEvent) >= 0.4 ? "primary" : "warning"}
            label={severity(selectedEvent).toFixed(4)}
          />
        </Box>
      </Stack>
    </Paper>
  );
}
