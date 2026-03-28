import { Alert, Box, Button, Divider, Stack, Typography } from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import type { FireEvent, ReverseGeocodeResponse } from "../../types/api";
import { haversineKm } from "../../utils/geo";
import { forecastButtonState } from "../../utils/forecast";
import { useAppStore } from "../../state/store";
import {
  safeNumber,
  coordinateKey,
  locationLabel,
  confidenceLabel,
  severity,
  primaryIntensity,
  formatIntensity,
  frpHumanLabel,
  riskTierFromScore,
  observationSummary,
  satelliteLabel,
  geometryProvenanceLabel
} from "./types";
import { ForecastPanel } from "./ForecastPanel";
import { QAReviewPanel } from "./QAReviewPanel";

interface ForecastMutationArgs {
  mutate: (event: FireEvent) => void;
  isPending: boolean;
}

interface FireFrontsTabProps {
  selectedEvent: FireEvent;
  resolvedGeocodes: Record<string, ReverseGeocodeResponse>;
  submitError: string | null;
  forecastMutation: ForecastMutationArgs;
}

export function FireFrontsTab({ selectedEvent, resolvedGeocodes, submitError, forecastMutation }: FireFrontsTabProps): JSX.Element {
  const forecast = useAppStore((s) => s.forecast);
  const safety = useAppStore((s) => s.safety);
  const requestAssistantBriefing = useAppStore((s) => s.requestAssistantBriefing);
  const archive = useAppStore((s) => s.archive);

  const isArchiveMode = archive.viewMode === "archive";
  const isSafetyMode = safety.enabled;

  const lat = safeNumber(selectedEvent.lat);
  const lon = safeNumber(selectedEvent.lon);
  const eventKey = lat !== null && lon !== null
    ? (selectedEvent.event_id && String(selectedEvent.event_id).trim().length > 0
      ? `event_id:${selectedEvent.event_id}`
      : `point:${lat.toFixed(4)}:${lon.toFixed(4)}:${String(selectedEvent.end_time || "")}`)
    : "";

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

  const riskTier = riskTierFromScore(score);
  const distanceToFireKm = isSafetyMode && safety.userLocation && lat !== null && lon !== null
    ? haversineKm(safety.userLocation.lat, safety.userLocation.lon, lat, lon)
    : null;
  const isNearby = isSafetyMode && (safety.safetyTier === 'DANGER' || safety.safetyTier === 'WARNING');
  const frpHuman = intensity?.unit === "MW" ? frpHumanLabel(intensity.value) : null;
  const runMeta = forecast.lastForecast?.runMeta ?? null;

  return (
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

      {/* Non-safety, non-archive: primary forecast button */}
      {!isSafetyMode && !isArchiveMode && (
        <ForecastPanel
          event={selectedEvent}
          button={button}
          isPending={forecastMutation.isPending}
          lat={lat}
          lon={lon}
          isSafetyMode={false}
          isArchiveMode={false}
          onRequestForecast={forecastMutation.mutate}
        />
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
          <Typography sx={{ fontSize: 11, color: "#e5e7eb" }}>{
            typeof selectedEvent.start_time === "string" && selectedEvent.start_time.trim().length > 0
              ? new Date(selectedEvent.start_time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
              : "n/a"
          }</Typography>
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

        <QAReviewPanel event={selectedEvent} runMeta={runMeta} />
      </Stack>

      {/* Safety mode: forecast button demoted to bottom */}
      {isSafetyMode && !isArchiveMode && (
        <ForecastPanel
          event={selectedEvent}
          button={button}
          isPending={forecastMutation.isPending}
          lat={lat}
          lon={lon}
          isSafetyMode={true}
          isArchiveMode={false}
          onRequestForecast={forecastMutation.mutate}
        />
      )}
    </Box>
  );
}
