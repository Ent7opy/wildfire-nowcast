import { Box, Typography } from "@mui/material";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import type { FireEvent, ReverseGeocodeResponse } from "../../types/api";
import {
  safeNumber,
  coordinateKey,
  locationLabel,
  confidenceLabel,
  formattedTime,
  primaryIntensity,
  formatIntensity
} from "./types";
import { useAppStore } from "../../state/store";

interface FireOverviewTabProps {
  topFires: FireEvent[];
  resolvedGeocodes: Record<string, ReverseGeocodeResponse>;
}

export function FireOverviewTab({ topFires, resolvedGeocodes }: FireOverviewTabProps): JSX.Element {
  const setSelectedEvent = useAppStore((s) => s.setSelectedEvent);
  const setLastClick = useAppStore((s) => s.setLastClick);
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);

  return (
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
                focusMapOnPoint(lat, lon, 8);
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
  );
}
