import { Box, Typography } from "@mui/material";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import type { WeatherWarningBrief, WarningSeverity, WarningType } from "../../types/api";

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

interface SeverityDisplay {
  label: string;
  color: string;
  bgColor: string;
  borderColor: string;
}

const SEVERITY_DISPLAY: Record<WarningSeverity, SeverityDisplay> = {
  red: {
    label: "RED",
    color: "#fca5a5",
    bgColor: "rgba(239,68,68,0.18)",
    borderColor: "rgba(239,68,68,0.55)",
  },
  orange: {
    label: "ORANGE",
    color: "#fdba74",
    bgColor: "rgba(249,115,22,0.15)",
    borderColor: "rgba(249,115,22,0.5)",
  },
  yellow: {
    label: "YELLOW",
    color: "#fde047",
    bgColor: "rgba(234,179,8,0.12)",
    borderColor: "rgba(234,179,8,0.45)",
  },
  green: {
    label: "GREEN",
    color: "#86efac",
    bgColor: "rgba(34,197,94,0.1)",
    borderColor: "rgba(34,197,94,0.35)",
  },
};

const WARNING_TYPE_LABELS: Record<WarningType, string> = {
  wind: "Wind",
  heat: "Extreme Heat",
  drought: "Drought / Forest Fire",
  thunderstorm: "Thunderstorm",
  rain: "Heavy Rain",
  snow: "Snow / Ice",
  fog: "Fog",
  other: "Weather Warning",
};

function formatTimeRemaining(expiresIso: string): string {
  const expires = new Date(expiresIso);
  const now = new Date();
  const diffMs = expires.getTime() - now.getTime();
  if (diffMs <= 0) return "expired";
  const diffH = Math.floor(diffMs / (1000 * 60 * 60));
  const diffM = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
  if (diffH > 0) return `${diffH}h ${diffM}m remaining`;
  return `${diffM}m remaining`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface WarningsBlockProps {
  warnings: WeatherWarningBrief[] | null;
}

export function WarningsBlock({ warnings }: WarningsBlockProps): JSX.Element | null {
  if (!warnings || warnings.length === 0) return null;

  // Sort by severity (red first)
  const severityOrder: WarningSeverity[] = ["red", "orange", "yellow", "green"];
  const sorted = [...warnings].sort(
    (a, b) => severityOrder.indexOf(a.severity) - severityOrder.indexOf(b.severity)
  );

  return (
    <Box
      data-testid="warnings-block"
      sx={{
        p: 1.6,
        borderRadius: 2.4,
        border: "1px solid rgba(249,115,22,0.25)",
        bgcolor: "rgba(22,27,34,0.55)",
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.85, mb: 1 }}>
        <WarningAmberIcon sx={{ fontSize: 14, color: "#f97316" }} />
        <Typography
          sx={{
            fontSize: 10,
            color: "#fff",
            fontWeight: 800,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
          }}
        >
          Active Weather Warnings
        </Typography>
      </Box>

      <Box sx={{ display: "flex", flexDirection: "column", gap: 0.7 }}>
        {sorted.map((w, i) => {
          const sev = SEVERITY_DISPLAY[w.severity] ?? SEVERITY_DISPLAY.yellow;
          const typeLabel = WARNING_TYPE_LABELS[w.warning_type] ?? "Warning";
          return (
            <Box
              key={`${w.source}-${w.warning_type}-${w.expires}`}
              sx={{
                p: 1,
                borderRadius: 1.5,
                border: `1px solid ${sev.borderColor}`,
                bgcolor: sev.bgColor,
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1, mb: 0.3 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.7 }}>
                  <Typography
                    data-testid={`warning-severity-badge-${i}`}
                    sx={{
                      fontSize: 8,
                      fontWeight: 900,
                      letterSpacing: "0.1em",
                      color: sev.color,
                      textTransform: "uppercase",
                      px: 0.6,
                      py: 0.1,
                      border: `1px solid ${sev.borderColor}`,
                      borderRadius: 999,
                      bgcolor: sev.bgColor,
                    }}
                  >
                    {sev.label}
                  </Typography>
                  <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#e5e7eb" }}>
                    {typeLabel}
                  </Typography>
                </Box>
                {w.country_code && (
                  <Typography sx={{ fontSize: 9, color: "#6b7280", fontFamily: "monospace" }}>
                    {w.country_code}
                  </Typography>
                )}
              </Box>
              <Typography sx={{ fontSize: 10, color: "#d1d5db", lineHeight: 1.4 }}>
                {w.headline}
              </Typography>
              <Typography sx={{ fontSize: 9, color: "#6b7280", mt: 0.3 }}>
                {formatTimeRemaining(w.expires)}
              </Typography>
            </Box>
          );
        })}
      </Box>

      <Typography sx={{ fontSize: 9, color: "#4b5563", mt: 0.9 }}>
        Source: MeteoAlarm
      </Typography>
    </Box>
  );
}
