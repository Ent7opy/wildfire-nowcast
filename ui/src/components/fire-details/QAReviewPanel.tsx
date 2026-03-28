import { Box, Typography } from "@mui/material";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import type { FireEvent } from "../../types/api";
import type { ForecastRunMeta } from "../../types/state";

interface QAReviewPanelProps {
  event: FireEvent;
  runMeta: ForecastRunMeta | null | undefined;
}

export function QAReviewPanel({ event, runMeta }: QAReviewPanelProps): JSX.Element {
  return (
    <>
      {event.review_required && (
        <Box sx={{ mt: 0.8, px: 1.25, py: 1, border: "1px solid rgba(250,204,21,0.3)", bgcolor: "rgba(250,204,21,0.08)", borderRadius: 1.6, display: "flex", alignItems: "center", gap: 0.85 }}>
          <WarningAmberIcon sx={{ fontSize: 14, color: "#facc15" }} />
          <Typography sx={{ fontSize: 11, color: "#fcd34d", fontWeight: 700 }}>
            Analyst review required — perimeter and intensity are provisional.
          </Typography>
        </Box>
      )}

      {runMeta?.weatherRunId === null && (
        <Box sx={{ mt: 0.8, px: 1.25, py: 1, border: "1px solid rgba(251,146,60,0.3)", bgcolor: "rgba(251,146,60,0.08)", borderRadius: 1.6, display: "flex", alignItems: "flex-start", gap: 0.85 }}>
          <WarningAmberIcon sx={{ fontSize: 14, color: "#fb923c", mt: 0.1 }} />
          <Typography sx={{ fontSize: 11, color: "#fdba74", fontWeight: 600, lineHeight: 1.5 }}>
            Spread forecast assumed calm conditions — no weather data was available for this area and time. The symmetric shape reflects this, not actual wind direction.
          </Typography>
        </Box>
      )}

      {runMeta && (runMeta.confidenceLevel === "low" || runMeta.fallbackUsed) && (
        <Box sx={{ mt: 0.8, px: 1.25, py: 1, border: "1px solid rgba(234,179,8,0.3)", bgcolor: "rgba(234,179,8,0.07)", borderRadius: 1.6, display: "flex", alignItems: "flex-start", gap: 0.85 }}>
          <WarningAmberIcon sx={{ fontSize: 14, color: "#eab308", mt: 0.1 }} />
          <Typography sx={{ fontSize: 11, color: "#fde047", fontWeight: 600, lineHeight: 1.5 }}>
            Low-confidence forecast — stale or fallback inputs were used. Treat spread contours as indicative only.
          </Typography>
        </Box>
      )}

      {runMeta?.weatherBiasApplied === false && (
        <Box sx={{ mt: 0.8, px: 1.25, py: 1, border: "1px solid rgba(148,163,184,0.2)", bgcolor: "rgba(148,163,184,0.06)", borderRadius: 1.6, display: "flex", alignItems: "flex-start", gap: 0.85 }}>
          <WarningAmberIcon sx={{ fontSize: 14, color: "#94a3b8", mt: 0.1 }} />
          <Typography sx={{ fontSize: 11, color: "#cbd5e1", fontWeight: 600, lineHeight: 1.5 }}>
            Regional weather bias correction was not applied — forecast accuracy may be lower than usual for this location.
          </Typography>
        </Box>
      )}
    </>
  );
}
