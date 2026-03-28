import { Button, Typography } from "@mui/material";
import type { FireEvent } from "../../types/api";

interface ForecastButtonState {
  label: string;
  disabled: boolean;
  reason?: string;
}

interface ForecastPanelProps {
  event: FireEvent;
  button: ForecastButtonState;
  isPending: boolean;
  lat: number | null;
  lon: number | null;
  isSafetyMode: boolean;
  isArchiveMode: boolean;
  onRequestForecast: (event: FireEvent) => void;
}

export function ForecastPanel({
  event,
  button,
  isPending,
  lat,
  lon,
  isSafetyMode,
  isArchiveMode,
  onRequestForecast
}: ForecastPanelProps): JSX.Element | null {
  if (isArchiveMode) {
    return null;
  }

  if (!isSafetyMode) {
    return (
      <>
        <Button
          variant="contained"
          disabled={button.disabled || isPending || lat === null || lon === null}
          onClick={() => onRequestForecast(event)}
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
      </>
    );
  }

  // Safety mode: demoted forecast button at bottom
  return (
    <Button
      variant="outlined"
      size="small"
      disabled={button.disabled || isPending || lat === null || lon === null}
      onClick={() => onRequestForecast(event)}
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
  );
}
