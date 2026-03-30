import { Box, Typography } from "@mui/material";
import AirIcon from "@mui/icons-material/Air";
import type { WeatherContext, WeatherForecastStep, RhFireRisk } from "../../types/api";
import { CompassRose } from "./CompassRose";

const COMPASS_LABELS = [
  "N", "NNE", "NE", "ENE",
  "E", "ESE", "SE", "SSE",
  "S", "SSW", "SW", "WSW",
  "W", "WNW", "NW", "NNW",
] as const;

/** Convert meteorological degrees (0–360, where wind comes *from*) to a
 *  16-point compass label like "NNW". */
export function windCompassLabel(deg: number): string {
  const idx = Math.round(((deg % 360) + 360) % 360 / 22.5) % 16;
  return COMPASS_LABELS[idx];
}

interface RhDisplay {
  label: string;
  color: string;
  bgColor: string;
  borderColor: string;
}

const RH_DISPLAY: Record<RhFireRisk, RhDisplay> = {
  critical: {
    label: "Critical",
    color: "#fca5a5",
    bgColor: "rgba(239,68,68,0.14)",
    borderColor: "rgba(239,68,68,0.45)",
  },
  elevated: {
    label: "Elevated",
    color: "#fcd34d",
    bgColor: "rgba(234,179,8,0.14)",
    borderColor: "rgba(234,179,8,0.4)",
  },
  normal: {
    label: "Normal",
    color: "#86efac",
    bgColor: "rgba(34,197,94,0.12)",
    borderColor: "rgba(34,197,94,0.35)",
  },
};

export function rhRiskDisplay(risk: RhFireRisk): RhDisplay {
  return RH_DISPLAY[risk];
}

const CARD_SX = { p: 1.6, borderRadius: 2.4, border: "1px solid rgba(255,255,255,0.06)", bgcolor: "rgba(22,27,34,0.55)" } as const;
const HEADER_SX = { fontSize: 10, color: "#4b5563", fontWeight: 800, letterSpacing: "0.12em", textTransform: "uppercase", mb: 0.5 } as const;
const METRIC_CARD_SX = { p: 1.2, bgcolor: "#0d1117", borderRadius: 1.7, border: "1px solid rgba(255,255,255,0.08)" } as const;
const METRIC_LABEL_SX = { fontSize: 10, color: "#4b5563", fontWeight: 800, letterSpacing: "0.1em", textTransform: "uppercase", mb: 0.4 } as const;

// ---------------------------------------------------------------------------
// Forecast timeline sub-component
// ---------------------------------------------------------------------------

interface ForecastColumnProps {
  label: string;
  step: WeatherForecastStep;
}

function ForecastColumn({ label, step }: ForecastColumnProps): JSX.Element {
  const rh = rhRiskDisplay(step.rh_fire_risk);
  const compass = windCompassLabel(step.wind_direction_deg);
  return (
    <Box
      sx={{
        flex: 1,
        p: 1,
        bgcolor: "#0d1117",
        borderRadius: 1.5,
        border: "1px solid rgba(255,255,255,0.07)",
        minWidth: 0,
      }}
    >
      <Typography sx={{ fontSize: 9, color: "#6b7280", fontWeight: 800, letterSpacing: "0.12em", textTransform: "uppercase", mb: 0.7 }}>
        {label}
      </Typography>

      {/* Wind */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.7, mb: 0.6 }}>
        <CompassRose directionDeg={step.wind_direction_deg} size={28} />
        <Box>
          <Typography sx={{ fontSize: 11, fontWeight: 800, color: "#fff", lineHeight: 1.2 }}>
            {step.wind_speed_ms.toFixed(1)} m/s
          </Typography>
          <Typography sx={{ fontSize: 9, color: "#9ca3af" }}>
            {compass}
          </Typography>
        </Box>
      </Box>

      {/* RH */}
      <Box sx={{ mb: 0.5 }}>
        <Typography sx={{ fontSize: 11, fontWeight: 800, color: "#fff" }}>
          {step.relative_humidity_pct.toFixed(0)}%
        </Typography>
        <Typography
          sx={{
            display: "inline-flex",
            px: 0.6,
            py: 0.1,
            borderRadius: 999,
            bgcolor: rh.bgColor,
            border: `1px solid ${rh.borderColor}`,
            color: rh.color,
            fontSize: 8,
            fontWeight: 900,
            letterSpacing: "0.07em",
            textTransform: "uppercase",
          }}
        >
          {rh.label}
        </Typography>
      </Box>

      {/* Temp */}
      <Typography sx={{ fontSize: 11, fontWeight: 700, color: "#d1d5db" }}>
        {step.temperature_c.toFixed(1)} °C
      </Typography>
    </Box>
  );
}

interface ForecastTimelineProps {
  weather: WeatherContext;
  forecast: WeatherForecastStep[];
}

function ForecastTimeline({ weather, forecast }: ForecastTimelineProps): JSX.Element {
  const nowStep: WeatherForecastStep = {
    forecast_hour: 0,
    valid_time: weather.source_run_time,
    wind_speed_ms: weather.wind_speed_ms,
    wind_direction_deg: weather.wind_direction_deg,
    relative_humidity_pct: weather.relative_humidity_pct,
    rh_fire_risk: weather.rh_fire_risk,
    temperature_c: weather.temperature_c,
    precip_mm_24h: weather.precip_mm_24h,
  };

  return (
    <Box data-testid="forecast-timeline" sx={{ mt: 1.2 }}>
      <Typography sx={{ ...HEADER_SX, mb: 0.8 }}>Near-term outlook</Typography>
      <Box sx={{ display: "flex", gap: 0.7 }}>
        <ForecastColumn label="Now" step={nowStep} />
        {forecast.map((step) => (
          <ForecastColumn key={step.forecast_hour} label={`+${step.forecast_hour}h`} step={step} />
        ))}
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Main WeatherBlock component
// ---------------------------------------------------------------------------

interface WeatherBlockProps {
  weather: WeatherContext | null;
  unavailableReason: string | null;
  isLoading?: boolean;
  forecast?: WeatherForecastStep[] | null;
}

export function WeatherBlock({ weather, unavailableReason, isLoading, forecast }: WeatherBlockProps): JSX.Element {
  if (isLoading) {
    return (
      <Box sx={CARD_SX}>
        <Typography sx={HEADER_SX}>Weather Conditions</Typography>
        <Typography sx={{ fontSize: 11, color: "#6b7280", fontStyle: "italic" }}>
          Loading weather data…
        </Typography>
      </Box>
    );
  }

  if (!weather) {
    return (
      <Box sx={CARD_SX}>
        <Typography sx={HEADER_SX}>Weather Conditions</Typography>
        <Typography sx={{ fontSize: 11, color: "#6b7280", fontStyle: "italic" }}>
          {unavailableReason || "Weather data not available for this location"}
        </Typography>
      </Box>
    );
  }

  const rh = rhRiskDisplay(weather.rh_fire_risk);
  const compass = windCompassLabel(weather.wind_direction_deg);
  const biasApplied = weather.bias_correction?.applied;

  return (
    <Box data-testid="weather-block" sx={CARD_SX}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.85, mb: 1.2 }}>
        <AirIcon sx={{ fontSize: 14, color: "#60a5fa" }} />
        <Typography sx={{ fontSize: 10, color: "#fff", fontWeight: 800, letterSpacing: "0.12em", textTransform: "uppercase" }}>
          Weather Conditions
        </Typography>
      </Box>

      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, mb: 1.2 }}>
        <Box sx={METRIC_CARD_SX}>
          <Typography sx={METRIC_LABEL_SX}>Wind</Typography>
          <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#fff" }}>
            {weather.wind_speed_ms.toFixed(1)} m/s
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mt: 0.5 }}>
            <CompassRose directionDeg={weather.wind_direction_deg} size={40} />
            <Typography data-testid="wind-direction" sx={{ fontSize: 11, color: "#9ca3af" }}>
              from {compass} ({weather.wind_direction_deg.toFixed(0)}°)
            </Typography>
          </Box>
        </Box>

        <Box sx={{ ...METRIC_CARD_SX, border: `1px solid ${rh.borderColor}` }}>
          <Typography sx={METRIC_LABEL_SX}>Humidity</Typography>
          <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#fff" }}>
            {weather.relative_humidity_pct.toFixed(0)}%
          </Typography>
          <Typography
            data-testid="rh-risk-label"
            sx={{
              display: "inline-flex",
              mt: 0.4,
              px: 0.7,
              py: 0.15,
              borderRadius: 999,
              bgcolor: rh.bgColor,
              border: `1px solid ${rh.borderColor}`,
              color: rh.color,
              fontSize: 9,
              fontWeight: 900,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            {rh.label} fire risk
          </Typography>
        </Box>

        <Box sx={METRIC_CARD_SX}>
          <Typography sx={METRIC_LABEL_SX}>Temperature</Typography>
          <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#fff" }}>
            {weather.temperature_c.toFixed(1)} °C
          </Typography>
        </Box>

        <Box sx={METRIC_CARD_SX}>
          <Typography sx={METRIC_LABEL_SX}>Precip (24h)</Typography>
          <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#fff" }}>
            {weather.precip_mm_24h.toFixed(1)} mm
          </Typography>
        </Box>
      </Box>

      <Typography data-testid="weather-provenance" sx={{ fontSize: 10, color: "#4b5563" }}>
        {weather.resolution_note} · Data age: {weather.data_age_hours.toFixed(1)}h
        {biasApplied && " · Bias-corrected (ERA5 affine)"}
      </Typography>

      {forecast && forecast.length > 0 && (
        <ForecastTimeline weather={weather} forecast={forecast} />
      )}
    </Box>
  );
}
