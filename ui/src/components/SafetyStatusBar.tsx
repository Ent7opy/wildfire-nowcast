import { Box, CircularProgress, Typography } from "@mui/material";
import GpsFixedIcon from "@mui/icons-material/GpsFixed";
import GpsOffIcon from "@mui/icons-material/GpsOff";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import ShieldIcon from "@mui/icons-material/Shield";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { useAppStore } from "../state/store";
import type { SafetyTier } from "../types/state";

const TIER_CONFIG: Record<SafetyTier, { label: string; bg: string; border: string; chip: string; text: string }> = {
  SAFE:    { label: "SAFE",    bg: "rgba(34,197,94,0.08)",   border: "#22c55e", chip: "#22c55e",  text: "#86efac" },
  WATCH:   { label: "WATCH",   bg: "rgba(234,179,8,0.08)",   border: "#eab308", chip: "#eab308",  text: "#fde047" },
  WARNING: { label: "WARNING", bg: "rgba(249,115,22,0.10)",  border: "#f97316", chip: "#f97316",  text: "#fdba74" },
  DANGER:  { label: "DANGER",  bg: "rgba(239,68,68,0.14)",   border: "#ef4444", chip: "#ef4444",  text: "#fca5a5" },
};

interface SafetyStatusBarProps {
  onDisable: () => void;
  onLocate: () => void;
  nearbyCount: number;
}

export default function SafetyStatusBar({ onDisable, onLocate, nearbyCount }: SafetyStatusBarProps): JSX.Element {
  const safety = useAppStore((s) => s.safety);
  const cfg = TIER_CONFIG[safety.safetyTier];

  const distanceLabel = safety.nearestFireDistanceKm !== null
    ? `${safety.nearestFireDistanceKm.toFixed(1)} km away`
    : "distance unknown";

  const tierIcon = safety.safetyTier === "SAFE"
    ? <ShieldIcon sx={{ fontSize: 16 }} />
    : <WarningAmberIcon sx={{ fontSize: 16 }} />;

  return (
    <Box
      sx={{
        width: "100%",
        borderRadius: 2,
        bgcolor: cfg.bg,
        border: `1px solid ${cfg.border}30`,
        borderLeft: `3px solid ${cfg.border}`,
        px: 2,
        py: 1,
        display: "flex",
        alignItems: "center",
        gap: 2,
        flexWrap: "wrap",
      }}
    >
      {/* Tier badge */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexShrink: 0 }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 0.5,
            px: 1,
            py: 0.4,
            borderRadius: 1,
            bgcolor: `${cfg.chip}20`,
            border: `1px solid ${cfg.chip}60`,
            color: cfg.chip,
          }}
        >
          {tierIcon}
          <Typography sx={{ fontSize: 11, fontWeight: 900, letterSpacing: "0.14em" }}>
            {cfg.label}
          </Typography>
        </Box>
        {safety.nearestFireDistanceKm !== null && (
          <Typography sx={{ fontSize: 11, color: cfg.text, fontWeight: 600 }}>
            Fire {distanceLabel}
          </Typography>
        )}
      </Box>

      {/* Mini stats */}
      <Box sx={{ display: "flex", gap: 2.5, flex: 1, alignItems: "center", flexWrap: "wrap" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <LocalFireDepartmentIcon sx={{ fontSize: 13, color: "#9ca3af" }} />
          <Typography sx={{ fontSize: 11, color: "#9ca3af" }}>
            <Box component="span" sx={{ color: "#e5e7eb", fontWeight: 700 }}>{nearbyCount}</Box>
            {" "}fires in {safety.proximityRadiusKm} km
          </Typography>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <Typography sx={{ fontSize: 11, color: "#9ca3af" }}>
            Risk:{" "}
            <Box component="span" sx={{ color: cfg.chip, fontWeight: 700 }}>
              {safety.safetyTier === "SAFE" ? "None Detected" : safety.safetyTier === "WATCH" ? "Monitor" : safety.safetyTier === "WARNING" ? "Be Ready" : "Evacuate"}
            </Box>
          </Typography>
        </Box>
      </Box>

      {/* GPS + exit controls */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexShrink: 0 }}>
        {safety.locationPermission === 'denied' && (
          <Typography sx={{ fontSize: 10, color: "#eab308", fontWeight: 600 }}>
            Location denied — enable in browser settings
          </Typography>
        )}
        {safety.locationPermission !== 'granted' && safety.locationPermission !== 'denied' && (
          <Box
            component="button"
            onClick={onLocate}
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 0.5,
              px: 1,
              py: 0.5,
              borderRadius: 1,
              border: "1px solid rgba(96,165,250,0.4)",
              bgcolor: "rgba(96,165,250,0.08)",
              color: "#60a5fa",
              cursor: "pointer",
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: "0.1em",
              "&:hover": { bgcolor: "rgba(96,165,250,0.14)" },
            }}
          >
            {safety.locationPermission === 'requesting'
              ? <CircularProgress size={11} sx={{ color: "#60a5fa", mr: 0.5 }} />
              : <GpsFixedIcon sx={{ fontSize: 12 }} />
            }
            {safety.locationPermission === 'requesting' ? "Locating…" : "Locate Me"}
          </Box>
        )}
        {safety.locationPermission === 'denied' && (
          <GpsOffIcon sx={{ fontSize: 14, color: "#eab308" }} />
        )}
        <Box
          component="button"
          onClick={onDisable}
          sx={{
            px: 1,
            py: 0.5,
            borderRadius: 1,
            border: "1px solid rgba(255,255,255,0.1)",
            bgcolor: "transparent",
            color: "#6b7280",
            cursor: "pointer",
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.08em",
            "&:hover": { color: "#9ca3af", borderColor: "rgba(255,255,255,0.2)" },
          }}
        >
          Exit Safety Mode
        </Box>
      </Box>
    </Box>
  );
}
