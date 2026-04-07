import { Box, Typography } from "@mui/material";
import ConstructionIcon from "@mui/icons-material/Construction";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import SatelliteAltIcon from "@mui/icons-material/SatelliteAlt";

/**
 * Full-screen maintenance page shown when the API backend is unreachable.
 * Explains the service is in active development and not running 24/7.
 */
export default function ServiceDownPage({ onRetry }: { onRetry: () => void }) {
  return (
    <Box
      sx={{
        minHeight: "100vh",
        bgcolor: "#010409",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        px: 3,
        textAlign: "center",
      }}
    >
      {/* Icon cluster */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 4 }}>
        <SatelliteAltIcon sx={{ fontSize: 32, color: "#60a5fa", opacity: 0.6 }} />
        <LocalFireDepartmentIcon sx={{ fontSize: 48, color: "#f97316" }} />
        <ConstructionIcon sx={{ fontSize: 32, color: "#eab308", opacity: 0.6 }} />
      </Box>

      {/* Title */}
      <Typography
        variant="h3"
        sx={{
          color: "#fff",
          fontWeight: 800,
          letterSpacing: "-0.02em",
          mb: 1.5,
          fontSize: { xs: 28, md: 36 },
        }}
      >
        Wildfire Nowcast
      </Typography>

      <Box
        sx={{
          px: 2,
          py: 0.5,
          mb: 4,
          borderRadius: 1,
          bgcolor: "rgba(234,179,8,0.12)",
          border: "1px solid rgba(234,179,8,0.3)",
        }}
      >
        <Typography
          sx={{
            color: "#eab308",
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
          }}
        >
          Service Offline
        </Typography>
      </Box>

      {/* Explanation */}
      <Box sx={{ maxWidth: 520, mb: 5 }}>
        <Typography sx={{ color: "#9ca3af", fontSize: 15, lineHeight: 1.75, mb: 2 }}>
          This platform ingests and processes global satellite fire data around the clock,
          which makes continuous hosting expensive for a project still under active development.
        </Typography>
        <Typography sx={{ color: "#9ca3af", fontSize: 15, lineHeight: 1.75, mb: 2 }}>
          To keep costs manageable, the backend services are spun up on demand — typically
          during <strong style={{ color: "#d1d5db" }}>weekends and development sessions</strong>.
        </Typography>
        <Typography sx={{ color: "#6b7280", fontSize: 13, lineHeight: 1.7 }}>
          If you'd like to see the system live, check back during the weekend
          or reach out for a scheduled demo.
        </Typography>
      </Box>

      {/* Retry button */}
      <Box
        component="button"
        onClick={onRetry}
        sx={{
          px: 3,
          py: 1.2,
          mb: 5,
          borderRadius: 2,
          border: "1px solid rgba(249,115,22,0.4)",
          bgcolor: "rgba(249,115,22,0.08)",
          color: "#f97316",
          fontSize: 13,
          fontWeight: 700,
          letterSpacing: "0.06em",
          cursor: "pointer",
          transition: "all 0.15s",
          "&:hover": {
            bgcolor: "rgba(249,115,22,0.16)",
            borderColor: "rgba(249,115,22,0.6)",
          },
        }}
      >
        Retry Connection
      </Box>

      {/* Footer */}
      <Box sx={{ position: "absolute", bottom: 24 }}>
        <Typography sx={{ color: "#374151", fontSize: 11, letterSpacing: "0.1em" }}>
          Wildfire Nowcast — Active Development
        </Typography>
      </Box>
    </Box>
  );
}
