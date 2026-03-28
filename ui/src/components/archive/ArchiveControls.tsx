import { useCallback, useState } from "react";
import { Box, Button, CircularProgress, LinearProgress, TextField, Tooltip, Typography } from "@mui/material";
import Brightness3Icon from "@mui/icons-material/Brightness3";
import SunriseIcon from "@mui/icons-material/WbTwilight";
import SunsetIcon from "@mui/icons-material/Nightlight";
import SunIcon from "@mui/icons-material/WbSunny";
import { useAppStore } from "../../state/store";
import { TIMEFRAME_DEFS } from "../../utils/time";
import type { ArchiveTimeframe } from "../../types/state";

export const TIMEFRAME_ICONS: Record<ArchiveTimeframe, React.ElementType> = {
  morning: SunriseIcon,
  afternoon: SunIcon,
  evening: SunsetIcon,
  night: Brightness3Icon,
};

interface ArchiveControlsProps {
  archiveData: {
    status: string;
    message?: string | null;
  };
  archiveRangeData: {
    status: string;
    message?: string | null;
    totalCount: number;
    completedCount: number;
    warning?: string | null;
    dayStatuses: unknown[];
  };
}

export function ArchiveControls({ archiveData, archiveRangeData }: ArchiveControlsProps): JSX.Element {
  const archive = useAppStore((s) => s.archive);
  const setArchiveDate = useAppStore((s) => s.setArchiveDate);
  const setArchiveTimeframe = useAppStore((s) => s.setArchiveTimeframe);
  const setArchiveSubMode = useAppStore((s) => s.setArchiveSubMode);
  const setArchiveRange = useAppStore((s) => s.setArchiveRange);

  const today = new Date().toISOString().slice(0, 10);
  const [rangeInputStart, setRangeInputStart] = useState<string>("");
  const [rangeInputEnd, setRangeInputEnd] = useState<string>("");

  const handleLoadRange = useCallback(() => {
    if (!rangeInputStart || !rangeInputEnd) return;
    setArchiveRange(rangeInputStart, rangeInputEnd);
  }, [rangeInputStart, rangeInputEnd, setArchiveRange]);

  return (
    <Box sx={{ mt: 1.5 }}>
      {/* Sub-mode selector: Single Day | Range */}
      <Box sx={{ display: "flex", gap: 0, mb: 1.5, p: 0.4, bgcolor: "#0d1117", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 1.5, width: "fit-content" }}>
        {(["single", "range"] as const).map((mode) => {
          const active = archive.archiveSubMode === mode;
          return (
            <Box
              key={mode}
              component="button"
              onClick={() => setArchiveSubMode(mode)}
              sx={{
                px: 1.5,
                py: 0.5,
                borderRadius: 1,
                fontSize: 10,
                fontWeight: 800,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                cursor: "pointer",
                border: "none",
                bgcolor: active ? (mode === "range" ? "rgba(96,165,250,0.15)" : "rgba(249,115,22,0.12)") : "transparent",
                color: active ? (mode === "range" ? "#60a5fa" : "#f97316") : "#6b7280",
                transition: "all 0.15s",
                "&:hover": { color: mode === "range" ? "#93c5fd" : "#fb923c" }
              }}
            >
              {mode === "single" ? "Single Day" : "Date Range"}
            </Box>
          );
        })}
      </Box>

      {/* Single-day controls */}
      {archive.archiveSubMode === "single" && (
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, flexWrap: "wrap" }}>
          <TextField
            type="date"
            size="small"
            value={archive.archiveDate ?? ""}
            onChange={(e) => setArchiveDate(e.target.value)}
            inputProps={{ max: today }}
            sx={{
              "& .MuiOutlinedInput-root": { bgcolor: "#0d1117", borderRadius: 2, fontSize: 12, color: "#e5e7eb" },
              "& input": { colorScheme: "light" }
            }}
          />
          <Box sx={{ display: "flex", gap: 0.5 }}>
            {TIMEFRAME_DEFS.map((def) => {
              const Icon = TIMEFRAME_ICONS[def.id];
              const isSelected = archive.archiveTimeframe === def.id;
              return (
                <Tooltip key={def.id} title={def.label}>
                  <Box
                    component="button"
                    onClick={() => setArchiveTimeframe(def.id)}
                    sx={{
                      p: 0.8,
                      borderRadius: 1.5,
                      border: isSelected ? "1px solid rgba(59,130,246,0.6)" : "1px solid rgba(255,255,255,0.1)",
                      bgcolor: isSelected ? "rgba(59,130,246,0.15)" : "#0d1117",
                      color: isSelected ? "#60a5fa" : "#6b7280",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      transition: "all 0.15s",
                      "&:hover": { borderColor: "rgba(59,130,246,0.4)", color: "#93c5fd" }
                    }}
                  >
                    <Icon sx={{ fontSize: 16 }} />
                  </Box>
                </Tooltip>
              );
            })}
          </Box>
          {archiveData.status === "checking" && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
              <CircularProgress size={14} sx={{ color: "#60a5fa" }} />
              <Typography sx={{ fontSize: 11, color: "#6b7280" }}>Checking data…</Typography>
            </Box>
          )}
          {archiveData.status === "ingesting" && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
              <CircularProgress size={14} sx={{ color: "#f97316" }} />
              <Typography sx={{ fontSize: 11, color: "#f97316" }}>
                {archiveData.message ?? "Ingesting data…"}
              </Typography>
            </Box>
          )}
          {archiveData.status === "unavailable" && (
            <Typography sx={{ fontSize: 11, color: "#ef4444" }}>
              {archiveData.message ?? "Data unavailable for this date."}
            </Typography>
          )}
        </Box>
      )}

      {/* Range controls */}
      {archive.archiveSubMode === "range" && (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
            <TextField
              type="date"
              size="small"
              label="Start"
              value={rangeInputStart}
              onChange={(e) => setRangeInputStart(e.target.value)}
              inputProps={{ max: today }}
              InputLabelProps={{ shrink: true, sx: { fontSize: 11, color: "#6b7280" } }}
              sx={{
                "& .MuiOutlinedInput-root": { bgcolor: "#0d1117", borderRadius: 2, fontSize: 12, color: "#e5e7eb" },
                "& input": { colorScheme: "light" }
              }}
            />
            <Typography sx={{ fontSize: 10, color: "#374151" }}>→</Typography>
            <TextField
              type="date"
              size="small"
              label="End"
              value={rangeInputEnd}
              onChange={(e) => setRangeInputEnd(e.target.value)}
              inputProps={{ max: today, min: rangeInputStart || undefined }}
              InputLabelProps={{ shrink: true, sx: { fontSize: 11, color: "#6b7280" } }}
              sx={{
                "& .MuiOutlinedInput-root": { bgcolor: "#0d1117", borderRadius: 2, fontSize: 12, color: "#e5e7eb" },
                "& input": { colorScheme: "light" }
              }}
            />
            <Button
              size="small"
              variant="outlined"
              onClick={handleLoadRange}
              disabled={!rangeInputStart || !rangeInputEnd || archiveRangeData.status === "loading"}
              sx={{
                fontSize: 10,
                fontWeight: 800,
                letterSpacing: "0.1em",
                borderColor: "rgba(96,165,250,0.4)",
                color: "#60a5fa",
                "&:hover": { borderColor: "rgba(96,165,250,0.7)", bgcolor: "rgba(96,165,250,0.07)" },
                "&.Mui-disabled": { borderColor: "rgba(255,255,255,0.08)", color: "#374151" }
              }}
            >
              Load Range
            </Button>
          </Box>

          {/* Range ingest progress */}
          {archiveRangeData.status === "loading" && archiveRangeData.totalCount > 0 && (
            <Box sx={{ maxWidth: 420 }}>
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 0.5 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                  <CircularProgress size={12} sx={{ color: "#60a5fa" }} />
                  <Typography sx={{ fontSize: 11, color: "#6b7280" }}>
                    {archiveRangeData.message ?? "Loading range…"}
                  </Typography>
                </Box>
                <Typography sx={{ fontSize: 10, color: "#374151", fontVariantNumeric: "tabular-nums" }}>
                  {archiveRangeData.completedCount}/{archiveRangeData.totalCount}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={archiveRangeData.totalCount > 0 ? (archiveRangeData.completedCount / archiveRangeData.totalCount) * 100 : 0}
                sx={{
                  height: 3,
                  borderRadius: 2,
                  bgcolor: "rgba(255,255,255,0.06)",
                  "& .MuiLinearProgress-bar": { bgcolor: "#60a5fa", borderRadius: 2 }
                }}
              />
            </Box>
          )}

          {/* Warning for large ranges */}
          {archiveRangeData.warning && (
            <Typography sx={{ fontSize: 11, color: "#eab308", maxWidth: 500 }}>
              ⚠ {archiveRangeData.warning}
            </Typography>
          )}

          {/* Error */}
          {archiveRangeData.status === "unavailable" && (
            <Typography sx={{ fontSize: 11, color: "#ef4444" }}>
              {archiveRangeData.message ?? "Range ingest unavailable."}
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
}
