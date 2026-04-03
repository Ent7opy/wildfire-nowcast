import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  IconButton,
  Slider,
  Tooltip,
  Typography
} from "@mui/material";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import SkipPreviousIcon from "@mui/icons-material/SkipPrevious";

import type { ArchiveDayIngestStatus, ArchiveRangeDayStatus } from "../api/client";
import { datesInRange } from "../utils/time";

interface ArchiveRangeScrubberProps {
  startDate: string;             // 'YYYY-MM-DD'
  endDate: string;               // 'YYYY-MM-DD'
  scrubDate: string | null;
  dayStatuses: ArchiveRangeDayStatus[];
  onScrub: (date: string) => void;
}

// Speed label → milliseconds per day step
const PLAY_SPEEDS_MS: Record<string, number> = {
  "0.5×": 2000,
  "1×": 1000,
  "2×": 500,
};

function statusColor(status: ArchiveDayIngestStatus | undefined): string {
  switch (status) {
    case "finished": return "#4ade80";
    case "started":  return "#f97316";
    case "failed":   return "#ef4444";
    default:         return "#374151"; // queued / unknown
  }
}

export default function ArchiveRangeScrubber({
  startDate,
  endDate,
  scrubDate,
  dayStatuses,
  onScrub,
}: ArchiveRangeScrubberProps) {
  // Memoize derived values so they don't recompute on every render
  const dates = useMemo(() => datesInRange(startDate, endDate), [startDate, endDate]);
  const statusByDate = useMemo(
    () => Object.fromEntries(dayStatuses.map((d) => [d.date, d.status])) as Record<string, ArchiveDayIngestStatus>,
    [dayStatuses]
  );

  const currentIndex = scrubDate ? Math.max(0, dates.indexOf(scrubDate)) : 0;

  const [isPlaying, setIsPlaying] = useState(false);
  const [speedLabel, setSpeedLabel] = useState<string>("1×");
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Refs for mutable playback state — lets advanceToNext stay stable across renders
  // so the playback interval is not unnecessarily restarted on each scrub step.
  const datesRef = useRef(dates);
  datesRef.current = dates;
  const statusByDateRef = useRef(statusByDate);
  statusByDateRef.current = statusByDate;
  const scrubDateRef = useRef(scrubDate);
  scrubDateRef.current = scrubDate;
  const onScrubRef = useRef(onScrub);
  onScrubRef.current = onScrub;

  const stopPlay = useCallback(() => {
    if (playRef.current !== null) {
      clearInterval(playRef.current);
      playRef.current = null;
    }
    setIsPlaying(false);
  }, []);

  const stopPlayRef = useRef(stopPlay);
  stopPlayRef.current = stopPlay;

  // Stable callback — reads current values from refs, so the playback interval
  // doesn't need to restart every time scrubDate changes.
  const advanceToNext = useCallback(() => {
    const currentDates = datesRef.current;
    const currentScrubDate = scrubDateRef.current;
    const currentStatusByDate = statusByDateRef.current;

    if (!currentScrubDate) {
      onScrubRef.current(currentDates[0]);
      return;
    }
    const idx = currentDates.indexOf(currentScrubDate);
    if (idx < 0 || idx >= currentDates.length - 1) {
      stopPlayRef.current();
      return;
    }
    const nextDate = currentDates[idx + 1];
    const nextStatus = currentStatusByDate[nextDate];
    // Only advance to dates that have finished ingesting
    if (nextStatus === "finished") {
      onScrubRef.current(nextDate);
    } else if (nextStatus === "failed") {
      // Dead-end — stop rather than loop forever
      stopPlayRef.current();
    }
    // 'started' / 'queued': silently wait and retry on the next interval tick (expected)
  }, []); // stable — intentionally no deps; reads from refs above

  // Restart interval only when play state or speed changes (not on every scrub step)
  useEffect(() => {
    if (!isPlaying) return;
    const intervalMs = PLAY_SPEEDS_MS[speedLabel] ?? 1000;
    playRef.current = setInterval(advanceToNext, intervalMs);
    return () => {
      if (playRef.current !== null) clearInterval(playRef.current);
    };
  }, [isPlaying, speedLabel, advanceToNext]);

  // Stop playback when component unmounts
  useEffect(() => () => stopPlay(), [stopPlay]);

  const handlePlayPause = () => {
    if (isPlaying) {
      stopPlay();
    } else {
      setIsPlaying(true);
    }
  };

  const handleSliderChange = (_: Event, value: number | number[]) => {
    if (Array.isArray(value)) return;
    const date = dates[value];
    if (date) onScrub(date);
    stopPlay();
  };

  const handlePrev = () => {
    stopPlay();
    const idx = scrubDate ? dates.indexOf(scrubDate) : 0;
    if (idx > 0) onScrub(dates[idx - 1]);
  };

  const handleNext = () => {
    stopPlay();
    const idx = scrubDate ? dates.indexOf(scrubDate) : -1;
    if (idx < dates.length - 1) onScrub(dates[idx + 1]);
  };

  return (
    <Box sx={{ bgcolor: "#0d1117", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 2, p: 2 }}>
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
        <Typography sx={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.14em", textTransform: "uppercase", color: "#6b7280" }}>
          Timeline Scrubber
        </Typography>
        <Typography sx={{ fontSize: 11, color: "#60a5fa", fontWeight: 700 }}>
          {scrubDate ?? startDate}
        </Typography>
      </Box>

      {/* Day status track */}
      <Box sx={{ display: "flex", gap: "2px", mb: 1.5, alignItems: "center" }}>
        {dates.map((d) => {
          const s = statusByDate[d];
          const isCurrent = d === scrubDate;
          return (
            <Tooltip key={d} title={`${d} · ${s ?? "queued"}`}>
              <Box
                component="button"
                onClick={() => { onScrub(d); stopPlay(); }}
                sx={{
                  flex: 1,
                  height: 10,
                  borderRadius: "2px",
                  cursor: "pointer",
                  border: isCurrent ? "1px solid #60a5fa" : "1px solid transparent",
                  bgcolor: statusColor(s),
                  transition: "all 0.1s",
                  "&:hover": { opacity: 0.8 }
                }}
              />
            </Tooltip>
          );
        })}
      </Box>

      {/* Slider */}
      <Slider
        value={currentIndex}
        min={0}
        max={Math.max(0, dates.length - 1)}
        step={1}
        onChange={handleSliderChange}
        size="small"
        sx={{
          color: "#60a5fa",
          "& .MuiSlider-thumb": { width: 12, height: 12 },
          "& .MuiSlider-track": { height: 3 },
          "& .MuiSlider-rail": { height: 3, bgcolor: "#374151" },
          mb: 0.5
        }}
      />

      {/* Controls */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mt: 0.5 }}>
        <IconButton size="small" onClick={handlePrev} disabled={currentIndex === 0} sx={{ color: "#9ca3af" }}>
          <SkipPreviousIcon fontSize="small" />
        </IconButton>
        <IconButton size="small" onClick={handlePlayPause} sx={{ color: isPlaying ? "#f97316" : "#60a5fa" }}>
          {isPlaying ? <PauseIcon fontSize="small" /> : <PlayArrowIcon fontSize="small" />}
        </IconButton>
        <IconButton size="small" onClick={handleNext} disabled={currentIndex >= dates.length - 1} sx={{ color: "#9ca3af" }}>
          <SkipNextIcon fontSize="small" />
        </IconButton>

        {/* Speed selector */}
        <Box sx={{ ml: "auto", display: "flex", gap: 0.5 }}>
          {Object.keys(PLAY_SPEEDS_MS).map((label) => (
            <Box
              key={label}
              component="button"
              onClick={() => setSpeedLabel(label)}
              sx={{
                px: 1,
                py: 0.3,
                borderRadius: 1,
                fontSize: 10,
                fontWeight: 700,
                cursor: "pointer",
                border: speedLabel === label ? "1px solid rgba(96,165,250,0.5)" : "1px solid rgba(255,255,255,0.1)",
                bgcolor: speedLabel === label ? "rgba(96,165,250,0.1)" : "transparent",
                color: speedLabel === label ? "#60a5fa" : "#6b7280",
              }}
            >
              {label}
            </Box>
          ))}
        </Box>

        <Typography sx={{ fontSize: 10, color: "#4b5563", ml: 1 }}>
          {dates.indexOf(scrubDate ?? "") + 1}/{dates.length}
        </Typography>
      </Box>

      {/* Legend */}
      <Box sx={{ display: "flex", gap: 2, mt: 1 }}>
        {[
          { color: "#4ade80", label: "Ready" },
          { color: "#f97316", label: "Loading" },
          { color: "#ef4444", label: "Failed" },
          { color: "#374151", label: "Queued" },
        ].map(({ color, label }) => (
          <Box key={label} sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: color }} />
            <Typography sx={{ fontSize: 9, color: "#6b7280" }}>{label}</Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );
}
