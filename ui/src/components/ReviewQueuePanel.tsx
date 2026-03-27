import { useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Divider,
  Stack,
  Tooltip,
  Typography
} from "@mui/material";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import DoNotDisturbOnIcon from "@mui/icons-material/DoNotDisturbOn";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiUnavailableError, getDenoiserReviewQueue, resolveDenoiserReviewItem } from "../api/client";
import { safeFloat } from "../map/layerUtils";
import { useAppStore } from "../state/store";
import type { DenoiserReviewItem, FireEvent } from "../types/api";

type ResolutionNote = "confirmed_fire" | "marked_noise";

interface ReviewQueuePanelProps {
  visibleEvents: FireEvent[];
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  } catch {
    return iso;
  }
}

function formatFrp(value: unknown): string {
  const num = safeFloat(value);
  return num === null ? "n/a" : `${num.toFixed(0)} MW`;
}

function formatConfidence(value: unknown): string {
  const num = safeFloat(value);
  return num === null ? "n/a" : `${(num * 100).toFixed(0)}%`;
}

function ReasonChip({ reason }: { reason: string }) {
  const label = reason === "fail_closed_hard_bypass" ? "Hard Bypass" : "Uncertainty";
  const color = reason === "fail_closed_hard_bypass" ? "#ef4444" : "#f97316";
  return (
    <Box
      component="span"
      sx={{
        px: 0.75,
        py: 0.2,
        borderRadius: 0.8,
        fontSize: 9,
        fontWeight: 700,
        letterSpacing: "0.1em",
        textTransform: "uppercase",
        bgcolor: `${color}22`,
        border: `1px solid ${color}55`,
        color
      }}
    >
      {label}
    </Box>
  );
}

interface ReviewItemRowProps {
  item: DenoiserReviewItem;
  matchedEvent: FireEvent | null;
  onResolve: (eventId: string, notes: ResolutionNote) => void;
  isResolving: boolean;
}

function ReviewItemRow({ item, matchedEvent, onResolve, isResolving }: ReviewItemRowProps) {
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);

  const frp = item.payload_json?.frp_max;
  const confidence = item.payload_json?.confidence_max;
  const sensor = matchedEvent?.sensor ?? null;
  const time = matchedEvent?.end_time ?? item.created_at;

  function handleFocus() {
    const lat = safeFloat(matchedEvent?.lat);
    const lon = safeFloat(matchedEvent?.lon);
    if (lat !== null && lon !== null) {
      focusMapOnPoint(lat, lon, 9);
    }
  }

  const canFocus = safeFloat(matchedEvent?.lat) !== null;

  return (
    <Box
      sx={{
        p: 1.25,
        borderRadius: 1.5,
        bgcolor: "#0d1117",
        border: "1px solid rgba(251,146,60,0.25)",
        cursor: canFocus ? "pointer" : "default",
        transition: "border-color 0.15s",
        "&:hover": canFocus ? { borderColor: "rgba(251,146,60,0.55)" } : {}
      }}
      onClick={handleFocus}
    >
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", mb: 0.75 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap" }}>
          <ReasonChip reason={item.reason} />
          {sensor && (
            <Typography sx={{ fontSize: 10, color: "#6b7280", fontWeight: 600 }}>
              {sensor}
            </Typography>
          )}
        </Box>
        <Typography sx={{ fontSize: 10, color: "#4b5563", whiteSpace: "nowrap", ml: 1 }}>
          {formatTimestamp(time)}
        </Typography>
      </Box>

      <Box sx={{ display: "flex", gap: 2, mb: 1 }}>
        <Box>
          <Typography sx={{ fontSize: 9, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 700 }}>
            FRP
          </Typography>
          <Typography sx={{ fontSize: 13, color: "#e5e7eb", fontWeight: 700 }}>
            {formatFrp(frp)}
          </Typography>
        </Box>
        <Box>
          <Typography sx={{ fontSize: 9, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 700 }}>
            Confidence
          </Typography>
          <Typography sx={{ fontSize: 13, color: "#e5e7eb", fontWeight: 700 }}>
            {formatConfidence(confidence)}
          </Typography>
        </Box>
        {item.payload_json?.event_score != null && (
          <Box>
            <Typography sx={{ fontSize: 9, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 700 }}>
              Score
            </Typography>
            <Typography sx={{ fontSize: 13, color: "#e5e7eb", fontWeight: 700 }}>
              {item.payload_json.event_score.toFixed(3)}
            </Typography>
          </Box>
        )}
      </Box>

      <Box
        sx={{ display: "flex", gap: 0.75 }}
        onClick={(e) => e.stopPropagation()}
      >
        <Tooltip title="Confirm this detection as a real fire">
          <span>
            <Button
              size="small"
              disabled={isResolving}
              startIcon={isResolving ? <CircularProgress size={12} /> : <CheckCircleOutlineIcon />}
              onClick={() => onResolve(item.event_id, "confirmed_fire")}
              sx={{
                fontSize: 10,
                fontWeight: 700,
                py: 0.4,
                px: 1,
                color: "#4ade80",
                borderColor: "rgba(74,222,128,0.35)",
                border: "1px solid",
                borderRadius: 1,
                textTransform: "none",
                "&:hover": { bgcolor: "rgba(74,222,128,0.1)", borderColor: "rgba(74,222,128,0.6)" },
                "&:disabled": { opacity: 0.4 }
              }}
            >
              Confirm Fire
            </Button>
          </span>
        </Tooltip>
        <Tooltip title="Mark this detection as noise / false positive">
          <span>
            <Button
              size="small"
              disabled={isResolving}
              startIcon={isResolving ? <CircularProgress size={12} /> : <DoNotDisturbOnIcon />}
              onClick={() => onResolve(item.event_id, "marked_noise")}
              sx={{
                fontSize: 10,
                fontWeight: 700,
                py: 0.4,
                px: 1,
                color: "#9ca3af",
                borderColor: "rgba(156,163,175,0.3)",
                border: "1px solid",
                borderRadius: 1,
                textTransform: "none",
                "&:hover": { bgcolor: "rgba(156,163,175,0.08)", borderColor: "rgba(156,163,175,0.5)" },
                "&:disabled": { opacity: 0.4 }
              }}
            >
              Mark as Noise
            </Button>
          </span>
        </Tooltip>
      </Box>
    </Box>
  );
}

export default function ReviewQueuePanel({ visibleEvents }: ReviewQueuePanelProps) {
  const queryClient = useQueryClient();
  const [resolvingIds, setResolvingIds] = useState<Set<string>>(new Set());

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["denoiser-review-queue"],
    queryFn: getDenoiserReviewQueue,
    refetchInterval: 15_000,
    staleTime: 15_000
  });

  const resolveMutation = useMutation({
    mutationFn: resolveDenoiserReviewItem,
    onMutate: ({ eventId }) => {
      setResolvingIds((prev) => new Set(prev).add(eventId));
    },
    onSettled: (_data, _err, { eventId }) => {
      setResolvingIds((prev) => {
        const next = new Set(prev);
        next.delete(eventId);
        return next;
      });
      void queryClient.invalidateQueries({ queryKey: ["denoiser-review-queue"] });
    }
  });

  const visibleEventIndex = useMemo(
    () =>
      new Map<string, FireEvent>(
        visibleEvents
          .filter((e) => e.event_id != null)
          .map((e) => [String(e.event_id), e])
      ),
    [visibleEvents]
  );

  const rows = data?.rows ?? [];
  const count = rows.length;

  function handleResolve(eventId: string, notes: ResolutionNote) {
    resolveMutation.mutate({ eventId, resolvedBy: "operator", resolvedNotes: notes });
  }

  return (
    <Box
      sx={{
        bgcolor: "#010409",
        border: "1px solid rgba(251,146,60,0.2)",
        borderRadius: 2.5,
        overflow: "hidden"
      }}
    >
      {/* Header */}
      <Box
        sx={{
          px: 2,
          py: 1.25,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid rgba(255,255,255,0.05)"
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <WarningAmberIcon sx={{ fontSize: 15, color: "#fb923c" }} />
          <Typography
            sx={{
              fontSize: 11,
              fontWeight: 800,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "#d1d5db"
            }}
          >
            Review Queue
          </Typography>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          {isLoading && <CircularProgress size={12} sx={{ color: "#6b7280" }} />}
          <Chip
            label={count}
            size="small"
            sx={{
              height: 18,
              fontSize: 10,
              fontWeight: 800,
              bgcolor: count > 0 ? "rgba(251,146,60,0.15)" : "rgba(107,114,128,0.15)",
              color: count > 0 ? "#fb923c" : "#6b7280",
              border: count > 0 ? "1px solid rgba(251,146,60,0.3)" : "1px solid rgba(107,114,128,0.2)",
              "& .MuiChip-label": { px: 0.75 }
            }}
          />
        </Box>
      </Box>

      {/* Body */}
      <Box sx={{ maxHeight: 400, overflowY: "auto", p: 1.5 }}>
        {isError && (
          <Alert
            severity="warning"
            sx={{ fontSize: 11, bgcolor: "transparent", color: "#f97316", "& .MuiAlert-icon": { color: "#f97316" } }}
          >
            {error instanceof ApiUnavailableError ? "API unavailable" : "Failed to load review queue"}
          </Alert>
        )}

        {!isLoading && !isError && count === 0 && (
          <Typography sx={{ fontSize: 12, color: "#4b5563", textAlign: "center", py: 2 }}>
            No flagged detections pending review
          </Typography>
        )}

        {count > 0 && (
          <Stack
            spacing={1}
            divider={<Divider sx={{ borderColor: "rgba(255,255,255,0.04)" }} />}
          >
            {rows.map((item) => (
              <ReviewItemRow
                key={item.event_id}
                item={item}
                matchedEvent={visibleEventIndex.get(item.event_id) ?? null}
                onResolve={handleResolve}
                isResolving={resolvingIds.has(item.event_id)}
              />
            ))}
          </Stack>
        )}
      </Box>
    </Box>
  );
}
