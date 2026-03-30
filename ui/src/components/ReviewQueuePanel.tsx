import { useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Collapse,
  Divider,
  Stack,
  Tooltip,
  Typography
} from "@mui/material";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import DoNotDisturbOnIcon from "@mui/icons-material/DoNotDisturbOn";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
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

const HARD_BYPASS = "fail_closed_hard_bypass";

const REASON_META: Record<string, { label: string; color: string; tooltip: string }> = {
  fail_closed_hard_bypass: {
    label: "High-Energy Alert",
    color: "#ef4444",
    tooltip:
      "Exceptionally high fire energy or confirmed forest conditions — treated as fire until reviewed."
  },
  fail_closed_or_uncertainty: {
    label: "Model Uncertain",
    color: "#f97316",
    tooltip: "Classifier score was borderline — human judgment required."
  }
};

function getReasonMeta(reason: string) {
  return (
    REASON_META[reason] ?? { label: reason, color: "#6b7280", tooltip: "" }
  );
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
  return num === null ? "n/a" : `${num.toFixed(0)}%`;
}

function formatScoreBand(score: number): string {
  if (score < 0.35) return "Low confidence";
  if (score < 0.45) return "Below threshold";
  if (score < 0.55) return "Borderline";
  if (score < 0.65) return "Above threshold";
  return "High confidence";
}

function sortItems(items: DenoiserReviewItem[]): DenoiserReviewItem[] {
  return [...items].sort((a, b) => {
    const frpA = safeFloat(a.payload_json?.frp_max) ?? 0;
    const frpB = safeFloat(b.payload_json?.frp_max) ?? 0;
    if (frpB !== frpA) return frpB - frpA;
    // oldest first within same FRP tier
    return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
  });
}

function ReasonChip({ reason }: { reason: string }) {
  const { label, color, tooltip } = getReasonMeta(reason);
  const chip = (
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
  return tooltip ? (
    <Tooltip title={tooltip} arrow placement="top">
      {chip}
    </Tooltip>
  ) : (
    chip
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
  const score = item.payload_json?.event_score;
  const sensor = matchedEvent?.sensor ?? null;
  const time = matchedEvent?.end_time ?? item.created_at;
  const isHardBypass = item.reason === HARD_BYPASS;
  const { color: reasonColor } = getReasonMeta(item.reason);

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
        border: `1px solid ${reasonColor}40`,
        cursor: canFocus ? "pointer" : "default",
        transition: "border-color 0.15s",
        "&:hover": canFocus ? { borderColor: `${reasonColor}88` } : {}
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
        {!isHardBypass && score != null && (
          <Box>
            <Typography sx={{ fontSize: 9, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 700 }}>
              Model
            </Typography>
            <Typography sx={{ fontSize: 13, color: "#e5e7eb", fontWeight: 700 }}>
              {formatScoreBand(score)}
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

interface QueueSectionProps {
  title: string;
  color: string;
  items: DenoiserReviewItem[];
  visibleEventIndex: Map<string, FireEvent>;
  onResolve: (eventId: string, notes: ResolutionNote) => void;
  resolvingIds: Set<string>;
  emptyText: string;
}

function QueueSection({
  title,
  color,
  items,
  visibleEventIndex,
  onResolve,
  resolvingIds,
  emptyText
}: QueueSectionProps) {
  const [open, setOpen] = useState(true);

  return (
    <Box>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          px: 0.5,
          py: 0.75,
          cursor: "pointer",
          userSelect: "none"
        }}
        onClick={() => setOpen((v) => !v)}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
          <Typography
            sx={{
              fontSize: 10,
              fontWeight: 800,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color
            }}
          >
            {title}
          </Typography>
          <Box
            sx={{
              px: 0.6,
              py: 0.1,
              borderRadius: 0.6,
              fontSize: 9,
              fontWeight: 800,
              lineHeight: 1.6,
              bgcolor: `${color}22`,
              border: `1px solid ${color}44`,
              color
            }}
          >
            {items.length}
          </Box>
        </Box>
        <Box sx={{ fontSize: 14, color: "#4b5563", display: "flex" }}>
          {open ? (
            <ExpandLessIcon fontSize="inherit" />
          ) : (
            <ExpandMoreIcon fontSize="inherit" />
          )}
        </Box>
      </Box>

      <Collapse in={open}>
        {items.length === 0 ? (
          <Typography sx={{ fontSize: 11, color: "#4b5563", py: 1, px: 0.5 }}>
            {emptyText}
          </Typography>
        ) : (
          <Stack spacing={1}>
            {items.map((item) => (
              <ReviewItemRow
                key={item.event_id}
                item={item}
                matchedEvent={visibleEventIndex.get(item.event_id) ?? null}
                onResolve={onResolve}
                isResolving={resolvingIds.has(item.event_id)}
              />
            ))}
          </Stack>
        )}
      </Collapse>
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

  const hardBypassItems = useMemo(
    () => sortItems(rows.filter((r) => r.reason === HARD_BYPASS)),
    [rows]
  );
  const uncertaintyItems = useMemo(
    () => sortItems(rows.filter((r) => r.reason !== HARD_BYPASS)),
    [rows]
  );

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
          <Stack spacing={0.5} divider={<Divider sx={{ borderColor: "rgba(255,255,255,0.05)" }} />}>
            <QueueSection
              title="High-Energy Alerts"
              color="#ef4444"
              items={hardBypassItems}
              visibleEventIndex={visibleEventIndex}
              onResolve={handleResolve}
              resolvingIds={resolvingIds}
              emptyText="No high-energy alerts pending"
            />
            <QueueSection
              title="Uncertain Detections"
              color="#f97316"
              items={uncertaintyItems}
              visibleEventIndex={visibleEventIndex}
              onResolve={handleResolve}
              resolvingIds={resolvingIds}
              emptyText="No uncertain detections pending"
            />
          </Stack>
        )}
      </Box>
    </Box>
  );
}
