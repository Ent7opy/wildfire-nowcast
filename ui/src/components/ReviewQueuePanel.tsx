import React, { useMemo, useState } from "react";
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
import type { DenoiserReviewItem, FireEvent } from "../types/api";
import { ReviewDecisionPanel } from "./ReviewDecisionPanel";

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
  type Keyed = { item: DenoiserReviewItem; frp: number; ts: number };
  const keyed: Keyed[] = items.map((item) => ({
    item,
    frp: safeFloat(item.payload_json?.frp_max) ?? 0,
    ts: new Date(item.created_at).getTime()
  }));
  keyed.sort((a, b) => b.frp !== a.frp ? b.frp - a.frp : a.ts - b.ts);
  return keyed.map((k) => k.item);
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

interface ActionButtonProps {
  tooltip: string;
  icon: React.ReactNode;
  label: string;
  color: string;
  hoverBg: string;
  borderColor: string;
  hoverBorderColor: string;
  isResolving: boolean;
  onClick: () => void;
}

function ActionButton({ tooltip, icon, label, color, hoverBg, borderColor, hoverBorderColor, isResolving, onClick }: ActionButtonProps) {
  return (
    <Tooltip title={tooltip}>
      <span>
        <Button
          size="small"
          disabled={isResolving}
          startIcon={isResolving ? <CircularProgress size={12} /> : icon}
          onClick={onClick}
          sx={{
            fontSize: 10,
            fontWeight: 700,
            py: 0.4,
            px: 1,
            color,
            borderColor,
            border: "1px solid",
            borderRadius: 1,
            textTransform: "none",
            "&:hover": { bgcolor: hoverBg, borderColor: hoverBorderColor },
            "&:disabled": { opacity: 0.4 }
          }}
        >
          {label}
        </Button>
      </span>
    </Tooltip>
  );
}

function countryFlagEmoji(countryCode: string | null | undefined): string {
  if (!countryCode || countryCode.length !== 2) return "";
  const code = countryCode.toUpperCase();
  // Regional indicator symbols: A = 0x1F1E6, offset from 'A' char code 65
  return String.fromCodePoint(
    0x1f1e6 + code.charCodeAt(0) - 65,
    0x1f1e6 + code.charCodeAt(1) - 65
  );
}

function LocationContext({ item }: { item: DenoiserReviewItem }) {
  const { country_code, region_name, nearest_place, terrain_label } = item;
  const hasAny = country_code || region_name || nearest_place || terrain_label;

  if (!hasAny) {
    return (
      <Typography sx={{ fontSize: 10, color: "#4b5563", mb: 0.75, fontStyle: "italic" }}>
        Location unavailable
      </Typography>
    );
  }

  const flag = countryFlagEmoji(country_code);
  const locationLine = [flag, region_name].filter(Boolean).join(" ") || null;

  return (
    <Box sx={{ mb: 0.75 }}>
      {locationLine && (
        <Typography sx={{ fontSize: 11, color: "#9ca3af", lineHeight: 1.4 }}>
          {locationLine}
        </Typography>
      )}
      {nearest_place && (
        <Typography sx={{ fontSize: 10, color: "#6b7280", fontStyle: "italic", lineHeight: 1.4 }}>
          {nearest_place}
        </Typography>
      )}
      {terrain_label && (
        <Typography sx={{ fontSize: 10, color: "#6b7280", lineHeight: 1.4 }}>
          {terrain_label}
        </Typography>
      )}
    </Box>
  );
}

interface ReviewItemRowProps {
  item: DenoiserReviewItem;
  matchedEvent: FireEvent | null;
  onResolve: (eventId: string, notes: ResolutionNote) => void;
  isResolving: boolean;
  expandedEventId: string | null;
  setExpandedEventId: (id: string | null) => void;
}

function ReviewItemRow({ item, matchedEvent, onResolve, isResolving, expandedEventId, setExpandedEventId }: ReviewItemRowProps) {
  const frp = item.payload_json?.frp_max;
  const confidence = item.payload_json?.confidence_max;
  const score = item.payload_json?.event_score;
  const sensor = matchedEvent?.sensor ?? null;
  const time = matchedEvent?.end_time ?? item.created_at;
  const isHardBypass = item.reason === HARD_BYPASS;
  const { color: reasonColor } = getReasonMeta(item.reason);
  const isExpanded = expandedEventId === item.event_id;

  function handleToggleExpand() {
    setExpandedEventId(isExpanded ? null : item.event_id);
  }

  return (
    <Box
      sx={{
        borderRadius: 1.5,
        bgcolor: "#0d1117",
        border: `1px solid ${isExpanded ? `${reasonColor}88` : `${reasonColor}40`}`,
        transition: "border-color 0.15s",
        "&:hover": { borderColor: `${reasonColor}88` }
      }}
    >
      {/* Clickable row header */}
      <Box
        sx={{ p: 1.25, cursor: "pointer" }}
        onClick={handleToggleExpand}
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
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Typography sx={{ fontSize: 10, color: "#4b5563", whiteSpace: "nowrap" }}>
              {formatTimestamp(time)}
            </Typography>
            <Box sx={{ fontSize: 14, color: "#4b5563", display: "flex" }}>
              {isExpanded ? <ExpandLessIcon fontSize="inherit" /> : <ExpandMoreIcon fontSize="inherit" />}
            </Box>
          </Box>
        </Box>

        <Box sx={{ display: "flex", gap: 2, mb: 0.75 }}>
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

        <LocationContext item={item} />

        <Box
          sx={{ display: "flex", gap: 0.75 }}
          onClick={(e) => e.stopPropagation()}
        >
          <ActionButton
            tooltip="Confirm this detection as a real fire"
            icon={<CheckCircleOutlineIcon />}
            label="Confirm Fire"
            color="#4ade80"
            hoverBg="rgba(74,222,128,0.1)"
            borderColor="rgba(74,222,128,0.35)"
            hoverBorderColor="rgba(74,222,128,0.6)"
            isResolving={isResolving}
            onClick={() => onResolve(item.event_id, "confirmed_fire")}
          />
          <ActionButton
            tooltip="Mark this detection as noise / false positive"
            icon={<DoNotDisturbOnIcon />}
            label="Mark as Noise"
            color="#9ca3af"
            hoverBg="rgba(156,163,175,0.08)"
            borderColor="rgba(156,163,175,0.3)"
            hoverBorderColor="rgba(156,163,175,0.5)"
            isResolving={isResolving}
            onClick={() => onResolve(item.event_id, "marked_noise")}
          />
        </Box>
      </Box>

      <Collapse in={isExpanded} unmountOnExit>
        <ReviewDecisionPanel
          eventId={item.event_id}
          borderColor={reasonColor}
          onViewOnMap={() => setExpandedEventId(null)}
        />
      </Collapse>
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
  expandedEventId: string | null;
  setExpandedEventId: (id: string | null) => void;
}

function QueueSection({
  title,
  color,
  items,
  visibleEventIndex,
  onResolve,
  resolvingIds,
  emptyText,
  expandedEventId,
  setExpandedEventId,
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
                expandedEventId={expandedEventId}
                setExpandedEventId={setExpandedEventId}
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
  const [expandedEventId, setExpandedEventId] = useState<string | null>(null);

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

  const [hardBypassItems, uncertaintyItems] = useMemo(() => {
    const hard: DenoiserReviewItem[] = [];
    const uncertain: DenoiserReviewItem[] = [];
    for (const row of rows) {
      (row.reason === HARD_BYPASS ? hard : uncertain).push(row);
    }
    return [sortItems(hard), sortItems(uncertain)];
  }, [rows]);

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
              color={REASON_META.fail_closed_hard_bypass.color}
              items={hardBypassItems}
              visibleEventIndex={visibleEventIndex}
              onResolve={handleResolve}
              resolvingIds={resolvingIds}
              emptyText="No high-energy alerts pending"
              expandedEventId={expandedEventId}
              setExpandedEventId={setExpandedEventId}
            />
            <QueueSection
              title="Uncertain Detections"
              color={REASON_META.fail_closed_or_uncertainty.color}
              items={uncertaintyItems}
              visibleEventIndex={visibleEventIndex}
              onResolve={handleResolve}
              resolvingIds={resolvingIds}
              emptyText="No uncertain detections pending"
              expandedEventId={expandedEventId}
              setExpandedEventId={setExpandedEventId}
            />
          </Stack>
        )}
      </Box>
    </Box>
  );
}
