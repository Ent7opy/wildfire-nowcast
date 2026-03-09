import { Alert, Box, Chip, CircularProgress, Stack, Typography } from "@mui/material";
import { useQuery } from "@tanstack/react-query";

import { getDataFreshnessStatus } from "../api/client";

const ORDERED_SOURCES = ["firms", "weather", "terrain", "perimeters"];

function statusColor(state: string | undefined): "success" | "warning" | "error" | "default" {
  const normalized = (state || "").toLowerCase();
  if (normalized === "fresh") {
    return "success";
  }
  if (normalized === "stale") {
    return "warning";
  }
  if (normalized === "missing") {
    return "error";
  }
  return "default";
}

function statusLabel(state: string | undefined): string {
  const normalized = (state || "").toLowerCase();
  if (normalized === "fresh") {
    return "Fresh";
  }
  if (normalized === "stale") {
    return "Stale";
  }
  if (normalized === "missing") {
    return "Missing";
  }
  return "Unknown";
}

export default function DataFreshnessBanner(): JSX.Element {
  const query = useQuery({
    queryKey: ["health-data-freshness"],
    queryFn: getDataFreshnessStatus,
    refetchInterval: 60_000
  });

  if (query.isLoading) {
    return (
      <Box display="flex" alignItems="center" gap={1} py={1}>
        <CircularProgress size={16} />
        <Typography variant="body2" color="text.secondary">
          Loading data freshness...
        </Typography>
      </Box>
    );
  }

  if (query.isError) {
    return <Alert severity="warning">Data freshness status is unavailable.</Alert>;
  }

  const snapshot = query.data;
  if (!snapshot) {
    return <Alert severity="warning">Data freshness status is unavailable.</Alert>;
  }
  const sources = snapshot.sources || {};

  return (
    <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" py={1}>
      {ORDERED_SOURCES.map((source) => {
        const details = sources[source] || {};
        const label = `${source.toUpperCase()} ${statusLabel(details.state)}${
          details.age_minutes !== undefined ? ` · ${Number(details.age_minutes).toFixed(1)}m` : ""
        }`;
        return <Chip key={source} label={label} color={statusColor(details.state)} variant="outlined" size="small" />;
      })}
    </Stack>
  );
}
