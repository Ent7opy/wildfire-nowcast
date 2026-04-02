import { useState, useEffect } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  InputAdornment,
  Slider,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import NotificationsActiveIcon from "@mui/icons-material/NotificationsActive";
import NotificationsOffIcon from "@mui/icons-material/NotificationsOff";
import RefreshIcon from "@mui/icons-material/Refresh";
import TuneIcon from "@mui/icons-material/Tune";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

import { configureAOIWatch, getWatchlist } from "../api/client";
import type { WatchlistItem } from "../types/api";

const QUERY_KEY = ["aoi-watchlist"];
const REFETCH_INTERVAL_MS = 60_000; // 1 min auto-refresh

function formatRelative(isoStr: string | null | undefined): string {
  if (!isoStr) return "Never";
  const diff = Date.now() - new Date(isoStr).getTime();
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function SpreadProbBar({ value, threshold }: { value: number | null | undefined; threshold: number | null | undefined }) {
  if (value == null) return <Typography variant="caption" color="text.secondary">No data</Typography>;
  const pct = Math.round(value * 100);
  const exceeded = threshold != null && value >= threshold;
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, minWidth: 120 }}>
      <Box
        sx={{
          flex: 1,
          height: 6,
          borderRadius: 3,
          bgcolor: "action.hover",
          overflow: "hidden",
        }}
      >
        <Box
          sx={{
            width: `${pct}%`,
            height: "100%",
            bgcolor: exceeded ? "error.main" : "warning.main",
            borderRadius: 3,
            transition: "width 0.4s",
          }}
        />
      </Box>
      <Typography variant="caption" sx={{ minWidth: 32, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
        {pct}%
      </Typography>
    </Box>
  );
}

interface WatchConfigDialogProps {
  item: WatchlistItem;
  open: boolean;
  onClose: () => void;
}

function WatchConfigDialog({ item, open, onClose }: WatchConfigDialogProps) {
  const [intervalMin, setIntervalMin] = useState<number>(item.watch_interval_minutes ?? 30);
  const [threshold, setThreshold] = useState<number>(item.watch_alert_threshold ?? 0.5);
  const queryClient = useQueryClient();

  useEffect(() => {
    if (open) {
      setIntervalMin(item.watch_interval_minutes ?? 30);
      setThreshold(item.watch_alert_threshold ?? 0.5);
    }
  }, [open, item.watch_interval_minutes, item.watch_alert_threshold]);

  const enable = useMutation({
    mutationFn: () =>
      configureAOIWatch(item.id, {
        enabled: true,
        interval_minutes: intervalMin,
        alert_threshold: threshold,
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: QUERY_KEY });
      onClose();
    },
  });

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>Configure Watch — {item.name}</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ pt: 1 }}>
          <Box>
            <Typography gutterBottom variant="body2">
              Check interval: <strong>{intervalMin} min</strong>
            </Typography>
            <Slider
              min={5}
              max={1440}
              step={5}
              value={intervalMin}
              onChange={(_, v) => setIntervalMin(v as number)}
              marks={[
                { value: 5, label: "5m" },
                { value: 60, label: "1h" },
                { value: 360, label: "6h" },
                { value: 1440, label: "1d" },
              ]}
              valueLabelDisplay="auto"
            />
          </Box>
          <TextField
            label="Alert threshold"
            type="number"
            value={threshold}
            onChange={(e) => setThreshold(Math.min(1, Math.max(0.01, parseFloat(e.target.value) || 0.5)))}
            inputProps={{ min: 0.01, max: 1, step: 0.05 }}
            InputProps={{
              endAdornment: <InputAdornment position="end">prob (0–1)</InputAdornment>,
            }}
            helperText="Alert fires when max spread probability ≥ this value"
            size="small"
          />
          {enable.isError && (
            <Alert severity="error" sx={{ mt: 1 }}>
              {String((enable.error as Error)?.message ?? "Failed to configure watch")}
            </Alert>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="inherit">Cancel</Button>
        <Button
          onClick={() => enable.mutate()}
          variant="contained"
          disabled={enable.isPending}
        >
          {enable.isPending ? <CircularProgress size={18} /> : "Enable Watch"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

function WatchlistRow({ item }: { item: WatchlistItem }) {
  const [configOpen, setConfigOpen] = useState(false);
  const queryClient = useQueryClient();

  const disable = useMutation({
    mutationFn: () => configureAOIWatch(item.id, { enabled: false }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: QUERY_KEY }),
  });

  return (
    <>
      <Card
        variant="outlined"
        sx={{
          borderColor: item.alert_active ? "error.main" : undefined,
          transition: "border-color 0.3s",
        }}
      >
        <CardContent sx={{ py: 1.5, "&:last-child": { pb: 1.5 } }}>
          <Stack direction="row" alignItems="center" spacing={1}>
            {item.alert_active && (
              <Tooltip title="Alert active — spread probability exceeds threshold">
                <WarningAmberIcon color="error" fontSize="small" />
              </Tooltip>
            )}
            <Typography variant="body2" sx={{ flex: 1, fontWeight: 500 }} noWrap>
              {item.name}
            </Typography>

            <Chip
              label={item.watch_enabled ? "watching" : "off"}
              size="small"
              color={item.watch_enabled ? "primary" : "default"}
              variant="outlined"
            />

            {item.watch_enabled && (
              <Tooltip title="Disable watch">
                <IconButton size="small" onClick={() => disable.mutate()} disabled={disable.isPending}>
                  {disable.isPending ? (
                    <CircularProgress size={16} />
                  ) : (
                    <NotificationsOffIcon fontSize="small" />
                  )}
                </IconButton>
              </Tooltip>
            )}
            {!item.watch_enabled && (
              <Tooltip title="Enable watch">
                <IconButton size="small" onClick={() => setConfigOpen(true)}>
                  <NotificationsActiveIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}

            {item.watch_enabled && (
              <Tooltip title="Reconfigure watch">
                <IconButton size="small" onClick={() => setConfigOpen(true)}>
                  <TuneIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </Stack>

          {item.watch_enabled && (
            <Stack direction="row" spacing={2} sx={{ mt: 1 }} alignItems="center">
              <Box sx={{ flex: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Spread prob
                </Typography>
                <SpreadProbBar value={item.watch_last_spread_prob} threshold={item.watch_alert_threshold} />
              </Box>
              <Box sx={{ minWidth: 72, textAlign: "right" }}>
                <Typography variant="caption" color="text.secondary" display="block">
                  Checked
                </Typography>
                <Typography variant="caption">
                  {formatRelative(item.watch_last_checked_at)}
                </Typography>
              </Box>
              {item.watch_alert_threshold != null && (
                <Box sx={{ minWidth: 64, textAlign: "right" }}>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Threshold
                  </Typography>
                  <Typography variant="caption">
                    {Math.round(item.watch_alert_threshold * 100)}%
                  </Typography>
                </Box>
              )}
            </Stack>
          )}
        </CardContent>
      </Card>

      <WatchConfigDialog item={item} open={configOpen} onClose={() => setConfigOpen(false)} />
    </>
  );
}

export default function WatchlistDashboard(): JSX.Element {
  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: QUERY_KEY,
    queryFn: getWatchlist,
    refetchInterval: REFETCH_INTERVAL_MS,
    refetchIntervalInBackground: false,
  });

  const activeAlerts = data?.items.filter((i) => i.alert_active).length ?? 0;

  return (
    <Card>
      <CardContent>
        <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1.5 }}>
          <Typography variant="subtitle2" sx={{ flex: 1 }}>
            AOI Watchlist
          </Typography>
          {activeAlerts > 0 && (
            <Chip
              icon={<WarningAmberIcon />}
              label={`${activeAlerts} alert${activeAlerts !== 1 ? "s" : ""}`}
              color="error"
              size="small"
            />
          )}
          <Tooltip title="Refresh">
            <IconButton size="small" onClick={() => void refetch()} disabled={isFetching}>
              {isFetching ? <CircularProgress size={16} /> : <RefreshIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
        </Stack>

        {isLoading && (
          <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )}

        {isError && (
          <Alert severity="warning" sx={{ mb: 1 }}>
            Could not load watchlist
          </Alert>
        )}

        {data && data.items.length === 0 && (
          <Typography variant="caption" color="text.secondary">
            No watched AOIs. Enable watch on an AOI to monitor it automatically.
          </Typography>
        )}

        {data && data.items.length > 0 && (
          <Stack spacing={1}>
            {data.items.map((item) => (
              <WatchlistRow key={item.id} item={item} />
            ))}
          </Stack>
        )}
      </CardContent>
    </Card>
  );
}
