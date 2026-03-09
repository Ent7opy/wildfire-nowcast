import {
  Box,
  Button,
  Card,
  CardContent,
  FormControlLabel,
  Link,
  Slider,
  Stack,
  Switch,
  Typography
} from "@mui/material";

import { apiPublicBaseUrl } from "../config/runtime";
import { useAppStore } from "../state/store";
import { FILTER_PRESETS } from "../utils/presets";
import { computeTimeRange, formatTimeWindow } from "../utils/time";
import { viewportBbox } from "../utils/mapMath";
import { buildFiresCsvExportUrl, buildMapPngExportUrl } from "../api/client";

export default function SidebarControls(): JSX.Element {
  const filters = useAppStore((s) => s.filters);
  const layers = useAppStore((s) => s.layers);
  const mapView = useAppStore((s) => s.mapView);
  const activePreset = useAppStore((s) => s.activePreset);
  const forecast = useAppStore((s) => s.forecast);
  const setFilters = useAppStore((s) => s.setFilters);
  const applyPreset = useAppStore((s) => s.applyPreset);
  const setRiskVisibility = useAppStore((s) => s.setRiskVisibility);
  const clearSelection = useAppStore((s) => s.clearSelection);

  const timeRange = computeTimeRange(filters);
  const bbox = viewportBbox(mapView);

  const csvUrl = buildFiresCsvExportUrl(apiPublicBaseUrl(), {
    bbox,
    startTime: timeRange.startTime,
    endTime: timeRange.endTime,
    limit: 1000
  });

  const pngUrl = buildMapPngExportUrl(apiPublicBaseUrl(), {
    bbox,
    startTime: timeRange.startTime,
    endTime: timeRange.endTime,
    minLikelihood: filters.minLikelihood,
    includeRisk: layers.showRisk,
    runId: forecast.lastForecast?.run.id
  });

  return (
    <Stack spacing={1.5}>
      <Card>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>
            Quick presets
          </Typography>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 1
            }}
          >
            {FILTER_PRESETS.map((preset) => (
              <Box key={preset.name}>
                <Button
                  fullWidth
                  variant={activePreset === preset.name ? "contained" : "outlined"}
                  onClick={() => applyPreset(preset)}
                >
                  {preset.name}
                </Button>
              </Box>
            ))}
            <Box>
              <Button fullWidth variant={activePreset === "Custom" ? "contained" : "outlined"} disabled>
                Custom
              </Button>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>
            Time window
          </Typography>
          <Slider
            value={[filters.hoursEnd, filters.hoursStart]}
            min={0}
            max={48}
            step={1}
            onChange={(_, value) => {
              if (!Array.isArray(value)) return;
              const [end, start] = value;
              setFilters({ hoursEnd: end, hoursStart: Math.max(start, end + 1) });
            }}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value}h`}
          />
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
            {formatTimeWindow(filters)}
          </Typography>

          <Typography variant="subtitle2" gutterBottom>
            Minimum event score
          </Typography>
          <Slider
            value={filters.minLikelihood}
            min={0}
            max={1}
            step={0.05}
            onChange={(_, value) => {
              if (Array.isArray(value)) return;
              setFilters({ minLikelihood: value });
            }}
            valueLabelDisplay="auto"
          />

          <FormControlLabel
            control={
              <Switch
                checked={filters.activeOnly}
                onChange={(event) => setFilters({ activeOnly: event.target.checked })}
              />
            }
            label="Active incidents only"
          />
          <FormControlLabel
            control={
              <Switch
                checked={filters.clusterPoints}
                onChange={(event) => setFilters({ clusterPoints: event.target.checked })}
              />
            }
            label="Cluster nearby points"
          />
          <FormControlLabel
            control={
              <Switch
                checked={layers.showRisk}
                onChange={(event) => setRiskVisibility(event.target.checked)}
                disabled={!filters.clusterPoints}
              />
            }
            label="Include risk index overlay"
          />
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>
            Export current view
          </Typography>
          <Stack spacing={1}>
            <Button component={Link} href={csvUrl} target="_blank" rel="noreferrer" variant="outlined">
              Export fires (CSV)
            </Button>
            <Button component={Link} href={pngUrl} target="_blank" rel="noreferrer" variant="outlined">
              Export map (PNG)
            </Button>
          </Stack>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>
            Map controls
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Pan and zoom to explore. The map updates as you move.
          </Typography>
          <Button onClick={() => clearSelection()} variant="outlined" fullWidth>
            Clear selection
          </Button>
        </CardContent>
      </Card>

      <Box px={1}>
        <Typography variant="caption" color="text.secondary">
          Events filters: {formatTimeWindow(filters)}, event score at least {filters.minLikelihood.toFixed(2)}, active-only={filters.activeOnly ? "on" : "off"}, cluster={filters.clusterPoints ? "on" : "off"}, risk={layers.showRisk ? "on" : "off"}
        </Typography>
      </Box>
    </Stack>
  );
}
