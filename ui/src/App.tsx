import { useState } from "react";
import { Alert, Box, Paper, Typography } from "@mui/material";

import DataFreshnessBanner from "./components/DataFreshnessBanner";
import FireDetailsPanel from "./components/FireDetailsPanel";
import FireMap from "./components/FireMap";
import ForecastNotification from "./components/ForecastNotification";
import MapLegend from "./components/MapLegend";
import SidebarControls from "./components/SidebarControls";
import { useForecastPolling } from "./hooks/useForecastPolling";
import { useUrlStateSync } from "./hooks/useUrlStateSync";
import type { FireEvent } from "./types/api";

export default function App(): JSX.Element {
  const [visibleEvents, setVisibleEvents] = useState<FireEvent[]>([]);

  useUrlStateSync();
  useForecastPolling();

  return (
    <Box sx={{ minHeight: "100vh", p: 2, bgcolor: "background.default" }}>
      <Typography variant="h5">Wildfire Nowcast & Forecast</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        Live satellite fire events with spread overlays.
      </Typography>
      <Alert severity="info" sx={{ mb: 1 }}>
        Forecast overlays are experimental and probabilistic (not deterministic). Use them as situational awareness, not as operational guidance.
      </Alert>

      <DataFreshnessBanner />
      <ForecastNotification />

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: {
            xs: "1fr",
            md: "320px minmax(0, 1fr) 360px"
          },
          gap: 1.5,
          alignItems: "stretch",
          minHeight: "calc(100vh - 220px)"
        }}
      >
        <Box sx={{ minHeight: 0, overflow: "auto" }}>
          <SidebarControls />
        </Box>

        <Paper sx={{ p: 1, position: "relative", minHeight: 600 }}>
          <Typography variant="h6" sx={{ mb: 1 }}>
            Map
          </Typography>
          <Box sx={{ position: "absolute", inset: "48px 8px 8px 8px" }}>
            <FireMap onVisibleEventsChange={setVisibleEvents} />
            <MapLegend />
          </Box>
        </Paper>

        <Box sx={{ minHeight: 0 }}>
          <FireDetailsPanel visibleEvents={visibleEvents} />
        </Box>
      </Box>
    </Box>
  );
}
