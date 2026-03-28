import { Box, Typography } from "@mui/material";
import PublicIcon from "@mui/icons-material/Public";
import type { ReactNode } from "react";

interface AppLayoutProps {
  toolbar: ReactNode;
  mainContent: ReactNode;
  sidebar: ReactNode;
  footer?: ReactNode;
  statsRow?: ReactNode;
  scrubber?: ReactNode;
}

export function AppLayout({ toolbar, mainContent, sidebar, statsRow, scrubber }: AppLayoutProps): JSX.Element {
  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "#010409", color: "#d1d5db" }}>
      <Box
        sx={{
          width: "100%",
          maxWidth: "1600px",
          mx: "auto",
          px: { xs: 2, md: 3, lg: 4 },
          py: { xs: 2, md: 3 },
          display: "flex",
          flexDirection: "column",
          gap: 2.5,
          minHeight: "100vh"
        }}
      >
        {/* Top toolbar row */}
        {toolbar}

        {/* Main two-column grid */}
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", lg: "minmax(0,8fr) minmax(360px,4fr)" },
            gap: 2.5,
            flex: 1,
            minHeight: 0
          }}
        >
          {/* Left column: map + scrubber + stats */}
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5, minHeight: 0 }}>
            <Box sx={{ minHeight: { xs: 420, md: 520 }, flex: 1 }}>
              {mainContent}
            </Box>
            {scrubber}
            {statsRow}
          </Box>

          {/* Right column: sidebar panels */}
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5, minHeight: 0 }}>
            {sidebar}
          </Box>
        </Box>
      </Box>

      {/* Footer */}
      <Box sx={{ borderTop: "1px solid rgba(255,255,255,0.05)", py: 4 }}>
        <Box
          sx={{
            maxWidth: "1600px",
            mx: "auto",
            px: { xs: 2, md: 4 },
            display: "flex",
            flexDirection: { xs: "column", md: "row" },
            alignItems: "center",
            justifyContent: "space-between",
            gap: 1.5
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <PublicIcon sx={{ fontSize: 16, color: "#f97316" }} />
            <Typography sx={{ fontSize: 10, color: "#fff", fontWeight: 900, letterSpacing: "0.2em", textTransform: "uppercase" }}>
              Earth Tools Ecosystem
            </Typography>
          </Box>
          <Typography sx={{ fontSize: 10, color: "#4b5563", fontWeight: 800, letterSpacing: "0.12em", textTransform: "uppercase" }}>
            Open Ecological Intelligence • 2026
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}
