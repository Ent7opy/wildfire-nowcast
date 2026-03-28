import { Box, IconButton, Typography } from "@mui/material";
import ZoomOutMapIcon from "@mui/icons-material/ZoomOutMap";
import { NewsCard } from "./NewsCard";
import type { GdeltArticle } from "./types";

interface FireDetectionsTabProps {
  newsData: GdeltArticle[] | undefined;
  newsLoading: boolean;
  newsError: boolean;
  onExpandNews: () => void;
}

export function FireDetectionsTab({ newsData, newsLoading, newsError, onExpandNews }: FireDetectionsTabProps): JSX.Element {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
      {/* News header */}
      <Box sx={{ px: 2.25, py: 1.2, borderBottom: "1px solid rgba(255,255,255,0.05)", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
        <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#6b7280", letterSpacing: "0.1em", textTransform: "uppercase" }}>
          {newsLoading ? "Loading…" : `${(newsData ?? []).length} reports`}
        </Typography>
        <IconButton
          size="small"
          onClick={onExpandNews}
          disabled={newsLoading || (newsData ?? []).length === 0}
          sx={{ color: "#4b5563", "&:hover": { color: "#9ca3af" }, p: 0.5 }}
        >
          <ZoomOutMapIcon sx={{ fontSize: 13 }} />
        </IconButton>
      </Box>
      {/* Scrollable list */}
      <Box sx={{ flex: 1, overflowY: "auto", p: 2.25, display: "flex", flexDirection: "column", gap: 1.5 }}>
        {newsLoading && (
          <Typography sx={{ fontSize: 12, color: "#6b7280" }}>Loading ground reports...</Typography>
        )}
        {newsError && (
          <Typography sx={{ fontSize: 12, color: "#ef4444" }}>Failed to load news. Check your connection.</Typography>
        )}
        {!newsLoading && !newsError && (newsData ?? []).length === 0 && (
          <Typography sx={{ fontSize: 12, color: "#6b7280" }}>No wildfire reports in the last 12 hours.</Typography>
        )}
        {(newsData ?? []).map((item, idx) => (
          <NewsCard key={idx} item={item} />
        ))}
      </Box>
    </Box>
  );
}
