import { Box, Typography } from "@mui/material";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import type { GdeltArticle } from "./types";
import { formatSeenDate } from "./types";

export function NewsCard({ item, expanded = false }: { item: GdeltArticle; expanded?: boolean }): JSX.Element {
  return (
    <Box
      component="a"
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      sx={{
        display: "block",
        flexShrink: 0,
        border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 2.5,
        overflow: "hidden",
        textDecoration: "none",
        transition: "all 160ms ease",
        bgcolor: "rgba(22,27,34,0.5)",
        "&:hover": { borderColor: "rgba(249,115,22,0.3)", bgcolor: "#1c2128" }
      }}
    >
      {item.socialimage && (
        <Box sx={{ position: "relative", height: expanded ? 140 : 110, overflow: "hidden" }}>
          <Box
            component="img"
            src={item.socialimage}
            alt=""
            sx={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              filter: "grayscale(100%)",
              transition: "filter 500ms ease",
              "&:hover": { filter: "grayscale(0%)" }
            }}
          />
          {item.sourcecountry && (
            <Box sx={{
              position: "absolute", top: 6, left: 6, px: 0.75, py: 0.25,
              bgcolor: "rgba(0,0,0,0.65)", backdropFilter: "blur(8px)",
              borderRadius: 0.75, border: "1px solid rgba(255,255,255,0.1)"
            }}>
              <Typography sx={{ fontSize: 8, fontWeight: 700, color: "#fff", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                {item.sourcecountry}
              </Typography>
            </Box>
          )}
        </Box>
      )}
      <Box sx={{ p: 1.5 }}>
        <Typography sx={{
          fontSize: expanded ? 12 : 11, fontWeight: 700, color: "#d1d5db", lineHeight: 1.4,
          display: "-webkit-box", WebkitLineClamp: expanded ? 3 : 2,
          WebkitBoxOrient: "vertical", overflow: "hidden"
        }}>
          {item.title}
        </Typography>
        <Box sx={{ mt: 1, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <Typography sx={{ fontSize: 9, fontWeight: 700, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            {formatSeenDate(item.seendate)}
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.4 }}>
            <Typography sx={{ fontSize: 9, fontWeight: 700, color: "rgba(249,115,22,0.5)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
              Read
            </Typography>
            <OpenInNewIcon sx={{ fontSize: 9, color: "rgba(249,115,22,0.5)" }} />
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
