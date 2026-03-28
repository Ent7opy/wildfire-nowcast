import { Box, Dialog, DialogContent, DialogTitle, IconButton, Typography } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import NewspaperIcon from "@mui/icons-material/Newspaper";
import { NewsCard } from "./NewsCard";
import type { GdeltArticle } from "./types";

interface NewsExpandedModalProps {
  open: boolean;
  articles: GdeltArticle[];
  onClose: () => void;
}

export function NewsExpandedModal({ open, articles, onClose }: NewsExpandedModalProps): JSX.Element {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: "#0d1117",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 3,
          boxShadow: "0 32px 100px rgba(0,0,0,0.6)",
          maxHeight: "85vh"
        }
      }}
    >
      <DialogTitle sx={{ px: 3, py: 2, borderBottom: "1px solid rgba(255,255,255,0.05)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <NewspaperIcon sx={{ fontSize: 16, color: "#f97316" }} />
          <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#fff", letterSpacing: "0.1em", textTransform: "uppercase" }}>
            Ground Reports
          </Typography>
          <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#6b7280", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            · {articles.length} reports · last 12h
          </Typography>
        </Box>
        <IconButton onClick={onClose} size="small" sx={{ color: "#6b7280", "&:hover": { color: "#fff" } }}>
          <CloseIcon sx={{ fontSize: 16 }} />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ p: 3 }}>
        <Box sx={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
          gap: 2,
          pt: 0.5
        }}>
          {articles.map((item, idx) => (
            <NewsCard key={idx} item={item} expanded />
          ))}
          {articles.length === 0 && (
            <Typography sx={{ fontSize: 13, color: "#6b7280", gridColumn: "1 / -1" }}>
              No wildfire reports found.
            </Typography>
          )}
        </Box>
      </DialogContent>
    </Dialog>
  );
}
