import { Box, Paper, Stack, Typography } from "@mui/material";

const fireLegend = [
  { label: "Very high (>=0.8)", color: "rgb(220, 38, 38)" },
  { label: "High (>=0.6)", color: "rgb(239, 68, 68)" },
  { label: "Medium (>=0.4)", color: "rgb(255, 107, 53)" },
  { label: "Low (>=0.2)", color: "rgb(251, 191, 36)" },
  { label: "Very low (<0.2)", color: "rgb(253, 224, 71)" }
];

export default function MapLegend(): JSX.Element {
  return (
    <Paper
      sx={{
        position: "absolute",
        left: 16,
        bottom: 16,
        p: 1.5,
        zIndex: 10,
        width: 250,
        bgcolor: "rgba(26,29,41,0.95)",
        backdropFilter: "blur(4px)"
      }}
    >
      <Typography variant="subtitle2" gutterBottom>
        Fire Severity Legend
      </Typography>
      <Stack spacing={0.75}>
        {fireLegend.map((entry) => (
          <Box key={entry.label} display="flex" alignItems="center" gap={1}>
            <Box width={12} height={12} borderRadius="50%" sx={{ bgcolor: entry.color, border: "1px solid rgba(255,255,255,0.3)" }} />
            <Typography variant="caption" color="text.secondary">
              {entry.label}
            </Typography>
          </Box>
        ))}
      </Stack>
    </Paper>
  );
}
