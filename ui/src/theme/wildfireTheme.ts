import { createTheme } from "@mui/material/styles";

export const wildfireTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#ff6b35"
    },
    secondary: {
      main: "#fbbf24"
    },
    error: {
      main: "#e63946"
    },
    background: {
      default: "#0a1628",
      paper: "#252930"
    },
    text: {
      primary: "#e0e0e0",
      secondary: "rgba(255,255,255,0.7)"
    }
  },
  shape: {
    borderRadius: 8
  },
  typography: {
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    h5: {
      fontWeight: 700
    },
    h6: {
      fontWeight: 600
    }
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          border: "1px solid rgba(255,255,255,0.08)",
          backgroundImage: "none"
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 600
        }
      }
    }
  }
});
