import { createTheme } from "@mui/material/styles";

export const wildfireTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#f97316"
    },
    secondary: {
      main: "#60a5fa"
    },
    error: {
      main: "#ef4444"
    },
    warning: {
      main: "#f59e0b"
    },
    background: {
      default: "#010409",
      paper: "#0d1117"
    },
    text: {
      primary: "#e5e7eb",
      secondary: "#9ca3af"
    }
  },
  shape: {
    borderRadius: 12
  },
  typography: {
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    h3: {
      fontWeight: 800
    },
    h5: {
      fontWeight: 700
    },
    h6: {
      fontWeight: 600
    }
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        "@keyframes pulse": {
          "0%, 100%": { opacity: 0.45 },
          "50%": { opacity: 1 }
        },
        body: {
          backgroundColor: "#010409"
        }
      }
    },
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
          fontWeight: 700
        }
      }
    },
    MuiOutlinedInput: {
      styleOverrides: {
        notchedOutline: {
          borderColor: "rgba(255,255,255,0.1)"
        },
        root: {
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: "rgba(255,255,255,0.2)"
          },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: "rgba(249,115,22,0.6)"
          }
        }
      }
    }
  }
});
