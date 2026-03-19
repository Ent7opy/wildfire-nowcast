import { useEffect } from "react";
import { Alert, Box, Button, Collapse } from "@mui/material";

import { useAppStore } from "../state/store";

export default function ForecastNotification(): JSX.Element | null {
  const notification = useAppStore((s) => s.forecast.notification);
  const setNotification = useAppStore((s) => s.setForecastNotification);
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);
  const setSelectedEvent = useAppStore((s) => s.setSelectedEvent);
  const setLastClick = useAppStore((s) => s.setLastClick);

  useEffect(() => {
    if (!notification || !(notification.ttlSeconds > 0)) return;
    const remaining = notification.createdAt + notification.ttlSeconds * 1000 - Date.now();
    if (remaining <= 0) {
      setNotification(null);
      return;
    }
    const timer = setTimeout(() => setNotification(null), remaining);
    return () => clearTimeout(timer);
  }, [notification, setNotification]);

  if (!notification) {
    return null;
  }

  const expired = notification.ttlSeconds > 0 && Date.now() - notification.createdAt > notification.ttlSeconds * 1000;
  if (expired) {
    return null;
  }

  const severity = notification.kind === "ready" ? "success" : notification.kind === "error" ? "error" : "info";
  const target = notification.target;

  return (
    <Collapse in>
      <Alert
        severity={severity}
        sx={{ mb: 1 }}
        action={
          notification.kind === "ready" && target ? (
            <Box display="flex" gap={1}>
              <Button
                color="inherit"
                size="small"
                onClick={() => {
                  const lat = target.lat;
                  const lon = target.lon;
                  if (typeof lat === "number" && typeof lon === "number") {
                    focusMapOnPoint(lat, lon, 7);
                    if (target.eventSnapshot) {
                      setSelectedEvent(target.eventSnapshot);
                    }
                    setLastClick({ lat, lng: lon });
                  }
                  setNotification(null);
                }}
              >
                Open
              </Button>
              <Button color="inherit" size="small" onClick={() => setNotification(null)}>
                Dismiss
              </Button>
            </Box>
          ) : (
            <Button color="inherit" size="small" onClick={() => setNotification(null)}>
              Dismiss
            </Button>
          )
        }
      >
        {notification.message}
      </Alert>
    </Collapse>
  );
}
