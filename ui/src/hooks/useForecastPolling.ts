import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";

import { ApiError, ApiUnavailableError, getJitForecastStatus } from "../api/client";
import { useAppStore } from "../state/store";

const MAX_POLLS = 300;

export function useForecastPolling(): void {
  const jobId = useAppStore((s) => s.forecast.jobId);
  const activeRequest = useAppStore((s) => s.forecast.activeRequest);
  const incrementPoll = useAppStore((s) => s.incrementForecastPoll);
  const completeForecastJob = useAppStore((s) => s.completeForecastJob);
  const clearForecastJob = useAppStore((s) => s.clearForecastJob);
  const setForecastNotification = useAppStore((s) => s.setForecastNotification);

  useEffect(() => {
    if (!jobId) {
      return;
    }
    const timer = setTimeout(() => {
      setForecastNotification({
        kind: "error",
        message: "Forecast timed out after 10 minutes.",
        createdAt: Date.now(),
        ttlSeconds: 45
      });
      clearForecastJob();
    }, MAX_POLLS * 2000);
    return () => clearTimeout(timer);
  }, [jobId, clearForecastJob, setForecastNotification]);

  const query = useQuery({
    queryKey: ["jit-forecast-status", jobId],
    queryFn: () => getJitForecastStatus(jobId as string),
    enabled: Boolean(jobId),
    refetchInterval: jobId ? 2000 : false,
    retry: false
  });

  useEffect(() => {
    if (!jobId || !query.data) {
      return;
    }

    const { status } = query.data;
    if (status === "completed") {
      const result = query.data.result;
      if (result?.run_id != null) {
        const runId = String(result.run_id);
        const weatherRunId = (result?.weather_run_id as string | null | undefined) ?? null;
        const confidenceLevel = (result?.confidence_level as string | null | undefined) ?? null;
        const fallbackUsed = Boolean(result?.fallback_used ?? false);
        const weatherBiasApplied = (result?.weather_bias_corrected as boolean | null | undefined) ?? null;
        completeForecastJob(runId, { weatherRunId, confidenceLevel, fallbackUsed, weatherBiasApplied });
        setForecastNotification({
          kind: "ready",
          message: `Forecast for the fire event from ${activeRequest?.locationLabel || "the selected area"} is ready!`,
          createdAt: Date.now(),
          ttlSeconds: 600,
          runId,
          target: {
            lat: activeRequest?.lat,
            lon: activeRequest?.lon,
            eventSnapshot: activeRequest?.eventSnapshot,
            eventId: activeRequest?.eventId,
            eventKey: activeRequest?.eventKey
          }
        });
      } else {
        clearForecastJob();
      }
      return;
    }

    if (status === "failed") {
      setForecastNotification({
        kind: "error",
        message: `Forecast failed: ${query.data.error || "Unknown error"}`,
        createdAt: Date.now(),
        ttlSeconds: 45
      });
      clearForecastJob();
      return;
    }

    incrementPoll();
  }, [
    activeRequest,
    clearForecastJob,
    completeForecastJob,
    incrementPoll,
    jobId,
    query.data,
    setForecastNotification
  ]);

  useEffect(() => {
    if (!jobId || !query.error) {
      return;
    }

    if (query.error instanceof ApiUnavailableError) {
      incrementPoll();
      return;
    }

    if (query.error instanceof ApiError) {
      const message =
        query.error.statusCode === 404
          ? "Job not found. It may have expired."
          : `Error checking job status: ${query.error.message}`;
      setForecastNotification({
        kind: "error",
        message,
        createdAt: Date.now(),
        ttlSeconds: 45
      });
      clearForecastJob();
      return;
    }

    setForecastNotification({
      kind: "error",
      message: "Error checking forecast status.",
      createdAt: Date.now(),
      ttlSeconds: 45
    });
    clearForecastJob();
  }, [jobId, query.error, clearForecastJob, incrementPoll, setForecastNotification]);
}
