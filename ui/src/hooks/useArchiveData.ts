import { useCallback, useEffect, useRef, useState } from "react";

import { checkArchiveAvailability, triggerArchiveIngest } from "../api/client";
import { useAppStore } from "../state/store";

export type ArchiveDataStatus = "idle" | "checking" | "ingesting" | "ready" | "error" | "unavailable";

export interface ArchiveDataState {
  status: ArchiveDataStatus;
  message: string | null;
  estimatedMinutes: number | null;
}

const POLL_INTERVAL_MS = 15_000;

export function useArchiveData(): ArchiveDataState {
  const archive = useAppStore((s) => s.archive);
  const { viewMode, archiveDate, archiveTimeframe } = archive;

  const [state, setState] = useState<ArchiveDataState>({ status: "idle", message: null, estimatedMinutes: null });

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    stopPolling();

    if (viewMode !== "archive" || !archiveDate || !archiveTimeframe) {
      setState({ status: "idle", message: null, estimatedMinutes: null });
      return;
    }

    const date = archiveDate;
    const timeframe = archiveTimeframe;

    let cancelled = false;

    async function check(): Promise<boolean> {
      try {
        const result = await checkArchiveAvailability(date, timeframe);
        return result.has_data;
      } catch {
        return false;
      }
    }

    async function run() {
      if (cancelled) return;
      setState({ status: "checking", message: null, estimatedMinutes: null });

      const hasData = await check();
      if (cancelled) return;

      if (hasData) {
        setState({ status: "ready", message: null, estimatedMinutes: null });
        return;
      }

      // No data — trigger ingestion
      let estimatedMinutes = 2;
      try {
        const ingestResult = await triggerArchiveIngest(date, timeframe);
        estimatedMinutes = ingestResult.estimated_minutes;
      } catch (err: unknown) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Ingestion unavailable for this date.";
        setState({ status: "unavailable", message, estimatedMinutes: null });
        return;
      }

      if (cancelled) return;
      setState({
        status: "ingesting",
        message: `Ingesting data for ${date} (${timeframe})… this may take ~${estimatedMinutes} min.`,
        estimatedMinutes,
      });

      // Poll until data is available, giving up after 2× the estimated window
      let pollCount = 0;
      const maxPolls = Math.ceil((estimatedMinutes * 60_000 * 2) / POLL_INTERVAL_MS);
      pollRef.current = setInterval(async () => {
        if (cancelled) {
          stopPolling();
          return;
        }
        pollCount++;
        const ready = await check();
        if (!cancelled && ready) {
          stopPolling();
          setState({ status: "ready", message: null, estimatedMinutes: null });
        } else if (!cancelled && pollCount >= maxPolls) {
          stopPolling();
          setState({
            status: "unavailable",
            message: "Ingest did not complete in time. Check that FIRMS_MAP_KEY is configured.",
            estimatedMinutes: null,
          });
        }
      }, POLL_INTERVAL_MS);
    }

    run();

    return () => {
      cancelled = true;
      stopPolling();
    };
  }, [viewMode, archiveDate, archiveTimeframe, stopPolling]);

  return state;
}
