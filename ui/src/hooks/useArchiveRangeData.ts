import { useCallback, useEffect, useRef, useState } from "react";

import { getArchiveRangeStatus, triggerArchiveRangeIngest } from "../api/client";
import type { ArchiveRangeDayStatus } from "../api/client";
import { useAppStore } from "../state/store";

export type ArchiveRangeStatus =
  | "idle"
  | "loading"       // ingest job enqueued, days being processed
  | "ready"          // all (or some) days finished
  | "error"
  | "unavailable";

export interface ArchiveRangeDataState {
  status: ArchiveRangeStatus;
  dayStatuses: ArchiveRangeDayStatus[];
  completedCount: number;
  totalCount: number;
  message: string | null;
  warning: string | null;
}

const POLL_INTERVAL_MS = 5_000;

export function useArchiveRangeData(): ArchiveRangeDataState {
  const archive = useAppStore((s) => s.archive);
  const setRangeJobId = useAppStore((s) => s.setRangeJobId);
  const { viewMode, archiveSubMode, rangeStart, rangeEnd, rangeJobId } = archive;

  const [state, setState] = useState<ArchiveRangeDataState>({
    status: "idle",
    dayStatuses: [],
    completedCount: 0,
    totalCount: 0,
    message: null,
    warning: null,
  });

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);
  // Track last seen poll result to avoid no-op setState calls every 5 s
  const lastPollKeyRef = useRef<string | null>(null);

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

    if (viewMode !== "archive" || archiveSubMode !== "range" || !rangeStart || !rangeEnd) {
      setState({ status: "idle", dayStatuses: [], completedCount: 0, totalCount: 0, message: null, warning: null });
      return;
    }

    const start = rangeStart;
    const end = rangeEnd;
    let cancelled = false;
    let activeJobId = rangeJobId;

    async function pollStatus(jobId: string) {
      if (cancelled) return;

      pollRef.current = setInterval(async () => {
        if (cancelled) {
          stopPolling();
          return;
        }
        try {
          const result = await getArchiveRangeStatus(jobId);
          if (cancelled) return;

          const isDone =
            result.overall_status === "completed" ||
            result.overall_status === "partial_failure" ||
            result.overall_status === "not_found";

          // Skip setState when nothing has changed to avoid spurious re-renders
          const pollKey = `${result.overall_status}:${result.completed_count}`;
          if (pollKey !== lastPollKeyRef.current) {
            lastPollKeyRef.current = pollKey;
            setState((prev) => ({
              ...prev,
              status: isDone ? "ready" : "loading",
              dayStatuses: result.days,
              completedCount: result.completed_count,
              totalCount: result.total_count,
              message: isDone
                ? null
                : `Loading range: ${result.completed_count}/${result.total_count} days ready…`,
            }));
          }

          if (isDone) {
            stopPolling();
          }
        } catch {
          // Keep polling on transient errors
        }
      }, POLL_INTERVAL_MS);
    }

    async function run() {
      if (cancelled) return;

      // If we already have a job ID (e.g., store was preserved across re-render), go straight to polling
      if (activeJobId) {
        setState((prev) => ({
          ...prev,
          status: "loading",
          message: `Resuming range ingest…`,
        }));
        await pollStatus(activeJobId);
        return;
      }

      setState({ status: "loading", dayStatuses: [], completedCount: 0, totalCount: 0, message: "Enqueueing range ingest…", warning: null });

      try {
        const result = await triggerArchiveRangeIngest(start, end);
        if (cancelled) return;

        activeJobId = result.range_job_id;
        setRangeJobId(result.range_job_id);

        setState((prev) => ({
          ...prev,
          status: "loading",
          totalCount: result.dates.length,
          message: `Loading range: 0/${result.dates.length} days ready… (~${result.estimated_minutes} min total)`,
          warning: result.warning,
        }));

        await pollStatus(result.range_job_id);
      } catch (err: unknown) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Failed to start range ingest.";
        setState({ status: "unavailable", dayStatuses: [], completedCount: 0, totalCount: 0, message, warning: null });
      }
    }

    run();

    return () => {
      cancelled = true;
      stopPolling();
    };
  // rangeJobId is intentionally excluded: the hook itself writes it via setRangeJobId, so
  // including it would cause a self-triggered re-run loop every time the job is enqueued.
  // activeJobId (local var) already handles the "resume existing job" path on mount.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode, archiveSubMode, rangeStart, rangeEnd, setRangeJobId, stopPolling]);

  return state;
}
