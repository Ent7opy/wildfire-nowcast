/**
 * Tests for useForecastPolling — specifically the run_id=0 falsy short-circuit bug (#370).
 *
 * The bug: `result?.run_id || ""` evaluates to `""` when run_id is 0 (falsy).
 * The fix:  `result?.run_id != null` is used instead, so 0 is treated as a valid run ID.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor, cleanup } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createElement } from "react";

import { useAppStore } from "../state/store";
import { useForecastPolling } from "../hooks/useForecastPolling";
import type { ForecastRequestContext } from "../types/state";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return createElement(QueryClientProvider, { client: queryClient }, children);
  };
}

function freshQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        refetchInterval: false,
        staleTime: 0,
        gcTime: 0
      }
    }
  });
}

function makeFetchMock(payload: unknown) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve(payload)
  });
}

function buildCompletedPayload(run_id: number | string, extras: Record<string, unknown> = {}) {
  return {
    job_id: "job-abc",
    status: "completed",
    result: {
      run_id,
      weather_run_id: null,
      confidence_level: "high",
      fallback_used: false,
      weather_bias_corrected: true,
      ...extras
    }
  };
}

const ACTIVE_FORECAST_STATE = {
  jobId: "job-abc",
  activeRequest: {
    lat: 37.5,
    lon: -122.0,
    locationLabel: "Test Area",
    eventSnapshot: {}
  } as ForecastRequestContext,
  pollCount: 0,
  notification: null,
  lastForecast: null
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useForecastPolling — run_id handling", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = freshQueryClient();

    // Seed store with an active forecast job so polling is enabled.
    // Use merge mode (no second arg) to preserve store action functions.
    useAppStore.setState({
      forecast: ACTIVE_FORECAST_STATE
    });
  });

  afterEach(async () => {
    cleanup();
    vi.unstubAllGlobals();
    // Reset forecast state to default — merge mode to preserve actions
    useAppStore.setState({
      forecast: {
        jobId: null,
        pollCount: 0,
        lastForecast: null,
        activeRequest: null,
        notification: null
      }
    });
    // Clear all cached queries
    await queryClient.resetQueries();
    queryClient.clear();
  });

  it("calls completeForecastJob when run_id is a normal positive integer", async () => {
    vi.stubGlobal("fetch", makeFetchMock(buildCompletedPayload(42)));

    const wrapper = makeWrapper(queryClient);
    renderHook(() => useForecastPolling(), { wrapper });

    await waitFor(() => {
      const { forecast } = useAppStore.getState();
      expect(forecast.jobId).toBeNull();
      expect(forecast.lastForecast?.run?.id).toBe("42");
    });
  });

  it("calls completeForecastJob when run_id is 0 (the falsy bug case)", async () => {
    vi.stubGlobal("fetch", makeFetchMock(buildCompletedPayload(0)));

    const wrapper = makeWrapper(queryClient);
    renderHook(() => useForecastPolling(), { wrapper });

    await waitFor(() => {
      const { forecast } = useAppStore.getState();
      // jobId must be cleared — if the bug is present it would stay "job-abc"
      expect(forecast.jobId).toBeNull();
      // run ID "0" must be recorded correctly
      expect(forecast.lastForecast?.run?.id).toBe("0");
    });
  });

  it("calls clearForecastJob when run_id is missing (null/undefined)", async () => {
    vi.stubGlobal(
      "fetch",
      makeFetchMock({
        job_id: "job-abc",
        status: "completed",
        result: {}
      })
    );

    const wrapper = makeWrapper(queryClient);
    renderHook(() => useForecastPolling(), { wrapper });

    await waitFor(() => {
      const { forecast } = useAppStore.getState();
      // clearForecastJob sets jobId to null but lastForecast stays null (no run recorded)
      expect(forecast.jobId).toBeNull();
      expect(forecast.lastForecast).toBeNull();
    });
  });

  it("sets an error notification and clears job on status=failed", async () => {
    vi.stubGlobal(
      "fetch",
      makeFetchMock({
        job_id: "job-abc",
        status: "failed",
        error: "Model exploded"
      })
    );

    const wrapper = makeWrapper(queryClient);
    renderHook(() => useForecastPolling(), { wrapper });

    await waitFor(() => {
      const { forecast } = useAppStore.getState();
      expect(forecast.jobId).toBeNull();
      expect(forecast.notification?.kind).toBe("error");
      expect(forecast.notification?.message).toContain("Model exploded");
    });
  });
});
