import { describe, expect, it } from "vitest";
import {
  firmsErrorToOutcome,
  runStatusToOutcome,
} from "@/lib/firms/freshness";

describe("firmsErrorToOutcome", () => {
  it("maps rate_limited and throttled_local to rate_limited+retry", () => {
    expect(firmsErrorToOutcome("rate_limited")).toEqual({
      outcome: "rate_limited",
      retryPending: true,
    });
    expect(firmsErrorToOutcome("throttled_local")).toEqual({
      outcome: "rate_limited",
      retryPending: true,
    });
  });
  it("maps network_error/upstream_error/parse_error to network_error+retry", () => {
    expect(firmsErrorToOutcome("network_error").outcome).toBe("network_error");
    expect(firmsErrorToOutcome("upstream_error").outcome).toBe("network_error");
    expect(firmsErrorToOutcome("parse_error").outcome).toBe("network_error");
    expect(firmsErrorToOutcome("network_error").retryPending).toBe(true);
  });
  it("maps config_missing to network_error+no-retry (operator-actionable)", () => {
    expect(firmsErrorToOutcome("config_missing")).toEqual({
      outcome: "network_error",
      retryPending: false,
    });
  });
});

describe("runStatusToOutcome", () => {
  it("status=ok → success, no retry", () => {
    expect(runStatusToOutcome({ status: "ok" })).toEqual({
      outcome: "success",
      retryPending: false,
    });
  });
  it("status=partial → partial, no retry", () => {
    expect(runStatusToOutcome({ status: "partial" })).toEqual({
      outcome: "partial",
      retryPending: false,
    });
  });
  it("status=error with timeout in error string → timeout, retry", () => {
    expect(runStatusToOutcome({ status: "error", error: "AbortError: timed out" })).toEqual({
      outcome: "timeout",
      retryPending: true,
    });
  });
  it("status=error generic → network_error, retry", () => {
    expect(runStatusToOutcome({ status: "error", error: "boom" })).toEqual({
      outcome: "network_error",
      retryPending: true,
    });
  });
  it("status=error with null/undefined/empty error → network_error, retry (defensive default)", () => {
    // The matcher writes `error: null` when status='error' came from a path that
    // didn't capture an Error.message (e.g. a thrown non-Error object). The
    // banner must still show *something* actionable rather than crashing.
    expect(runStatusToOutcome({ status: "error", error: null })).toEqual({
      outcome: "network_error",
      retryPending: true,
    });
    expect(runStatusToOutcome({ status: "error", error: undefined })).toEqual({
      outcome: "network_error",
      retryPending: true,
    });
    expect(runStatusToOutcome({ status: "error" })).toEqual({
      outcome: "network_error",
      retryPending: true,
    });
    expect(runStatusToOutcome({ status: "error", error: "" })).toEqual({
      outcome: "network_error",
      retryPending: true,
    });
  });
  it("status=error with case-variant timeout strings → timeout (regex /i flag)", () => {
    // Documents that the AbortError|timeout match is case-insensitive — the
    // matcher captures `error.name + ': ' + error.message`, and casing varies
    // across runtime polyfills (Node vs undici vs Vercel edge).
    expect(runStatusToOutcome({ status: "error", error: "aborterror" }).outcome).toBe("timeout");
    expect(runStatusToOutcome({ status: "error", error: "Request TIMEOUT after 30s" }).outcome).toBe("timeout");
    expect(runStatusToOutcome({ status: "error", error: "fetch timeout" }).outcome).toBe("timeout");
  });
});
