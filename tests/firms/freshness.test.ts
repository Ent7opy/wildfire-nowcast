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
});
