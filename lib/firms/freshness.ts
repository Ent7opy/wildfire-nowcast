/**
 * Stage 8 — pure mapping from FIRMS client / matcher / brief outcomes onto
 * the user-facing `job_runs.outcome` taxonomy.
 *
 * Operator-facing `status` (ok|partial|error|running) stays on `job_runs`
 * for compatibility — that audience is unchanged. `outcome` is what the
 * AOI-page freshness banner reads.
 */
import type { FirmsFetchErrCode } from "./client";

export type FreshnessOutcome =
  | "success"
  | "rate_limited"
  | "network_error"
  | "timeout"
  | "partial";

export type FreshnessOutcomeWithRetry = {
  outcome: FreshnessOutcome;
  /**
   * Signal (not promise) that the next 15-min cron tick is the retry. Banner
   * copy uses this to render "(retrying)" instead of "(failed)".
   */
  retryPending: boolean;
};

/** FIRMS fetch failure → freshness outcome. */
export function firmsErrorToOutcome(code: FirmsFetchErrCode): FreshnessOutcomeWithRetry {
  switch (code) {
    case "rate_limited":
    case "throttled_local":
      return { outcome: "rate_limited", retryPending: true };
    case "network_error":
      return { outcome: "network_error", retryPending: true };
    case "upstream_error":
    case "parse_error":
      return { outcome: "network_error", retryPending: true };
    case "config_missing":
      // Operator-facing failure; nothing to retry until secret is set. Surface
      // as network_error to the user — the banner gets them attention either
      // way; the operator-facing `status='error'` is the actionable signal.
      return { outcome: "network_error", retryPending: false };
  }
}

/** Bucket-run terminal status → freshness outcome. */
export function runStatusToOutcome(args: {
  status: "ok" | "partial" | "error";
  error?: string | null;
}): FreshnessOutcomeWithRetry {
  if (args.status === "ok") return { outcome: "success", retryPending: false };
  if (args.status === "partial") return { outcome: "partial", retryPending: false };
  // status === 'error' without a more specific code → treat as network error
  // and mark retry pending. AbortError is mapped to timeout below.
  if (args.error && /AbortError|timeout/i.test(args.error)) {
    return { outcome: "timeout", retryPending: true };
  }
  return { outcome: "network_error", retryPending: true };
}
