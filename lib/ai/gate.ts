/**
 * Stage 3 LLM gate — pure-TS implementation of `docs/SPEC-A-prime-v1.md` §Flow 6.
 *
 * Pass conditions (any one is sufficient):
 *   1. First detection for this AOI in the last 72h ("prior_absence")
 *   2. ≥ 2 pixels inside the alert buffer in the current event ("multi_pixel")
 *   3. Any detection with FRP > 5 MW inside the buffer ("high_frp")
 *   4. Any detection within 0.5 × alert_distance_km of the AOI ("close_proximity")
 *
 * Reject conditions (short-circuit before the four pass checks):
 *   - AOI is paused (rules.paused_until in the future)        → "paused"
 *   - Event has already been briefed (last_brief_at != null)  → "already_briefed"
 *
 * The function is pure: callers gather inputs from the DB and call the gate.
 * No network, no DB, no clock dependency beyond an injected `now`.
 */

export type GateInputs = {
  /** Hard reject — used both for "paused" and "already_briefed". */
  pausedUntil: Date | null;
  lastBriefAt: Date | null;
  /**
   * Most recent brief on this AOI across all events. Used for the
   * "first detection in last 72h" check (the prior-absence signal).
   */
  lastAoiEventBriefedAt: Date | null;
  /** Number of detections that contributed to the current event row. */
  detectionCountInEvent: number;
  /** Peak FRP across detections in the current event. */
  peakFrpMw: number | null;
  /** Nearest detection's distance to the AOI polygon, in km. */
  nearestDistanceKm: number;
  /** AOI rule: alert_distance_km (a.k.a. distance_buffer_km). */
  alertDistanceKm: number;
  /** AOI rule: minimum FRP threshold for the high_frp gate (default 5). */
  minFrpMw: number;
  /** Override clock for tests. */
  now?: Date;
};

export type GateReason =
  | "paused"
  | "already_briefed"
  | "prior_absence"
  | "multi_pixel"
  | "high_frp"
  | "close_proximity"
  | "no_signal";

export type GateOutcome =
  | { pass: true; reason: Exclude<GateReason, "paused" | "already_briefed" | "no_signal"> }
  | { pass: false; reason: Extract<GateReason, "paused" | "already_briefed" | "no_signal"> };

const PRIOR_ABSENCE_WINDOW_MS = 72 * 60 * 60 * 1000;

export function evaluateGate(inputs: GateInputs): GateOutcome {
  const now = inputs.now ?? new Date();

  if (inputs.pausedUntil && inputs.pausedUntil.getTime() > now.getTime()) {
    return { pass: false, reason: "paused" };
  }
  if (inputs.lastBriefAt) {
    return { pass: false, reason: "already_briefed" };
  }

  const noPriorBrief = inputs.lastAoiEventBriefedAt == null;
  const priorAbsence =
    noPriorBrief ||
    now.getTime() - inputs.lastAoiEventBriefedAt!.getTime() >= PRIOR_ABSENCE_WINDOW_MS;
  if (priorAbsence) {
    return { pass: true, reason: "prior_absence" };
  }

  if (inputs.detectionCountInEvent >= 2) {
    return { pass: true, reason: "multi_pixel" };
  }

  if (
    inputs.peakFrpMw != null &&
    inputs.peakFrpMw > Math.max(0, inputs.minFrpMw)
  ) {
    return { pass: true, reason: "high_frp" };
  }

  if (inputs.nearestDistanceKm <= 0.5 * inputs.alertDistanceKm) {
    return { pass: true, reason: "close_proximity" };
  }

  return { pass: false, reason: "no_signal" };
}
