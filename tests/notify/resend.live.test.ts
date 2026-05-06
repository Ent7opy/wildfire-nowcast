/**
 * Live Resend smoke test — gated behind `RESEND_LIVE=1`.
 *
 * Off by default (CI and local). Run manually after Vanyo wires up
 * `RESEND_API_KEY` to confirm an actual email lands. Uses
 * `RESEND_TEST_MODE=1` so the send goes through Resend's test sender
 * (`onboarding@resend.dev`) — no domain verification required.
 *
 *   RESEND_LIVE=1 RESEND_API_KEY=... pnpm vitest run tests/notify/resend.live
 *
 * Skipped silently when the flag isn't set so `pnpm test` stays clean.
 */
import { describe, expect, it } from "vitest";
import { sendEmail } from "@/lib/notify/resend";

const live = process.env.RESEND_LIVE === "1" && !!process.env.RESEND_API_KEY;
const describeLive = live ? describe : describe.skip;

if (!live) {
  console.warn(
    "[live] Skipping Resend live test — set RESEND_LIVE=1 and RESEND_API_KEY to enable.",
  );
}

describeLive("Resend live", () => {
  it("sends a real email through onboarding@resend.dev", async () => {
    process.env.RESEND_TEST_MODE = "1";
    const to = process.env.RESEND_LIVE_TO ?? "delivered@resend.dev";
    const result = await sendEmail({
      to,
      subject: "Wildfire Nowcast — Stage 4 live smoke test",
      markdown:
        "# Live smoke test\n\nThis is a real Resend send via the test sender.\n",
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.providerMessageId.length).toBeGreaterThan(0);
    expect(result.latencyMs).toBeLessThan(10_000);
  }, 30_000);
});
