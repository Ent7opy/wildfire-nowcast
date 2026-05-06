/**
 * Stage 7 — shared handler for the four notify-action GET endpoints.
 *
 * All four routes are unauthenticated (token is the bearer secret) and
 * idempotent on re-click. The HTML response is intentionally tiny — no
 * Tailwind, no client JS — so a mail-client preview pane can render it
 * without surprises.
 *
 * GET-vs-POST: email clients open links via GET, so each route is safe
 * against bot prefetching by virtue of the secret in the URL. A scanner-
 * induced snooze self-resolves in 24h; a scanner-induced pause is
 * recoverable from the dashboard.
 */
import { NextResponse } from "next/server";
import { tryGetDb } from "@/lib/db/client";
import {
  loadActionToken,
  redeemActionToken,
  type ActionKind,
} from "@/lib/notify/action-tokens";
import {
  applySnooze,
  applyPause,
  applyUnsubscribe,
  applyFeedback,
  type FeedbackValue,
} from "@/lib/notify/actions";

export type ActionRouteContext = {
  params: Promise<{ token: string }>;
};

export async function handleAction(
  action: ActionKind,
  ctx: ActionRouteContext,
  opts: { feedbackValue?: FeedbackValue } = {},
): Promise<NextResponse> {
  const { token } = await ctx.params;
  const db = tryGetDb();
  if (!db) {
    return htmlResponse(
      503,
      renderError(
        "Service unavailable",
        "The database is not configured. Please try again later.",
      ),
    );
  }

  const now = new Date();

  // Pre-load so we can render specific failure messages.
  const loaded = await loadActionToken(db, token);
  if (!loaded) {
    return htmlResponse(
      404,
      renderError("Link not found", "This action link is invalid or has been deleted."),
    );
  }
  if (loaded.action !== action) {
    return htmlResponse(
      400,
      renderError(
        "Wrong action",
        "This link does not match the requested action.",
      ),
    );
  }
  if (loaded.expiresAt.getTime() <= now.getTime()) {
    return htmlResponse(
      410,
      renderError(
        "Link expired",
        "This link has expired. You can adjust this AOI from the dashboard.",
      ),
    );
  }

  const result = await redeemActionToken(db, {
    token,
    expectedAction: action,
    redeemedValue: opts.feedbackValue,
    now,
  });
  if (!result.ok) {
    return htmlResponse(
      400,
      renderError("Could not redeem", `Redemption failed: ${result.reason}`),
    );
  }

  if (action === "snooze") {
    let pausedUntilStr = "";
    if (result.first) {
      const out = await applySnooze(db, result.loaded, now);
      pausedUntilStr = out.pausedUntil.toISOString();
    } else {
      pausedUntilStr = "(already snoozed)";
    }
    return htmlResponse(
      200,
      renderConfirmation({
        title: "Snoozed for 24 hours",
        body: `Notifications for this AOI are paused until ${pausedUntilStr}. You can resume from the dashboard.`,
        aoiId: result.loaded.aoiId,
      }),
    );
  }

  if (action === "pause") {
    if (result.first) {
      await applyPause(db, result.loaded, now);
    }
    return htmlResponse(
      200,
      renderConfirmation({
        title: "Paused",
        body: "Notifications for this AOI are paused indefinitely. You can resume from the dashboard.",
        aoiId: result.loaded.aoiId,
      }),
    );
  }

  if (action === "unsubscribe") {
    let outcomeText = "(already unsubscribed)";
    if (result.first) {
      const out = await applyUnsubscribe(db, result.loaded, now);
      outcomeText = out.autoPaused
        ? "Email removed and AOI paused (no remaining notification channels)."
        : `Email removed. ${out.remainingChannels.length} channel(s) remain.`;
    }
    return htmlResponse(
      200,
      renderConfirmation({
        title: "Unsubscribed",
        body: outcomeText,
        aoiId: result.loaded.aoiId,
      }),
    );
  }

  if (action === "feedback") {
    if (!opts.feedbackValue) {
      return htmlResponse(
        400,
        renderError("Missing value", "Feedback link is missing the ?v=yes|no parameter."),
      );
    }
    if (!result.loaded.briefId) {
      return htmlResponse(
        400,
        renderError("No brief", "This feedback link is not associated with a brief."),
      );
    }
    await applyFeedback(db, result.loaded, opts.feedbackValue, now);
    return htmlResponse(
      200,
      renderConfirmation({
        title: "Thanks for the feedback",
        body:
          opts.feedbackValue === "yes"
            ? "Glad it was useful — we use this to keep the briefs sharp."
            : "Noted — we'll keep working on the signal-to-noise.",
        aoiId: result.loaded.aoiId,
      }),
    );
  }

  return htmlResponse(500, renderError("Unhandled action", action));
}

function htmlResponse(status: number, body: string): NextResponse {
  return new NextResponse(body, {
    status,
    headers: {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" : c === '"' ? "&quot;" : "&#39;",
  );
}

function renderConfirmation(args: { title: string; body: string; aoiId: string }): string {
  const title = escapeHtml(args.title);
  const body = escapeHtml(args.body);
  const aoiId = escapeHtml(args.aoiId);
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${title} — Wildfire Nowcast</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body { font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
         max-width: 32rem; margin: 4rem auto; padding: 0 1rem; line-height: 1.5; color: #111; }
  h1 { font-size: 1.25rem; margin-bottom: .5rem; }
  p  { margin: .25rem 0 1rem; }
  a  { color: #1f5fb8; }
</style>
</head>
<body>
  <h1>${title}</h1>
  <p>${body}</p>
  <p><a href="/dashboard/aoi/${aoiId}">Open this AOI in the dashboard</a></p>
</body>
</html>`;
}

function renderError(title: string, message: string): string {
  const t = escapeHtml(title);
  const m = escapeHtml(message);
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${t} — Wildfire Nowcast</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body { font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
         max-width: 32rem; margin: 4rem auto; padding: 0 1rem; line-height: 1.5; color: #111; }
  h1 { font-size: 1.25rem; margin-bottom: .5rem; }
</style>
</head>
<body>
  <h1>${t}</h1>
  <p>${m}</p>
  <p><a href="/dashboard">Open dashboard</a></p>
</body>
</html>`;
}
