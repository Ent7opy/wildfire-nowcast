/**
 * Stage 4 — Resend email client.
 *
 * Single function: `sendEmail({ to, from, subject, markdown })` returns a
 * typed `SendResult`. Mirrors `lib/ai/gateway.ts`:
 *   - lazy env read (never at import time)
 *   - `RESEND_API_KEY` unset → `{ ok: false, code: "config_missing" }`
 *   - no retry-with-backoff (Stage 4 is single-attempt per poll; the next
 *     poll naturally retries failed rows because the unique idempotency
 *     index excludes `status = 'failed'`)
 *
 * Test mode: `RESEND_TEST_MODE === "1"` rewrites `from` to
 * `onboarding@resend.dev` and appends `[TEST]` to the subject. Production
 * `from` defaults to `onboarding@resend.dev` until sender-domain verification
 * lands (deferred per blocker resolution 2026-05-06).
 *
 * Implementation choice (raw fetch vs the resend SDK): raw fetch — fewer
 * deps, ~30 LOC, and the typed response shape we need is small. Documented
 * in `pm/research-log/2026-05-06-stage4-notification-dispatch.md`.
 */
import { renderMarkdownToHtml } from "./markdown";

export type SendErrCode =
  | "config_missing"
  | "rate_limited"
  | "provider_error"
  | "validation_failed";

export type SendResult =
  | { ok: true; providerMessageId: string; latencyMs: number }
  | { ok: false; code: SendErrCode; message: string; latencyMs: number };

export type SendEmailArgs = {
  to: string;
  from?: string;
  subject: string;
  markdown: string;
  html?: string;
  replyTo?: string;
};

const DEFAULT_FROM = "onboarding@resend.dev";
const TEST_FROM = "onboarding@resend.dev";
const RESEND_ENDPOINT = "https://api.resend.com/emails";
const SUBJECT_MAX = 90;

let loggedSenderOnce = false;

export function buildEnvelope(args: SendEmailArgs): {
  to: string;
  from: string;
  subject: string;
  text: string;
  html: string;
  reply_to?: string;
} {
  const testMode = process.env.RESEND_TEST_MODE === "1";
  const configuredFrom =
    args.from ?? process.env.NOTIFY_FROM_ADDRESS ?? DEFAULT_FROM;
  const from = testMode ? TEST_FROM : configuredFrom;
  const subjectBase = truncate(args.subject, SUBJECT_MAX);
  const subject = testMode ? `${subjectBase} [TEST]` : subjectBase;
  const html = args.html ?? renderMarkdownToHtml(args.markdown);
  const envelope: ReturnType<typeof buildEnvelope> = {
    to: args.to,
    from,
    subject,
    text: args.markdown,
    html,
  };
  if (args.replyTo) envelope.reply_to = args.replyTo;
  return envelope;
}

export function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max);
}

export async function sendEmail(args: SendEmailArgs): Promise<SendResult> {
  const apiKey = process.env.RESEND_API_KEY;
  const startedAt = Date.now();
  if (!apiKey) {
    return {
      ok: false,
      code: "config_missing",
      message: "RESEND_API_KEY is not configured",
      latencyMs: Date.now() - startedAt,
    };
  }

  const envelope = buildEnvelope(args);

  if (
    !loggedSenderOnce &&
    process.env.RESEND_TEST_MODE !== "1" &&
    process.env.NOTIFY_FROM_ADDRESS &&
    process.env.NOTIFY_FROM_ADDRESS !== DEFAULT_FROM
  ) {
    loggedSenderOnce = true;
    console.info(
      `[notify] Resend sender configured: ${process.env.NOTIFY_FROM_ADDRESS}`,
    );
  }

  let res: Response;
  try {
    res = await fetch(RESEND_ENDPOINT, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(envelope),
    });
  } catch (err) {
    return {
      ok: false,
      code: "provider_error",
      message: err instanceof Error ? err.message : String(err),
      latencyMs: Date.now() - startedAt,
    };
  }

  const latencyMs = Date.now() - startedAt;
  if (res.status === 429) {
    return {
      ok: false,
      code: "rate_limited",
      message: `Resend rate limited (429)`,
      latencyMs,
    };
  }

  let body: unknown;
  try {
    body = await res.json();
  } catch {
    body = null;
  }

  if (!res.ok) {
    const message =
      isErrorBody(body) && body.message
        ? body.message
        : `Resend HTTP ${res.status}`;
    const code: SendErrCode = res.status >= 400 && res.status < 500
      ? "validation_failed"
      : "provider_error";
    return { ok: false, code, message, latencyMs };
  }

  const id = isOkBody(body) ? body.id : null;
  if (!id) {
    return {
      ok: false,
      code: "provider_error",
      message: "Resend response missing id",
      latencyMs,
    };
  }
  return { ok: true, providerMessageId: id, latencyMs };
}

function isOkBody(b: unknown): b is { id: string } {
  return (
    typeof b === "object" &&
    b !== null &&
    "id" in b &&
    typeof (b as { id: unknown }).id === "string" &&
    (b as { id: string }).id.length > 0
  );
}

function isErrorBody(b: unknown): b is { message?: string } {
  return typeof b === "object" && b !== null;
}

/** Test-only reset hook for the per-process "logged sender once" flag. */
export function _resetSenderLogForTests(): void {
  loggedSenderOnce = false;
}
