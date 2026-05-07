# Token-endpoint security audit (Stage 6 + 7)

Date: 2026-05-07
Author: scout
Scope: the five public, unauthenticated, token-bearing endpoints landed in
Stage 6 / Stage 7 — `GET /brief/share/[token]`,
`GET /api/notify/{snooze,pause,unsubscribe,feedback}/[token]`.

## TL;DR

The token plumbing is fundamentally sound: 256-bit `randomBytes` hex tokens,
unique-indexed in the DB, scoped per `(aoi_id, brief_id, action, channel,
target)`, with bounded TTLs (30d for snooze/pause/unsubscribe, 90d for
feedback, 30d for share). Side-effects driven from the loaded row, not URL
params, so no cross-AOI confused-deputy attack. Input is HTML-escaped
through `escapeHtml`; no DOMPurify gap visible.

One concrete fix shipped in this PR; the rest are catalogued below as
brainstorm-only or out-of-scope per the audit's hard rules.

## Shipped this PR

**F1 — Failure-mode oracle on action endpoints (`app/api/notify/_lib/handle.ts:64-102`).**
Before: invalid token → 404 "Link not found", wrong-action token → 400
"Wrong action", expired token → 410 "Link expired", `redeemActionToken`
failure → 400 with the structured reason in the body. An attacker who
somehow obtained a valid token (mailbox compromise, log access, partial
leak) could query all four `/api/notify/{action}/[token]` endpoints and
read off which `action` the token was bound to from the status-code/body
delta. With 256-bit entropy the prereq (knowing a valid token) is the
hard part, but distinguishing failure modes costs us nothing and removes
an oracle. Now: every redemption failure returns the same 404 + opaque
message.

The `?v=` validation on `/feedback` still returns 400 before any DB
lookup — that path doesn't touch the token row, so it leaks nothing.

## Catalogued, not fixed

**F2 — Race in `redeemActionToken` (`lib/notify/action-tokens.ts:139-188`).**
The `loadActionToken` → `UPDATE` pair is non-transactional. Two
concurrent clicks on the same snooze link could both observe
`redeemed_at = null` and both invoke `applySnooze`. In practice this is
benign because every side-effect function is idempotent: snooze takes
`max(current, candidate)`, pause sets a fixed indefinite timestamp,
unsubscribe filters by `(type, target)`, feedback uses
`INSERT ... ON CONFLICT DO UPDATE`. Worth tightening to a single SQL
`UPDATE ... WHERE redeemed_at IS NULL RETURNING ...` if we ever add a
non-idempotent side effect, but no current code path is at risk.

**F3 — Tokens land in Vercel access logs.**
URL-bearer tokens are by construction logged wherever the platform logs
request URLs. Standard practice for magic-link / one-click systems
(Stripe, Substack, Resend itself). Mitigations already in place: short
TTL, idempotent redemption, `Cache-Control: no-store` on responses.
Structural — would need to migrate to header-bearer tokens (breaks the
GET-from-email model) or to single-use POST-with-confirmation (worse
UX). Not worth it at current threat model. Document in a future ADR if
we ever ship to a tenant whose log-access policy is weaker than
Vercel's.

**F4 — Dead-row growth (`aoi_briefs.share_token`, `notify_action_tokens`).**
Expired share tokens are nulled out only when the user revokes via
`clearBriefShareToken` (`lib/db/aoi-repository.ts:660`). An expired
share token row stays `NOT NULL` with `share_expires_at < now()`
forever; `getBriefByShareToken` rejects it but the column squats on the
unique index. Same for `notify_action_tokens` — no purge job. At
current scale (one user, hundreds of briefs / month) this is invisible.
At 10k users it would be a few hundred MB of dead tokens per year, no
correctness risk. Future cleanup: a daily cron that nulls
`share_token` / `share_expires_at` for `share_expires_at < now() - 7d`
and deletes `notify_action_tokens` rows with `expires_at < now() - 7d`.

**F5 — Rate limiting (out of scope per audit hard rules).**
None of the five endpoints have rate limits. Brute-forcing a 256-bit
hex token is not feasible (~2^256 search space), and Vercel's edge
network applies coarse abuse limits, but a per-IP / per-token-prefix
rate limit on the `/api/notify/*` routes would be cheap defence in
depth. Track as a Stage-8+ candidate — probably belongs alongside
whatever observability we add for misuse.

**F6 — `feedback` action, dual-purpose `redeemed_value`.**
By design (`action-tokens.ts:170-186`), a recipient can flip yes ↔ no
indefinitely until token expiry (90d). This is intentional per the code
comment ("for `feedback` we DO want the second click to flip"). Worth
flagging because it means a single feedback token observed in transit
can be flipped by a network attacker; the row in `brief_feedback` is
not authentication-bound. Acceptable for product-quality telemetry,
not acceptable if we ever start using feedback as a trust signal in
ranking.

## Verified clean

- **Token entropy**: 32 bytes from `node:crypto.randomBytes`, hex-encoded.
  256 bits, suitable for bearer-secret use without hashing.
- **String-comparison timing**: equality is DB-side on a unique-indexed
  column, no app-side `===` on hashed material.
- **Cross-AOI / confused deputy**: `applySnooze`/`applyPause`/
  `applyUnsubscribe`/`applyFeedback` all key off `loaded.aoiId` from the
  token row, never from URL params.
- **HTML escaping**: `escapeHtml` covers `& < > " '`. Render paths only
  inject AOI ids and a static title/body — no untrusted user content
  goes in.
- **CORS**: no explicit headers; Next.js default same-origin policy.
  Endpoints are GET-with-bearer-secret, no credentials-required cookies,
  so cross-origin reads are harmless even if they happened.
- **Public share page (`app/brief/share/[token]/page.tsx`)** uses
  `renderMarkdownToHtml` on `brief.renderedMarkdown`. The markdown is
  AI-generated from a system prompt + structured Zod payload, not from
  user input. If we ever let users edit briefs before sharing, audit
  the markdown renderer for XSS.

## Hard rules respected

- No change to token entropy or hashing.
- No rate limiting added.
- No existing tests broken.
- Diff < 50 LOC.
