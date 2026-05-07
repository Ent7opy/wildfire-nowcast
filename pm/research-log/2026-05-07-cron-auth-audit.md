# Cron-route authentication audit (`/api/aoi/poll`)

Date: 2026-05-07
Author: scout
Scope: the shared-bearer auth path on `app/api/aoi/poll/route.ts` —
the only route that bypasses Clerk and is invoked by the GitHub Actions
cron (`.github/workflows/firms-poll.yml`). A successful unauthenticated
poll fans out to FIRMS, the AI Gateway, and Resend, so this is a
real-cost endpoint.

## TL;DR

The auth path is in good shape. The five concrete failure modes the
audit was asked to look at are all handled correctly:

1. Constant-time comparison — implemented (custom XOR loop, lines
   518-525), not `===`.
2. Missing-env behaviour — fails closed at line 132 (returns 503
   before parsing the auth header at all).
3. Logging on failed auth — generic `"Invalid bearer token"`; the
   provided token is never echoed into a log line or response body.
4. Method check — only `POST` is exported; App Router auto-405s
   GET/HEAD/PUT/etc. without invoking handler code.
5. Header parsing — `slice("Bearer ".length).trim()` tolerates
   trailing whitespace; a doubled-prefix (`Bearer Bearer X`) passes
   only the first slice, leaves `Bearer X`, and fails the compare.

Tested end-to-end in `tests/poll-route-auth.test.ts` (cases for unset
`CRON_SECRET`, missing bearer, wrong bearer, malformed body, happy
path).

**Nothing concrete to ship.** The remaining notes below are
brainstorm-only or doc-level. Catalogued so a future audit doesn't
re-derive them.

## Shipped this PR

None. Every change I considered was either a no-op refactor or
a behaviour change that would loosen rather than tighten the route.

## Brainstorm-only / out-of-scope

**B1 — Bearer scheme matching is case-sensitive
(`route.ts:140`).** RFC 7235 §2.1 says HTTP auth scheme tokens are
case-insensitive; `auth.startsWith("Bearer ")` would reject
`bearer test-cron-secret` even with the right secret. The only
real-world callers are the GH Actions workflow (sends `Bearer `) and
Vanyo's local debugging (chooses casing). Loosening to
case-insensitive matching would broaden the attack surface (more
inputs accepted) without unlocking any legitimate caller. Leaving
strict on purpose.

**B2 — Length-leak in the constant-time compare
(`route.ts:519`).** The early-return `if (a.length !== b.length)`
gives an attacker a length oracle on the secret. With a 32-byte
random `CRON_SECRET` and an attacker bounded to ~1 req/sec by Vercel
+ FIRMS rate limits, recovering the length costs nothing but doesn't
help — they still need to brute-force 256 bits of entropy of known
length. Not worth widening the function to pad-then-compare.

**B3 — Compare is hand-rolled instead of
`crypto.timingSafeEqual`.** The XOR loop is correct (no
short-circuit, fixed iterations once length matches). Switching to
`timingSafeEqual` would require Buffer construction on both sides
and gain nothing measurable in V8. Mentioning it so a reviewer
who greps for `timingSafeEqual` and finds nothing knows it was a
deliberate choice.

**B4 — No replay protection.** Anyone who captures a valid request
(mitm-capable adversary, leaked Vercel deploy logs, leaked
`act` step output, leaked `CRON_SECRET` env in any repo fork) can
replay forever. This is intrinsic to the shared-bearer model. The
mitigation is operational, not code-level: rotate `CRON_SECRET` if
GH Actions logs are ever made public, and don't echo
`$CRON_SECRET` in workflow steps. The current workflow doesn't
echo it (line 74 inlines it into `-H "Authorization: Bearer
${CRON_SECRET}"`, which `set -x` would expose — and `set -x`
isn't on, just `set -euo pipefail`).

**B5 — Bucket-validation regex on the request body
(`route.ts:94`).** `^5x5:[EW]\d{3}_[NS]\d{2}$` — restrictive enough
that an authed attacker can't smuggle SQL into `body.bucket`. Even
if they could, the value flows into the parametrised
`getActiveBuckets` filter and `bucketToBbox`, both of which only
use it as a key/lookup. No injection surface.

**B6 — Status-code taxonomy nit.** Missing-bearer returns 400
`validation_failed` (line 141-145) rather than 401 `unauthenticated`.
Semantically 401 is more correct per RFC 7235 §3.1, but the GH
Actions workflow's retry logic
(`.github/workflows/firms-poll.yml:83`) already treats 400, 401, and
503 as non-retryable, so behaviour wouldn't change. Skipping —
breaks no test, fixes no real bug.

**B7 — `apiError` JSON for the 401 path is hand-rolled
(`route.ts:148-156`) instead of using `apiError("unauthenticated",
…)`.** The reason is that `apiError` maps `unauthenticated → 401`
which is what the existing call wants — so this could be
collapsed to `return apiError("unauthenticated", "Invalid bearer
token")`. One-line cleanup, no behaviour change. Left it untouched
to keep this audit zero-LOC; flagging for the next chore-loop pass
to fold in if it touches this file anyway.

## Files reviewed

- `app/api/aoi/poll/route.ts` (entire file, focus on lines 130-157,
  518-525)
- `middleware.ts` (confirmed `/api/aoi/poll` is excluded from
  `isProtectedRoute`)
- `lib/api/handlers.ts` (confirmed cron route does not flow through
  `withDb` / `requireUserId`)
- `lib/api/errors.ts` (status-code map)
- `.github/workflows/firms-poll.yml` (confirmed bearer is sent via
  header, not URL; no `set -x`; secret is sourced from
  `secrets.CRON_SECRET`, not `vars`)
- `tests/poll-route-auth.test.ts` (existing test coverage)
