# Brief 19 — Stage 5: auth (Clerk) + per-user AOIs

## Why this exists

Stage 5 of the A' pivot. Stages 1–4 built the full detection → match →
brief → email pipeline against a single hard-coded `STUB_USER_ID =
"stub-user-1"`. Every AOI in the database belongs to that one stub row;
every API call ignores who is calling. Stage 5 closes that gap: real
users sign in via Clerk, AOIs become **owned** by the calling user, the
poll iterates **all** users' AOIs, and the email dispatcher sends to the
real user's account email pulled from Clerk.

This is the stage that makes US-1 ("first AOI in under 5 minutes" — the
spec opens with "Sign-in via Clerk (Google / email OTP). No credit
card.") and US-7 ("BYO Gemini key for heavy users" — needs a per-user
identity to attach the key to) addressable. It is also the precondition
for Stage 6's signed snooze/pause links (US-5) which need a user
identity to bind the token to.

**Read in order:**
1. `pm/PM_CLAUDE.md`
2. `pm/decisions/0005-problem-chosen-a-prime.md`
3. `pm/decisions/0006-stage-pr-workflow.md` — workflow you follow (branch off `master`, draft PR, PM_CLAUDE opens it)
4. `pm/decisions/0007-*` (auto-merge gate update — stage PRs auto-merge once CI is green and the reviewer LGTMs; do NOT instruct anyone to "wait for Vanyo")
5. `pm/blockers.md` — Stage 5 (Clerk) keys are already provisioned (resolved 2026-05-06). `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY` are on Vercel Preview + Production. Verified via `vercel env ls`.
6. `docs/SPEC-A-prime-v1.md` — §User stories US-1 (5-minute path through Clerk), US-2 (per-AOI rules — these now live under one identity), US-7 (BYO key); §Open questions item 1 (Vanyo defaults to Clerk); §Data model `users` table shape (`clerk_id text unique`, `email`, `gemini_key_ciphertext`); §API surface (`GET /api/me`, the auth surface)
7. `docs/pivot-architecture.md` — §1 architecture diagram (Clerk-issued id is the PK in `users`); §3.1 binding `users` SQL shape — `id TEXT PRIMARY KEY` is the Clerk user_id directly; §6 R-table risk R1 / R-bottom: "Clerk 10k MAU ceiling (trivial to migrate to Supabase Auth)"
8. `pm/briefs/18-stage4-notification-dispatch.md` and `pm/briefs/17-stage3-brief-generation.md` — structural template; Stage 5 plugs into the same poll route's user-iteration shape and replaces the stub-user assumptions in `lib/notify/dispatch.ts`'s `users.email` lookup
9. `db/schema/index.ts` — current `users` table (Stage 1, keyed on `text` already so Clerk ids drop in cleanly), `aois.user_id` FK, the literal `export const STUB_USER_ID = "stub-user-1"` that this stage removes
10. `lib/api/current-user.ts` — the single resolver that returns `STUB_USER_ID` today; this is the seam Stage 5 cuts at
11. `app/api/aoi/route.ts`, `app/api/aoi/[id]/route.ts`, `app/api/aoi/[id]/rules/route.ts`, `app/api/aoi/poll/route.ts` — every route currently funnels through `withDb` → `currentUserId()`; Stage 5 makes that function async and Clerk-backed
12. `lib/notify/dispatch.ts` — `LoadedBrief.userEmail` is sourced from the `users` table today; after Stage 5 the row is populated by the Clerk webhook
13. `app/layout.tsx` — currently a bare HTML shell; Stage 5 wraps it with `<ClerkProvider>`
14. Clerk Next.js docs: https://clerk.com/docs/quickstarts/nextjs (App Router middleware + `auth()` + `<ClerkProvider>` + the `[[...rest]]` sign-in/sign-up route convention) and https://clerk.com/docs/integrations/webhooks/sync-data (Svix-signed `user.created` / `user.updated` / `user.deleted` webhooks)

## Goal

Land — on a `stage-5-clerk-auth` branch off `master` — Clerk-authenticated
sign-in/sign-up flows, route-level auth on `/api/aoi/*`, a Svix-verified
Clerk webhook that syncs the `users` table on `user.created` /
`user.updated` / `user.deleted`, removal of every `STUB_USER_ID`
reference from runtime code paths, per-user AOI isolation enforced at
the repository layer, the cron poll iterating **all** non-archived AOIs
across **all** users, and the unit + integration tests covering all of
it. Build-without-blocking discipline holds: if Clerk env vars are
missing, the app builds and starts; authenticated routes return a clear
`config_missing` error rather than crashing. PR draft markdown ready;
PM_CLAUDE opens the PR.

## Scope (strict)

### Schema additions (new migration `db/migrations/0004_stage5.sql` + `0004_stage5.test.sql`)

The Stage 1 `users` table is already keyed on `text` (per
`docs/pivot-architecture.md` §3.1: "Clerk-issued id is the PK"). The
column shape is correct for Clerk drop-in. Two changes only:

- **Drop the seeded `stub-user-1` row** as the migration's last step.
  Defensive: do this in a `DELETE FROM users WHERE id = 'stub-user-1'`
  inside a `DO $$ ... $$` block that first verifies the row's
  `created_at` matches the seed timestamp (or simply swallows
  `no rows affected`). Cascading FKs will null/cascade per their
  existing `ON DELETE` rules (`aois` is `CASCADE` — that is acceptable
  because we are dropping a stub identity, not a real user; verify by
  asserting the test seed creates a *fresh* Clerk-style user before
  deleting the stub).
- **No new tables.** The `gemini_api_key_enc bytea` column on `users`
  already exists from Stage 1 — Stage 5 does NOT touch it (US-7 is out
  of scope per spec §Open questions item 2).
- **No `aois.user_id` change.** The FK is already `text NOT NULL
  REFERENCES users(id) ON DELETE CASCADE`; Clerk user ids slot in.

Do NOT widen the schema. The Clerk webhook handler INSERTs / UPDATEs /
soft-deletes `users` rows directly; no new tables for sessions or
provider mapping (Clerk owns that).

Hand-author SQL same as Stages 1–4 (`0004_stage5.sql` for Neon,
`0004_stage5.test.sql` for PGlite). No PostGIS in this migration.
Update `db/schema/index.ts` to **remove** `export const STUB_USER_ID`
(this is a deliberate breaking change — every consumer is updated in
the same PR).

### Auth context (`lib/auth/context.ts`)

Single async function: `requireUserId(): Promise<{ userId: string } | { error: AuthErrorCode }>`.

```ts
export type AuthErrorCode = "unauthenticated" | "config_missing";

export async function requireUserId(): Promise<
  | { ok: true; userId: string }
  | { ok: false; code: AuthErrorCode }
>;
```

- Reads `CLERK_SECRET_KEY` lazily (NOT at import time). Missing →
  return `{ ok: false, code: "config_missing" }`. Never throws.
- When configured, calls Clerk's `auth()` helper from
  `@clerk/nextjs/server`. If `userId` is null → `{ ok: false, code:
  "unauthenticated" }`. If non-null → `{ ok: true, userId }`.
- **No DB write here.** The webhook is the only path that creates
  `users` rows. If a request arrives with a Clerk userId that the DB
  has not yet seen (webhook lag — race between sign-up and first AOI
  call), the route falls back to a JIT INSERT in the repository layer
  (see "JIT user provisioning" below).

Also export a `_setTestAuth(fn | null)` injection point analogous to
Stage 2's `_setTestFirmsFetch` so unit tests can stub auth without
running Clerk's runtime. Production passes `undefined`.

### Middleware (`app/middleware.ts` — new file)

`app/middleware.ts` does not exist today. Create it using
`clerkMiddleware` from `@clerk/nextjs/server`:

- Protect: `/api/aoi/:path*` (all CRUD + rules + briefs).
- Do NOT protect: `/api/aoi/poll` (cron — uses `CRON_SECRET`),
  `/api/webhooks/clerk` (Svix-signed by Clerk), `/api/health` (already
  scaffolded in Stage 0 if present), the marketing root `/`,
  `/sign-in/*`, `/sign-up/*`.
- Build-without-blocking: if `CLERK_SECRET_KEY` is unset, the
  middleware exports a no-op pass-through (returns `NextResponse.next()`
  unconditionally). Authenticated routes will then short-circuit at the
  `requireUserId()` call inside the handler with `config_missing`.
  This prevents middleware from crashing the entire app at boot when
  Clerk env vars are absent.

### Sign-in / sign-up pages

Two route directories using Clerk's pre-built components:
`app/sign-in/[[...rest]]/page.tsx` and `app/sign-up/[[...rest]]/page.tsx`.
Each is a thin RSC wrapper around `<SignIn />` and `<SignUp />` from
`@clerk/nextjs`. No custom styling beyond Clerk's defaults — Stage 6
owns marketing polish.

The `[[...rest]]` segment (Clerk's catch-all convention) is required so
Clerk's internal sub-routes (verification, OAuth callback) resolve
under the same path.

### Provider wrap (`app/layout.tsx`)

Wrap `<body>` contents with `<ClerkProvider>`. When `CLERK_SECRET_KEY`
is unset, render children without the provider — ClerkProvider throws
at render time if its publishable key is missing, so guard with a
`process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` truthiness check. Add a
`<ClerkConfigBanner>` component (server component, ~10 LOC) that
renders a yellow banner reading "Auth not configured — running in
read-only public mode" when the env var is missing. Same
build-without-blocking discipline as the AI Gateway / Resend banners.

### Clerk webhook (`app/api/webhooks/clerk/route.ts`)

`POST` handler that receives Clerk webhook events. Per Clerk docs, the
webhook is signed with Svix:

1. Read raw body (NOT `req.json()` — Svix verifies over the raw bytes).
2. Verify signature using the `svix` package and
   `CLERK_WEBHOOK_SIGNING_SECRET` (a separate Clerk-issued secret —
   provisioned via the Clerk dashboard's Webhook page; **append a
   blocker** for Vanyo to set this on Vercel + add it to Clerk's
   webhook endpoint URL `https://<vercel-url>/api/webhooks/clerk`).
3. Switch on `evt.type`:
   - `user.created` / `user.updated` → UPSERT a `users` row keyed on
     `id = evt.data.id`, `email = primary email from
     evt.data.email_addresses`, `display_name = evt.data.first_name +
     last_name` (null if both absent).
   - `user.deleted` → soft-delete: SET `users.deleted_at = now()`.
     CASCADE on `aois.user_id` will remove the user's AOIs; that is
     the spec's "delete = full removal" stance for v1 (no separate
     export-and-delete flow until Stage 6).
4. Return `200 { ok: true }` on success; `401` on signature failure;
   `400` on unknown event type (logged but not retried).

If `CLERK_WEBHOOK_SIGNING_SECRET` is unset, return `503
config_missing` and log a warning. Do NOT process unsigned payloads.

### JIT user provisioning (repository layer)

Webhooks can lag (Clerk may take 1-2s to fire `user.created` after
sign-up). A user could navigate to `/api/aoi` immediately after
sign-up and find no row in `users`. Two options:

- **A — JIT INSERT in `withDb`:** before invoking the route handler,
  if `userId` is set but no `users` row exists, INSERT a stub row
  using the Clerk `User` object fetched via `clerkClient.users.getUser(userId)`.
- **B — Block the request with 425 "user not yet provisioned":** UI
  retries.

**Pick A.** It is the user-friendliest path and the webhook still acts
as the canonical sync. The JIT INSERT is a single `INSERT ... ON
CONFLICT (id) DO NOTHING` — idempotent against the webhook arriving
mid-request. Add a TODO comment noting that if Clerk webhook delivery
becomes unreliable, the JIT path is the safety net.

### Update `lib/api/handlers.ts`

`withDb` becomes:

```ts
export async function withDb<T>(
  handler: (ctx: WithDbContext) => Promise<NextResponse<T> | NextResponse<ApiErrorBody>>,
): Promise<NextResponse<T> | NextResponse<ApiErrorBody>> {
  const db = tryGetDb();
  if (!db) return dbUnavailableResponse();
  const auth = await requireUserId();
  if (!auth.ok) {
    if (auth.code === "config_missing") return apiError("service_unavailable", "Auth not configured");
    return apiError("unauthenticated", "Sign in required", undefined, 401);
  }
  await ensureUserExists(db, auth.userId); // JIT provisioning
  try {
    return await handler({ db, userId: auth.userId });
  } catch (err) {
    return mapDomainError(err);
  }
}
```

**Delete `lib/api/current-user.ts`.** Its single export is gone;
nothing else references it after this PR. Verify with `grep -r
currentUserId`.

### Poll route changes (`app/api/aoi/poll/route.ts`)

Today's poll already enumerates buckets globally — `getActiveBuckets`
returns every active bucket regardless of user. **Confirm this** by
re-reading `lib/firms/buckets.ts` (the function should not filter by
`user_id`); if it does, widen it. The matcher already iterates AOIs
per bucket without a user filter. Stage 5's only change here is:

- The cron remains unauthenticated (uses `CRON_SECRET`).
- After Stage 4's dispatcher hook, `loadBrief` in `lib/notify/dispatch.ts`
  reads `users.email` for the AOI owner — this now resolves to the
  real authenticated user's email rather than the stub. **No code
  change in `dispatch.ts`** — the SQL JOIN already keys on
  `aois.user_id → users.id`. Confirm by reading `loadBrief`.

If `getActiveBuckets` or `loadBrief` make any STUB_USER_ID assumption
that grep missed, widen them.

### `GET /api/me` endpoint

Add `app/api/me/route.ts`:

```ts
export async function GET() {
  return withDb(async ({ db, userId }) => {
    const user = await getUserById(db, userId);
    return NextResponse.json({
      id: user.id,
      email: user.email,
      hasByoKey: user.geminiApiKeyEnc != null,
    });
  });
}
```

Per spec §API surface — minimal shape, no display-name in v1 response
(deferred until Stage 6 settings UI).

### Inject points for tests

- `lib/auth/context.ts` exports `_setTestAuth(fn | null)`. Production
  leaves it null and falls through to Clerk's `auth()`.
- The Clerk webhook handler accepts a `deps?: { verify?: typeof
  svixVerify }` parameter so tests stub Svix without a real signing
  secret.

### Tests

Three layers, mirroring Stages 3 and 4:

1. **PGlite unit tests** (no Docker, fast):
   - `tests/auth/context.test.ts` — every branch of `requireUserId`:
     `CLERK_SECRET_KEY` unset → `config_missing`; auth() returns null
     → `unauthenticated`; auth() returns userId → `ok`. Use
     `_setTestAuth` to stub Clerk's `auth()`.
   - `tests/auth/withDb.test.ts` — every branch of the new `withDb`:
     unauthenticated → 401; config_missing → 503; authed user with no
     row → JIT INSERT happens; authed user with row → handler runs.
   - `tests/auth/per-user-isolation.test.ts` — seed two users (`u-a`,
     `u-b`) each with one AOI; assert `listAois` for `u-a` returns
     only `u-a`'s AOI; assert `getAoiById(u-a, u_b's aoi_id)` returns
     null; assert `archiveAoi(u-a, u_b's aoi_id)` throws
     `AoiNotFoundError` (not a generic 500 — proves the existence
     check uses `user_id`).
   - `tests/auth/webhook.test.ts` — every webhook branch:
     `user.created` → row inserted; `user.updated` → row updated;
     `user.deleted` → row soft-deleted with cascade verified on a
     seeded AOI; bad signature → 401; missing signing secret → 503;
     unknown event type → 400 with logged warning.
   - `tests/auth/me.test.ts` — `/api/me` returns the right shape;
     unauthenticated → 401.
   - **Refactor existing repository tests** (`tests/aoi-repository.test.ts`)
     to seed real Clerk-style user ids (e.g. `user_2abc...`) instead
     of `STUB_USER_ID`. The literal `STUB_USER_ID` should not appear
     in any `tests/**` file after this PR.

2. **`@testcontainers/postgresql` integration test:**
   - `tests/auth/full-flow.integration.test.ts` — full authed flow:
     stub auth as `user_alice`, POST AOI, run the cron poll
     (unauthenticated, with stub FIRMS + stub AI Gateway + stub Resend
     deps), assert the resulting `notifications_log` row's `target` is
     Alice's webhook-synced email. Run again as `user_bob` with a
     different polygon; assert Bob's poll output does not affect
     Alice's `aoi_briefs` rows.

3. **Live Clerk test — gated.** `tests/auth/clerk.live.test.ts` runs
   only when `CLERK_LIVE=1` AND `CLERK_SECRET_KEY` are both set. It is
   `it.skip`'d otherwise. CI does not set these. The test asserts:
   `clerkClient.users.getUser(testUserId)` returns a non-null user
   with a non-empty email. Vanyo runs locally to spot-check that the
   real Clerk SDK loads. The test does NOT exercise sign-in flows
   (those need a browser; Stage 6 owns Playwright).

Default `pnpm test` runs everything except the live test.

## Out of scope for Stage 5 (do NOT build)

- **Social SSO providers (Google, GitHub, etc.).** Clerk's default
  email/password + email-OTP is enough for v1. Spec US-1 mentions
  "Google / email OTP" but defers; Vanyo's call later. Configuration is
  done in Clerk dashboard, not in code; no code change needed when
  Vanyo turns it on.
- **MFA enforcement** — Clerk supports it; v1 does not require it.
- **Team / org features** — explicit non-goal per spec scope boundaries
  ("Not multi-org / shared workspaces").
- **BYO Gemini key (US-7) decryption / settings UI** — Stage 6. The
  `gemini_api_key_enc` column stays untouched.
- **User deletion / data export beyond cascade** — Stage 6 owns the
  signed-token unsubscribe + GDPR-style export per US-5 / US-6.
- **Admin panel / impersonation** — observability for v1.1+.
- **Sign in with Vercel** as an alternate provider — Vanyo's call later
  (open question below).
- **`POST /api/notifications/webhook/{token}`** for snooze/pause links
  — Stage 6.
- **Marketing landing page polish, sign-in CTAs, onboarding wizard,
  AOI-create UI** — Stage 6. Stage 5 ships the auth plumbing only;
  the only UI is Clerk's pre-built components.
- **Rate-limit per-user (60 req/min/user per spec §API surface)** —
  Stage 6. v1 launch traffic is light; rate-limiting without users is
  premature.
- **Bearer-token API access for MCP (US-8)** — Stage 8b owns MCP. v1
  Clerk session cookie is sufficient.

Do NOT touch `db/schema/postgis.ts`, the `industrial_mask_static`
seed, any FIRMS code paths, `lib/ai/`, or `lib/notify/`. Stage 5 is
strictly a cross-cutting auth layer.

## Build-without-blocking discipline (per ADR 0006)

`CLERK_SECRET_KEY` and `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` are set on
Vercel per resolved blocker (2026-05-06). The build still must be safe
if either is **not** reachable (local dev without `.env.local`,
ephemeral CI, key revocation):

- `lib/auth/context.ts` returns `{ ok: false, code: "config_missing" }`
  when `CLERK_SECRET_KEY` is unset — never throws at import time.
- `app/middleware.ts` is a pass-through when `CLERK_SECRET_KEY` is
  unset.
- `app/layout.tsx` skips `<ClerkProvider>` and renders the
  config-banner when `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` is unset.
- Authenticated routes return 503 `service_unavailable` with body
  `{"code":"service_unavailable","message":"Auth not configured"}`
  rather than crashing.
- `app/api/webhooks/clerk/route.ts` returns 503 when
  `CLERK_WEBHOOK_SIGNING_SECRET` is unset; never processes unsigned
  payloads.
- The cron poll continues to run; `users.email` lookups in
  `loadBrief` short-circuit to "no email channel" for users without an
  email (the dispatcher already handles missing email by recording
  `skipped, channel_not_implemented` per Stage 4 brief §Channel
  resolution).
- All non-live tests must pass without any Clerk env var set.

## Workflow (per ADR 0006 + ADR 0007)

1. `git checkout -b stage-5-clerk-auth` off latest `master`
2. Build in commits that group naturally (target: 6–9 commits, ≤500 LOC each):
   - `db/`: `0004_stage5.sql` + `.test.sql` + remove `STUB_USER_ID` from `db/schema/index.ts`
   - `lib/auth/context.ts` + unit tests
   - `app/middleware.ts` + `app/layout.tsx` `<ClerkProvider>` wrap + config banner
   - `app/sign-in/[[...rest]]/page.tsx` + `app/sign-up/[[...rest]]/page.tsx`
   - `app/api/webhooks/clerk/route.ts` + Svix verification + tests
   - `lib/api/handlers.ts` rewrite (`withDb` becomes async-auth) + delete `lib/api/current-user.ts` + `app/api/me/route.ts`
   - test-suite refactor: every `STUB_USER_ID` reference in `tests/` becomes a real Clerk-style id; new `tests/auth/*` files
   - integration test: `tests/auth/full-flow.integration.test.ts`
   - live-test scaffold (skipped unless `CLERK_LIVE=1`)
3. `pnpm install` (Clerk + Svix are new deps), `pnpm typecheck`, `pnpm lint`, `pnpm test` (Docker running locally for integration coverage), `pnpm build` — all green
4. `git push origin stage-5-clerk-auth`
5. Report back to PM_CLAUDE with: branch SHA, all check statuses, draft PR description in markdown
6. PM_CLAUDE opens the PR via `gh pr create --draft --base master --label stage-pr:5`
7. **Auto-merge applies.** Per ADR 0007, once CI is green and the
   reviewer subagent LGTMs the diff, the PR auto-merges. The
   `stage-pr:5` label is informational. Do not instruct any reviewer
   to "wait for Vanyo".

## Output

1. Branch on origin: `stage-5-clerk-auth`
2. Draft PR description in your reply (sections: Summary, What changed,
   How to test, Build-without-blocking notes, Things to challenge in
   review, Linked: brief 19 / ADR 0006 / ADR 0007 / spec §User stories
   US-1 + US-7 / pivot-arch §3.1)
3. `pm/research-log/2026-05-06-stage5-clerk-auth.md` — what shipped,
   deferrals, deviations from brief, open questions for PM. Include a
   note on whether you ran the live Clerk test locally and what
   `clerkClient.users.getUser` returned.

## Time budget

~3 hours. If you hit a 20-minute block on any single error, stop and
report. The two known sharp edges:
- **Svix webhook signature verification.** Clerk wraps Svix; the raw
  body MUST be read as bytes before any `req.json()` consumes the
  stream. Next.js 16 App Router gives you `await req.text()` as the
  raw-body escape hatch — use that, then parse. Document the choice in
  the research log.
- **`<ClerkProvider>` + RSC layout interaction.** Clerk's provider is a
  client component; wrapping the root layout's children works in
  Clerk's quickstart, but Next.js 16's Cache Components mode
  (`use cache`) on the marketing page can interact badly with
  client-only providers. If the marketing page in Stage 5 is still the
  Stage-0 placeholder (no `use cache`), there is no conflict; if Stage
  6 later moves to `use cache`, that's Stage 6's problem to revisit.
  Note in the research log if you observe any RSC/cache surprises.

## Branch + label

- Branch: `stage-5-clerk-auth`
- PR base: `master`
- Label: `stage-pr:5` (PM_CLAUDE applies; informational under ADR 0007)
