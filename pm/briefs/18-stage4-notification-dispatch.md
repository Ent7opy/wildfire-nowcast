# Brief 18 — Stage 4: notification dispatch (Resend)

## Why this exists

Stage 4 of the A' pivot. Stage 3 turned matched events into validated
`aoi_briefs` rows; nothing yet leaves the database. Stage 4 closes the
last leg of the deliverable: every newly-persisted brief becomes an
**email** to the AOI owner via Resend, with idempotent send-tracking and
the same build-without-blocking discipline used for Stages 2 and 3.

This is the stage where US-1 ("watch confirmed in ≤ 5 min") and US-3
("brief in my inbox when fire touches my place") actually fire end-to-end.

**Read in order:**
1. `pm/PM_CLAUDE.md`
2. `pm/decisions/0005-problem-chosen-a-prime.md`
3. `pm/decisions/0006-stage-pr-workflow.md` — workflow you follow (branch off `master`, draft PR, PM_CLAUDE opens it)
4. `pm/decisions/0007-*` (auto-merge gate update — **stage PRs no longer require Vanyo's manual review**; see "Workflow" below)
5. `pm/blockers.md` — Stage 4 entry for `RESEND_API_KEY` is already resolved (2026-05-06). Sender-domain verification is deferred — use `onboarding@resend.dev` per the resolution note.
6. `docs/SPEC-A-prime-v1.md` — §User stories US-1, US-3, US-5; §Flows steps 4–6 ("Brief persisted → dispatcher sends to each active channel"); §Data model `notifications` row shape
7. `docs/pivot-architecture.md` — §3.7 `notifications_log` (binding SQL shape) + §"Stage 4 — Notifications" (binding file layout: `lib/notify/resend.ts`, `lib/notify/dispatch.ts`)
8. `pm/briefs/17-stage3-brief-generation.md` — the immediately preceding brief; Stage 4 plugs into the same poll route after `generateBriefForEvent` returns `{ status: "generated" }`
9. `lib/ai/generate.ts`, `app/api/aoi/poll/route.ts`, `db/schema/index.ts` — current code surface; `aoi_briefs.rendered_markdown` is the email body source, `aoi_rules.notify_channels` is the recipient source

## Goal

Land — on a `stage-4-notification-dispatch` branch off `master` — the
`notifications_log` schema table, a Resend client + dispatcher that turns
each newly-persisted `aoi_briefs` row into an email send (per active
`email`-typed channel on the AOI's rules), idempotent persistence so a
re-poll never double-sends, the hook into `/api/aoi/poll` after
`generateBriefForEvent`, and the unit + integration tests covering all of
it. Webhook channels are persisted as `skipped` with reason
`channel_not_implemented` (out of scope this stage). PR draft markdown
ready; PM_CLAUDE opens the PR.

## Scope (strict)

### Schema additions (new migration `db/migrations/0003_stage4.sql` + Drizzle schema)

Per `docs/pivot-architecture.md` §3.7, add:

- **`notifications_log`** — one row per send attempt. Columns:
  - `id uuid pk default gen_random_uuid()`
  - `aoi_id uuid not null fk → aois ON DELETE CASCADE`
  - `brief_id uuid not null fk → aoi_briefs ON DELETE CASCADE` (Stage 4
    treats brief as the source-of-truth, so `RESTRICT` would also be
    defensible — pick `CASCADE` for parity with §3.7)
  - `channel text not null` — `'email'` for v1; `'webhook'` reserved
  - `target text not null` — the actual recipient (e.g. the email address). Stored plaintext; this is operator-readable, not user-readable.
  - `target_hash text not null` — `sha256(target)`, used as the rate-limit key per spec §3.7
  - `status text not null` — one of `'sent' | 'failed' | 'skipped' | 'config_missing'`
  - `provider_message_id text` — Resend's message id on success; null otherwise
  - `error text` — failure reason (truncated to 500 chars); null on success
  - `skip_reason text` — set when `status = 'skipped'`: `'channel_not_implemented' | 'paused' | 'quiet_hours' | 'duplicate'`
  - `sent_at timestamptz not null default now()`
- **Unique index for idempotency:**
  `unique (brief_id, channel, target_hash) WHERE status IN ('sent', 'skipped')`
  — re-running the dispatcher for the same brief/channel/target after a
  successful send or a deliberate skip is a no-op. Failed sends are NOT
  in the unique key, so retry on next poll is allowed.
- **Index for rate-limit lookups (spec §3.7):**
  `(aoi_id, target_hash, sent_at desc)`.

Hand-author SQL (same pattern as Stages 1–3 — `0003_stage4.sql` for Neon,
`0003_stage4.test.sql` for PGlite). No PostGIS in this migration. Update
`db/schema/index.ts` with the `notificationsLog` Drizzle table and remove
the "DEFERRED to later stages" comment for `notifications_log`.

Also add `last_notified_at timestamptz null` to `aoi_briefs` so the UI
(Stage 6) and any "stuck briefs" admin query can short-circuit. The
dispatcher SETs this column when the FIRST email send for the brief
returns `status = 'sent'`.

### Resend client (`lib/notify/resend.ts`)

Single function: `sendEmail({ to, from, subject, markdown, html?, replyTo? })`
returning a typed result. Mirror the `lib/ai/gateway.ts` shape:

```ts
type SendResult =
  | { ok: true; providerMessageId: string; latencyMs: number }
  | { ok: false; code: SendErrCode; message: string; latencyMs: number };

type SendErrCode = "config_missing" | "rate_limited" | "provider_error" | "validation_failed";
```

- Use the `resend` npm package (`new Resend(process.env.RESEND_API_KEY)`) or
  a direct `fetch` to `https://api.resend.com/emails` — pick whichever is
  smaller. Read the env var **lazily inside the function**, never at
  import time.
- If `RESEND_API_KEY` is missing → return `{ ok: false, code: "config_missing", message: "RESEND_API_KEY is not configured" }` without throwing. Do not initialise the SDK at module scope.
- Convert the brief's `rendered_markdown` to HTML using a lightweight
  Markdown→HTML converter (`marked` is already a transitive dep via
  Next; if not, add `marked` — small, well-typed, no plugin surface).
  Send both `text` (the raw markdown) and `html` to Resend so plaintext
  clients render cleanly.
- **Test mode:** when `RESEND_TEST_MODE === "1"`, force the `from`
  address to Resend's no-domain test sender `onboarding@resend.dev` and
  set the subject suffix `[TEST]`. Per blocker resolution
  (`pm/blockers.md` 2026-05-06), real domain verification is deferred
  until after Stage 4 ships.
- Production `from`: configurable via `NOTIFY_FROM_ADDRESS`
  (default `onboarding@resend.dev` so first deploy works without
  domain). When `NOTIFY_FROM_ADDRESS` differs from the default, log a
  one-line info on first send per process indicating the configured
  sender — helpful for verifying the domain switch later.
- No retry-with-backoff inside the client — Stage 4 is single-attempt
  per poll. Failures are persisted with `status = 'failed'` and the
  unique-index design means the **next** poll sees the failed row and
  permits a fresh attempt.

### Dispatcher (`lib/notify/dispatch.ts`)

Single entry point: `dispatchBrief(db, briefId, deps?)`. Called once per
brief that Stage 3's orchestrator just persisted. Steps:

1. Load the brief + its AOI + `aoi_rules.notify_channels` + the user's
   account email. Single SQL query (see `lib/ai/generate.ts`'s
   `loadEventContext` pattern; same two-backend discipline).
2. **Channel resolution.** Iterate over `notify_channels`:
   - `{ type: "email", target }` → use `target`.
   - `{ type: "webhook", target }` → record a `status = 'skipped',
     skip_reason = 'channel_not_implemented'` row and continue.
   - **Fallback:** if `notify_channels` is empty, default to one
     synthetic email channel using the user's `users.email` (per spec
     §User journey step 4 "channels = account email"). Mark these
     synthesised sends in the dispatcher return shape so the integration
     test can assert the fallback fired.
3. **Per-channel idempotency check.** Before sending, query
   `notifications_log` for any existing row with
   `(brief_id, channel, target_hash) AND status IN ('sent', 'skipped')`.
   If found, record `status = 'skipped', skip_reason = 'duplicate'` in
   the return shape (do NOT insert a second row — the unique index
   would reject and the conflict-handling cost is wasted) and continue.
4. **Pause / quiet-hours gate.** Defer most of US-2's quiet-hours logic
   to Stage 6 (rules UI), but enforce the trivial cases here:
   - If `aoi_rules.paused_until > now()` → `skipped, paused`.
   - **Quiet hours: skip-only, no digest merging.** If `quiet_hours` is
     set and `now()` in the configured tz falls inside `[startHour, endHour)`,
     record `skipped, quiet_hours` and return — the spec's "morning
     digest at the top of the quiet window" (US-2 acceptance #3) is a
     Stage 6 concern. Add a `// TODO Stage 6:` comment at the gate.
5. **Build envelope.** Subject = the brief payload's `summary` field
   truncated to 90 chars (it is the spec's "first thing the reader
   sees" per §LLM brief format notes). Body = `rendered_markdown`. No
   snooze/pause/unsubscribe links yet — those need signed tokens (US-5)
   and ship in Stage 6 alongside the rules UI.
6. **Send via `sendEmail`.** On `config_missing`, persist `status =
   'config_missing'` and return without erroring (build-without-blocking).
7. **Persist row.** INSERT `notifications_log` with the result. On the
   first successful send for this brief, also UPDATE
   `aoi_briefs.last_notified_at = now()`. Both writes happen in a single
   transaction (mirror `persistBrief` in `lib/ai/generate.ts`).
8. Return a discriminated outcome shape:
   ```ts
   type DispatchOutcome = {
     briefId: string;
     attempts: Array<
       | { status: "sent"; channel: string; target: string; providerMessageId: string }
       | { status: "failed"; channel: string; target: string; error: string }
       | { status: "skipped"; channel: string; target: string; reason: string }
       | { status: "config_missing"; channel: string; target: string }
     >;
   };
   ```

The dispatcher is the **only** way Stage 4 sends emails. The poll route
calls it once per `generated` outcome from Stage 3.

### Hook into the existing poll (`app/api/aoi/poll/route.ts`)

After the `generateBriefForEvent` loop in `runOneBucket`, add a second
loop over briefs that returned `status === "generated"`. For each:

1. Call `dispatchBrief(db, outcome.briefId)`.
2. Roll counts up into the per-bucket outcome.

Extend `BucketRunOutcome`:
- `notificationsSent: number`
- `notificationsFailed: number`
- `notificationsSkipped: number` (any non-`sent`, non-`failed`, non-`config_missing` outcome)
- `notificationConfigMissing: boolean` (true if any attempt returned `config_missing`)

Roll bucket totals into the parent `job_runs` row by adding
`notifications_sent integer not null default 0` to the
`0003_stage4.sql` migration — same pattern as
`briefs_generated` from Stage 3. Update `closeJobRun`'s `CloseArgs`
accordingly.

If `RESEND_API_KEY` is unset, the dispatcher returns `config_missing`
per attempt; the route logs a single warning per poll (not per bucket
or per brief) and continues. The poll's overall `status` stays `'ok'`
because brief generation succeeded — only delivery was skipped.

### Inject points for tests

Mirror Stage 3's `_setTestBriefGen`:

- Add `_setTestNotifyDispatch(fn | null)` exported from
  `app/api/aoi/poll/route.ts` so integration tests can stub the
  dispatcher.
- Add a `deps?: { send?: typeof sendEmail; now?: Date }` parameter to
  `dispatchBrief` so unit tests can stub the Resend call without an env
  var. Production passes `undefined`.

### Tests

Two layers, mirroring Stage 3's split:

1. **PGlite unit tests** (no Docker, fast):
   - `tests/notify/dispatch.test.ts` — every dispatcher branch:
     - happy path: one email channel → one `sent` row, `last_notified_at` set
     - empty `notify_channels` → fallback to user email, `sent`
     - duplicate brief on second invocation → `skipped, duplicate`, no second row
     - `aoi_rules.paused_until` in future → `skipped, paused`
     - quiet-hours window matches → `skipped, quiet_hours` (test the tz math against `America/Los_Angeles`)
     - webhook channel → `skipped, channel_not_implemented`
     - `RESEND_API_KEY` unset (stubbed `send` returning `config_missing`) → row written with `status = 'config_missing'`, NO `last_notified_at` update
     - `send` returns `{ ok: false, code: "provider_error" }` → row written with `status = 'failed'`, NO `last_notified_at` update, dispatcher does NOT throw
   - `tests/notify/resend.test.ts` — pure parser/envelope tests:
     - subject truncation at 90 chars
     - markdown→html renders one canonical brief snapshot (the Spring Creek fixture from Stage 3 — reuse it)
     - `RESEND_TEST_MODE=1` rewrites `from` and adds `[TEST]` suffix
     - missing `RESEND_API_KEY` → `{ ok: false, code: "config_missing" }` without throwing

2. **`@testcontainers/postgresql` integration test:**
   - `tests/notify/poll-to-notify.integration.test.ts` — full flow:
     seed AOI + email channel, stub FIRMS to return one matching
     detection, stub the AI Gateway to return a valid Spring Creek
     brief, stub `sendEmail` to record calls + return `ok: true`.
     Assert exactly one `notifications_log` row with `status = 'sent'`,
     `aoi_briefs.last_notified_at` populated, `job_runs.notifications_sent
     = 1`. Re-run the poll; assert no second send (idempotency).

3. **Live Resend test — gated.** A single integration test
   `tests/notify/resend.live.test.ts` that hits the real Resend API runs
   only when `RESEND_LIVE=1` AND `RESEND_API_KEY` are both set in the
   environment. It is `it.skip`'d otherwise. CI does not set these.
   Vanyo runs it locally to spot-check that an actual email lands. The
   test asserts: `ok: true`, `providerMessageId` is non-empty,
   `latencyMs < 10000`. It uses `RESEND_TEST_MODE=1` so the send goes
   through Resend's test sender — no domain verification required.

Default `pnpm test` runs everything except the live test.

## Out of scope for Stage 4 (do NOT build)

- **Webhook channel delivery** (Slack/Discord-compat payloads) — record
  as `skipped, channel_not_implemented`. Spec §3.7 reserves the column;
  v1.1 or Stage 6 owns the Slack/Discord work.
- **Snooze / pause / unsubscribe signed-token links** in email body —
  US-5; needs Clerk-backed user identity (Stage 5) and the rules UI
  (Stage 6). Brief renderer already declines to add these (per Stage 3
  brief).
- **Quiet-hours digest merging** (release held briefs at top of window) —
  US-2 acceptance #3; Stage 6. Stage 4 is "skip during quiet hours,
  full stop" with a `// TODO Stage 6:` marker.
- **Retry-with-backoff** — Stage 4 is single-attempt per poll; the next
  poll naturally retries failed rows because the unique index excludes
  `status = 'failed'`.
- **Per-recipient preferences / list-unsubscribe headers** — Stage 6.
- **Rate limit per `(aoi_id, target_hash)` over a 15-min window** —
  spec §3.7 mentions it, but there is no looping detection source until
  Stage 5/6 surfaces it; the idempotency unique index already prevents
  the most common double-send (same brief twice). The rate-limit window
  query lands when there's a pattern to throttle.
- **Inbound `POST /api/notifications/webhook/{token}`** route for
  snooze/pause links — Stage 6.
- **Auth (Clerk)** — Stage 5. Keep `STUB_USER_ID` everywhere; the
  fallback `users.email` is read for that single seeded user.
- **Sender-domain verification on Resend** — deferred per blocker
  resolution; `onboarding@resend.dev` is the v1 default.
- **Cost telemetry / Resend usage rollup** — observability for v1.1+.

Do NOT touch `db/schema/postgis.ts`, the `industrial_mask_static` seed,
any FIRMS code paths, or anything in `lib/ai/`. Stage 4 is a strictly
post-brief layer.

## Build-without-blocking discipline (per ADR 0006)

`RESEND_API_KEY` is set on Vercel per resolved blocker (2026-05-06). The
build still must be safe if the key is **not** reachable (local dev
without `.env.local`, ephemeral CI, key revocation):

- `lib/notify/resend.ts` returns `{ ok: false, code: "config_missing" }`
  when the env var is unset — never throws at import time.
- `dispatchBrief` persists the `config_missing` outcome and returns
  without throwing.
- `app/api/aoi/poll/route.ts` continues to run (matching, briefs) and
  reports per-bucket `notificationConfigMissing: true` if the key is
  missing; the poll's overall status is still `'ok'`.
- All non-live tests must pass without `RESEND_API_KEY` set.

## Open question — recipient resolution

The current schema has **no per-AOI "recipients" array** — the recipient
source is `aoi_rules.notify_channels` (typed `Array<{ type: "email" |
"webhook"; target: string }>`). When that array is empty, the spec
(§User journey step 4) says "channels = account email", so we fall back
to `users.email`.

This is unambiguous from the existing schema + spec, so **no Stage 4
blocker is added**. If during build the dev agent finds a case the
schema can't express (e.g. multiple email recipients per AOI — Vanyo
mentioned this once but it is not in the spec), record it in the
research log and surface it for v1.1 — do NOT widen scope mid-stage.

## Workflow (per ADR 0006 + ADR 0007)

1. `git checkout -b stage-4-notification-dispatch` off latest `master`
2. Build in commits that group naturally (target: 5–8 commits, ≤500 LOC each):
   - `db/`: `notificationsLog` schema + `0003_stage4.sql` + `0003_stage4.test.sql` + `last_notified_at` + `notifications_sent`
   - `lib/notify/resend.ts` + envelope tests
   - `lib/notify/dispatch.ts` + unit tests (every branch)
   - hook into `app/api/aoi/poll/route.ts` + integration test
   - live-test scaffold (skipped unless `RESEND_LIVE=1`)
3. `pnpm install`, `pnpm typecheck`, `pnpm lint`, `pnpm test` (Docker
   running locally for integration coverage), `pnpm build` — all green
4. `git push origin stage-4-notification-dispatch`
5. Report back to PM_CLAUDE with: branch SHA, all check statuses, draft PR
   description in markdown
6. PM_CLAUDE opens the PR via `gh pr create --draft --base master --label stage-pr:4`
7. **Auto-merge applies.** Per ADR 0007 (which superseded ADR 0006's
   "Vanyo always reviews stage PRs" clause), once CI is green and the
   reviewer subagent LGTMs the diff, the PR auto-merges. The
   `stage-pr:4` label is informational — it no longer disables
   auto-merge. Do not instruct any reviewer to "wait for Vanyo".

## Output

1. Branch on origin: `stage-4-notification-dispatch`
2. Draft PR description in your reply (sections: Summary, What changed,
   How to test, Build-without-blocking notes, Things to challenge in
   review, Linked: brief 18 / ADR 0006 / ADR 0007 / spec §3.7 /
   pivot-arch §"Stage 4 — Notifications")
3. `pm/research-log/2026-05-06-stage4-notification-dispatch.md` — what
   shipped, deferrals, deviations from brief, open questions for PM.
   Include a note on whether you ran the live Resend test locally and
   what the latency / message-id looked like.

## Time budget

~3 hours. If you hit a 20-minute block on any single error, stop and
report. The two known sharp edges:
- **Resend SDK vs raw fetch.** The `resend` npm package is small but
  pulls a whole HTTP layer; raw `fetch` to
  `https://api.resend.com/emails` is ~30 LOC and zero deps. Prefer
  raw fetch unless the SDK gives you a typed response shape worth the
  weight. Document the choice in the research log.
- **Cross-driver SQL for the new unique partial index.** PGlite supports
  partial unique indexes but the syntax can drift from Neon-PostgreSQL
  on edge cases. Use the same hand-authored split (`0003_stage4.sql`
  for Neon, `0003_stage4.test.sql` for PGlite) and keep the DDL
  literal in both files; do not try to share via Drizzle's index DSL —
  the existing Stage 1/2/3 migrations all hand-author this.

## Branch + label

- Branch: `stage-4-notification-dispatch`
- PR base: `master`
- Label: `stage-pr:4` (PM_CLAUDE applies; informational under ADR 0007)
