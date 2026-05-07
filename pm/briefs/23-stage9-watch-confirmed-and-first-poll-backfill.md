# Brief 23 — Stage 9: watch-confirmed email + first-AOI backfill

## Why this exists

Stage 8 (authority perimeter + freshness) merged on 2026-05-07. The same day,
the scout's stop-line brainstorm
(`pm/research-log/2026-05-07-v1-stop-line.md`) named the next gap with
unusual precision: *"the missing piece on the cold-start path is not the map,
the snooze tokens, the LICENSE, or the perimeter — it is the watch-confirmed
email plus first-AOI backfill, neither of which is in `master`, both of
which gate acceptance #2's measurability, and together they are smaller than
any single shipped stage."*

This is SPEC Flow 1 steps 5 and 6 verbatim
(`docs/SPEC-A-prime-v1.md` lines 99–101):

> 5. Confirmation email "Now watching {AOI name}. First poll at {UTC time}."
> 6. First cron tick within 15 min runs a backfill poll over last 24 h so a
>    user arriving mid-fire gets immediate context.

Neither has code in `master`. The verification:

- **Step 5**: `app/api/aoi/route.ts` (the AOI POST handler) returns 201 with
  the AOI shape — there is no email-dispatch call after persistence. The
  Stage 4 dispatcher (`lib/notify/dispatch.ts`) only fires from the cron
  brief path, never from AOI creation. No `lib/notify/watch-confirmed.ts`
  exists; no `notification_kind` discriminator exists in
  `notifications_log`.
- **Step 6**: `app/api/aoi/poll/route.ts` enumerates `getActiveBuckets(db)`
  and polls each. There is no AOI-creation hook into the poll route. The
  cron route only accepts `{ bucket?, source? }` in its body schema — no
  AOI-scoped invocation path.

The cost: launch acceptance #2 ("cold start to first watch ≤ 5 min, measured
on a clean browser") is **structurally unmeasurable** today. The terminal
event of the SLA — an email arriving — does not fire. The launch-readiness
table flags this:
`pm/launch-readiness.md:22` — status `partial`, gating "Stage 7 + a small
follow-up chore for the confirmation email (out of Stage 7 scope per brief
21)." Stage 9 *is* that chore, plus the second half of Flow 1 (backfill).

The product reason both belong in the same stage: they share the trigger
(AOI created, first time we have a polygon to watch). Splitting them into
two PRs duplicates the AOI-POST surgery and the test scaffolding for "what
happens immediately after AOI creation."

**Read in order:**

1. `pm/PM_CLAUDE.md` — doctrine. No fabricated data, no scope creep.
2. `pm/decisions/0006-stage-pr-workflow.md` — branch / draft PR / label.
3. `pm/decisions/0007-*` — auto-merge gate (CI green + reviewer LGTM
   merges; do NOT instruct any reviewer to "wait for Vanyo").
4. `pm/blockers.md` — pre-existing blockers (ICNF, Vercel-Hobby, Clerk
   webhook). Stage 9 should add no new blockers if it sticks to existing
   infra.
5. `pm/research-log/2026-05-07-v1-stop-line.md` — full reasoning for why
   this is the next stage. Sections 4 ("backfill on first AOI") and 6
   ("if only one thing gets done before launch, it is those two").
6. `docs/SPEC-A-prime-v1.md` — §Core flows / Flow 1 (steps 5 + 6 are the
   spec contract); §Acceptance for v1 launch item #2.
7. `pm/briefs/22-stage8-authority-perimeter-and-freshness.md` — tone,
   structure, build-without-blocking discipline, two-backend reminder.
8. `pm/briefs/18-stage4-notification-dispatch.md` — the dispatcher pattern
   Stage 9's email reuses (single-attempt, idempotent, `config_missing`
   path).
9. `app/api/aoi/route.ts` — AOI POST handler. The watch-confirmed dispatch
   is wired here, after `createAoi` returns and before the 201 is sent. The
   backfill kickoff is wired in the same place.
10. `app/api/aoi/poll/route.ts` — cron route. Stage 9 either reuses this
    route via an internal sub-call, or lifts its per-bucket inner loop into
    a shared helper. See §"Backfill" below for the design choice.
11. `lib/notify/dispatch.ts` — Stage 4's brief-dispatch. The watch-confirmed
    email is a **second kind of email** (not a brief); it does NOT call
    `dispatchBrief`. It uses the lower-level `sendEmail` from
    `lib/notify/resend.ts` directly, plus a row in `notifications_log` so
    the launch-week observability lights up uniformly.
12. `lib/notify/resend.ts` — `sendEmail({ to, subject, markdown })`. The
    `config_missing` (RESEND_API_KEY unset) and `rate_limited` paths
    already return typed results; Stage 9 reuses both.
13. `lib/firms/client.ts` — `fetchAreaCsv({ source, bbox, dayRange })`. The
    backfill calls this with `dayRange: 1` over the new AOI's bucket bbox.
14. `lib/firms/buckets.ts` — `bucketToBbox(bucket)`; the AOI's
    `regionBucket` is already populated by `createAoi`.
15. `lib/firms/matcher.ts` — `matchDetectionsToAois`. The backfill runs
    this against just the new AOI (single-AOI scope) — see §"Backfill".
16. `lib/db/aoi-repository.ts` — `createAoi` returns the new AOI row
    including `regionBucket`. No schema motion needed for the AOI itself.
17. `db/schema/index.ts` — `notifications_log` and `job_runs`. Stage 9
    needs *one* additive column on `notifications_log` (`kind`) so the
    watch-confirmed row can be distinguished from brief rows in launch-week
    queries. No new tables.

## Goal

Land — on a `stage-9-watch-confirmed-and-first-poll-backfill` branch off
`master` — two product-visible improvements that close SPEC Flow 1
end-to-end:

1. **Watch-confirmed email.** When `POST /api/aoi` succeeds, dispatch a
   one-shot email to the signed-in user's email address: "Now watching
   {AOI name}. First poll at {UTC time}." Body lists AOI name, bucket
   (region summary), area in hectares, expected first-poll time, link to
   the AOI page, link to support.
2. **First-AOI backfill.** When `POST /api/aoi` succeeds, immediately run
   a backfill poll scoped to *this AOI only*: fetch the last 24 h of FIRMS
   detections for this AOI's bucket, run the matcher, generate any briefs
   the gate passes, dispatch them through the existing Stage 4 path. The
   user lands on the AOI detail page with brief history already populated
   if there is an active fire near their polygon.

Together: a user arriving mid-fire sees their first brief within seconds
of creating their AOI, not on the next 15-minute cron tick.

Test coverage at the levels Stages 3–8 established. Build-without-blocking
holds: `RESEND_API_KEY` unset → email skipped, AOI still creates;
`FIRMS_MAP_KEY` unset → backfill skipped, AOI still creates. PR draft
markdown ready; PM_CLAUDE opens the PR.

## Scope (strict)

### Schema motion (additive only)

`db/migrations/0007_stage9.sql`:

```sql
ALTER TABLE "notifications_log"
  ADD COLUMN "kind" text NOT NULL DEFAULT 'brief';

CREATE INDEX "notifications_log_kind_created_at_idx"
  ON "notifications_log" ("kind", "created_at" DESC);
```

Drizzle-side: extend `notificationsLog` in `db/schema/index.ts` with
`kind: text("kind").notNull().default("brief")`. Existing rows are
backfilled to `'brief'` by the `DEFAULT`; the watch-confirmed dispatcher
inserts with `'watch_confirmed'`.

**Why a `kind` column instead of a separate table:** the existing
`notifications_log` row shape (channel + target + status + skip_reason +
provider_message_id + idempotency hash) is exactly what watch-confirmed
needs. A second table would duplicate the columns and force the
launch-readiness "did this user receive any email" query to UNION across
two sources. The discriminator is a one-column add.

**Why `NOT NULL DEFAULT 'brief'` rather than nullable:** every existing
row is a brief dispatch. Defaulting to `'brief'` is honest about that
history; a NULL would force consumer code to handle a third state that
doesn't exist.

The existing `brief_id` column on `notifications_log` becomes nullable
*by interpretation* — a watch-confirmed row has no brief_id. **Confirm in
schema:** if `brief_id` is currently `NOT NULL`, this stage drops the
NOT NULL constraint. If it's already nullable (likely — the dispatcher
inserts pre-brief skip rows), no change.

### Watch-confirmed email module (`lib/notify/watch-confirmed.ts`)

New module. Single export:

```ts
export type WatchConfirmedOutcome =
  | { status: "sent"; providerMessageId: string }
  | { status: "skipped"; reason: "no_recipient" | "config_missing" | "duplicate" }
  | { status: "failed"; error: string };

export async function dispatchWatchConfirmed(
  db: AppDb,
  args: {
    aoiId: string;
    userId: string;
    aoiName: string;
    regionBucket: string;
    areaHa: number;
    firstPollAt: Date;     // see §"First-poll time" below
    aoiUrl: string;        // absolute URL to /dashboard/aoi/{id}
    now?: Date;
    sendImpl?: typeof sendEmail;
  },
): Promise<WatchConfirmedOutcome>;
```

Behaviour:

1. Resolve the user's email by looking up `users.email` via `userId`.
   If the row is missing (impossible post-Clerk-JIT; defensive) or the
   email is the JIT placeholder pattern (matches the `isPendingPlaceholder`
   check `lib/notify/dispatch.ts` already exposes — refactor it to a
   shared helper if needed), return `{ status: "skipped", reason: "no_recipient" }`.
2. Compute idempotency hash: `sha256("watch_confirmed:" + aoiId)`. Insert
   the `notifications_log` row with `kind = 'watch_confirmed'`,
   `brief_id = NULL`, `target_hash = <hash>`. The unique idempotency
   index on `(brief_id, channel, target_hash, status)` excludes
   `status='failed'` already; for the watch-confirmed flow we want
   "exactly once per AOI ever," so check for any prior row with this
   `target_hash` and `kind='watch_confirmed'` *before* sending. If one
   exists with `status='sent'`, return `{ status: "skipped", reason: "duplicate" }`.
3. Build the email body via a small renderer in
   `lib/notify/watch-confirmed-template.ts`:

   ```
   Subject: Now watching {AOI name}

   Hi,

   Your area "{AOI name}" ({areaHa} ha, region {regionBucketHumanized})
   is now being watched.

   We poll NASA FIRMS every 15 minutes for new fire detections.
   Your first poll is scheduled for {firstPollAt as local time + UTC}.
   If a detection inside or near your polygon meets the alert thresholds,
   you will receive a situation brief.

   View this AOI: {aoiUrl}
   Edit alert rules: {aoiUrl}/rules

   — Wildfire Nowcast
   Free, open, AI-native fire intelligence for stewardship — depth over speed.
   ```

   `regionBucketHumanized` is a one-liner that turns `5x5:W125_N35` into
   "5°×5° tile, SW corner 125°W 35°N" — a small helper in
   `lib/geo/region-bucket.ts` (read-only addition; do not change the
   bucket key format).

4. Call `sendImpl ?? sendEmail`. On `config_missing`, write a row with
   `status='config_missing'` and return that. On success, persist
   `status='sent'` and return the providerMessageId.

5. **No retry.** If the send fails, the row is `status='failed'`; we do
   not re-dispatch. The user can recreate the AOI if they don't get the
   email within minutes — and the *real* signal that watching is working
   is the first cron tick recording a `job_runs` row, which the AOI page's
   freshness banner (Stage 8) already surfaces.

#### First-poll time

Strawman: `firstPollAt = createdAt + 15 minutes` (the upper bound of the
cron interval). The cron actually fires every 15 min on a fixed schedule;
the user's AOI may be picked up earlier than +15 min. Stating the
upper bound is honest pessimism — a brief arriving sooner is a positive
surprise; one arriving "13 min late" because the user computed a shorter
interval is a trust break.

If Stage 9's backfill (below) is synchronous, the user often gets a brief
*before* the 15-min mark, in which case the email's "first poll at..."
line is already true at send time. The email copy should not over-claim
that the brief is already there; the backfill runs after the 201 returns,
so the email is sent before the backfill completes (or in parallel —
acceptable; idempotency hash makes ordering safe).

### First-AOI backfill (`lib/firms/backfill.ts` + AOI POST wiring)

**Dev decision per local feedback (strawman with rationale; resolve in
build):**

The cleanest design that fits Vercel's serverless model is **synchronous
backfill inside the AOI POST handler**, with these caveats:

- The route's `maxDuration` is currently default (10s on Hobby). The
  backfill needs longer — likely 20–40s for a worst-case bucket fetch +
  matcher + brief generation + dispatch. Vercel Hobby max is 60s for
  Node functions; bump `maxDuration = 60` on the AOI POST route exactly
  as the cron route does (`app/api/aoi/poll/route.ts:86`).
- A 60s 201 response is bad UX. Mitigation: do the backfill **after**
  responding 201. Two tactical paths:
  - **Path A — Vercel `waitUntil`** (`@vercel/functions` `waitUntil(promise)`).
    Schedules a promise to run after the response. Works on Vercel
    serverless. Documented in Vercel's runtime API.
  - **Path B — fire-and-forget setTimeout** (don't `await` the backfill;
    return 201 first). Works locally; on Vercel serverless the function
    instance can be torn down before the promise resolves.
  - **Path C — internal HTTP self-call** to a new `/api/aoi/{id}/backfill`
    route, fire-and-forget from the POST handler.

  **Strawman: Path A (`waitUntil`).** It's the documented Vercel pattern
  for "do this after responding," it does not require a new route, and it
  preserves the single AOI-creation transaction. If the dev agent finds
  `waitUntil` does not behave on the project's Vercel plan, fall back to
  Path C (the new route is small and the internal POST call is one
  fetch). Document the choice in the research log.

- **The backfill's failure path must not break AOI creation.** Wrap the
  backfill in a try/catch that logs and exits. The AOI is already
  persisted; a backfill failure is a soft failure that the next 15-min
  cron tick recovers from automatically.

**Backfill implementation** (`lib/firms/backfill.ts`):

```ts
export type BackfillOutcome = {
  aoiId: string;
  status: "ok" | "skipped" | "error";
  reason?: "config_missing" | "fetch_failed" | "no_detections";
  detectionsFetched: number;
  detectionsMatched: number;
  eventsCreated: number;
  briefsGenerated: number;
  notificationsSent: number;
  durationMs: number;
};

export async function backfillForNewAoi(
  db: AppDb,
  args: {
    aoiId: string;
    userId: string;
    regionBucket: string;
    now?: Date;
    fetchImpl?: FirmsFetchFn;
    briefGen?: typeof generateBriefForEvent;
    notifyDispatch?: typeof dispatchBrief;
  },
): Promise<BackfillOutcome>;
```

Behaviour:

1. Open a `job_runs` row with `job_name='aoi-backfill'`,
   `bucket=<regionBucket>`, `started_at=now()`, `status='running'`.
   Reuses the existing helper pattern from the cron route (lift the
   `openJobRun` / `closeJobRun` helpers into `lib/db/job-runs.ts` so both
   the cron route and backfill share them — small refactor).
2. `FIRMS_MAP_KEY` unset → close run as `status='ok'`,
   `outcome='success'`, no work done; return `{ status: "skipped",
   reason: "config_missing" }`. **Do not 503 the AOI POST.**
3. Resolve bucket bbox via `bucketToBbox(regionBucket)`. Call
   `fetchAreaCsv({ source: 'VIIRS_NOAA20_NRT', bbox, dayRange: 1 })`.
   On error, close run as error and return
   `{ status: "error", reason: "fetch_failed" }`.
4. Run the matcher with the *single-AOI* scope. The current
   `matchDetectionsToAois(db, { bucket, source, detections })` matches
   against all AOIs in the bucket. For a true single-AOI scope we add
   an optional `aoiIds?: string[]` filter. Strawman:
   `matchDetectionsToAois(db, { bucket, source, detections, aoiIds: [aoiId] })`.
   The matcher's SQL gains an `AND a.id = ANY(${aoiIds})` clause when
   `aoiIds` is provided. **No-op when `aoiIds` is undefined** so existing
   cron callers are untouched.
5. For each created event, run brief generation via `generateBriefForEvent`
   exactly as the cron does. For each generated brief, dispatch via
   `dispatchBrief` exactly as the cron does. Both calls reuse the
   existing skip-when-config-missing semantics.
6. Close the `job_runs` row with the aggregated counts.

**Why a separate `aoi-backfill` job_name and not `firms-poll`:** the
launch-readiness latency calculation (acceptance #5) needs to distinguish
"15-min-cron poll → brief" from "AOI-creation backfill → brief" because
the latter has a near-zero detection-to-poll lag by construction.
Lumping them inflates the P95.

#### AOI POST handler wiring (`app/api/aoi/route.ts`)

```ts
export const maxDuration = 60;
// (added at the top of the file alongside `runtime`)

export async function POST(req: NextRequest) {
  const parsed = await parseJson(req, aoiCreateSchema);
  if (!parsed.ok) return parsed.response;

  return withDb(async ({ db, userId }) => {
    const { aoi, rules } = await createAoi(db, { userId, ... });

    // After-response work: watch-confirmed email + backfill.
    // Both are best-effort — failures must not break AOI creation.
    const after = (async () => {
      try {
        await dispatchWatchConfirmed(db, {
          aoiId: aoi.id,
          userId,
          aoiName: aoi.name,
          regionBucket: aoi.regionBucket,
          areaHa: aoi.areaHa,
          firstPollAt: new Date(Date.now() + 15 * 60 * 1000),
          aoiUrl: absoluteAoiUrl(aoi.id),
        });
      } catch (err) {
        console.warn(`[aoi.post] watch-confirmed dispatch failed: ${err}`);
      }
      try {
        await backfillForNewAoi(db, {
          aoiId: aoi.id,
          userId,
          regionBucket: aoi.regionBucket,
        });
      } catch (err) {
        console.warn(`[aoi.post] backfill failed: ${err}`);
      }
    })();

    // Schedule after-response work without blocking the 201.
    waitUntil(after);

    return NextResponse.json({ aoi: ..., rules: ... }, { status: 201 });
  });
}
```

`waitUntil` from `@vercel/functions` (or `next/server` re-export — confirm
during build). In tests, the after-response promise is awaited via a
test-only injection (export `_runAfterResponseSync` flag) so PGlite tests
can assert email + backfill happened.

`absoluteAoiUrl` is a tiny helper that reads `process.env.NEXT_PUBLIC_APP_URL`
(already set, mirrors share-link minting). Locally falls back to
`http://localhost:3000`.

### Idempotency

Two layers:

1. **Watch-confirmed email**: idempotency hash =
   `sha256("watch_confirmed:" + aoiId)`. The dispatcher checks for a
   prior `notifications_log` row with `kind='watch_confirmed'` and this
   `target_hash` before sending. If a prior `status='sent'` exists → skip.
   This handles the (unlikely but possible) edge of `POST /api/aoi`
   being retried by a client.
2. **Backfill**: not idempotent in the schema sense — calling it twice on
   the same AOI within a minute would produce duplicate `job_runs` rows.
   That's fine; the underlying `firms_detections` insert is dedup'd by
   the existing `(source, lat, lon, detected_at)` unique constraint, the
   matcher dedups events, and brief generation skips already-briefed
   events (`already_briefed` skip reason). Cost: one extra FIRMS call
   per duplicate POST. Acceptable for v1.

### Tests

Mirror Stages 4 + 8:

1. **PGlite unit tests:**
   - `tests/notify/watch-confirmed.test.ts`:
     - happy path → asserts `notifications_log` row inserted with
       `kind='watch_confirmed'`, `status='sent'`, providerMessageId set.
     - duplicate call → second invocation returns
       `{ status: "skipped", reason: "duplicate" }`, no second row.
     - `RESEND_API_KEY` unset (stub `sendImpl` returning `config_missing`)
       → row inserted with `status='config_missing'`, outcome reflects.
     - JIT-placeholder email → `status='skipped'`,
       `reason='no_recipient'`, no `sendImpl` call (assert).
   - `tests/notify/watch-confirmed-template.test.ts`:
     - subject equals `Now watching {AOI name}`.
     - body contains AOI name, area, regionBucket humanized, firstPollAt,
       aoiUrl, positioning line footer.
   - `tests/firms/backfill.test.ts`:
     - happy path with stub FIRMS returning 2 detections that match the
       AOI → asserts `eventsCreated >= 1`, `briefsGenerated >= 0`
       (depending on gate stub).
     - `FIRMS_MAP_KEY` unset → returns `{ status: "skipped",
       reason: "config_missing" }`, no `firms_detections` rows written.
     - FIRMS fetch error → `{ status: "error", reason: "fetch_failed" }`,
       `job_runs` row closed with `status='error'`.
     - Single-AOI scope: a second AOI in the same bucket exists, but the
       backfill matcher does NOT create events for it (assert `aoi_events`
       has rows only for the new AOI).
   - `tests/aoi/post.after-response.test.ts`:
     - synchronous test mode (test-only flag): POST `/api/aoi` →
       assert 201 returned; assert `notifications_log` has the
       `watch_confirmed` row; assert `job_runs` has the `aoi-backfill`
       row.
     - email send fails (stub throws) → assert AOI was still created
       (the row exists in `aois`) and the 201 was returned.
     - backfill fails (stub throws) → same: AOI still created, 201
       returned.

2. **`@testcontainers/postgresql` integration test:**
   - `tests/aoi-creation.backfill.integration.test.ts`:
     - real Neon+PostGIS schema; stub the FIRMS client to return one
       detection inside the new AOI's polygon.
     - POST `/api/aoi` with a real polygon → 201 returns; await the
       test-only after-response signal → assert the new AOI exists,
       `aoi_events` has one event, `aoi_briefs` has zero or one brief
       (depending on gate), `notifications_log` has the watch_confirmed
       row.

3. **No live test gate.** The watch-confirmed email reuses Resend; the
   backfill reuses the FIRMS client. Both already have live-test paths in
   their respective stages (Stage 4 for Resend, Stage 2 for FIRMS).
   Stage 9 introduces no new external dep.

Default `pnpm test` runs everything. No new env vars required.

## Out of scope for Stage 9 (do NOT build)

- **Resend sender-domain verification.** Still on the Stage 4 deferred
  list; out of scope.
- **Async / queue-based backfill.** SPEC's Flow 1 step 6 says "first cron
  tick within 15 min." The strawman ships synchronous + `waitUntil`. A
  Vercel Queue / Inngest / Trigger.dev migration is a v1.1 question if
  `waitUntil` proves unreliable.
- **Backfill window > 24h.** The matcher and brief generator already
  scope to "current poll." Wider history is pulled by the launch-week
  query path (`listAllBriefsForUser`), not by backfill.
- **Watch-confirmed for AOI *updates*.** A geometry edit might warrant
  a "now watching the new shape" email; out of scope. Edits are rare in
  the v1 archetype (land-trust polygons are stable).
- **Backfill scoped to a *re-activated* AOI** (paused → unpaused).
  Same reasoning — out of scope. The next cron tick covers it.
- **Webhook channel for the watch-confirmed event.** US-2 acceptance has
  webhook channels for briefs; the watch-confirmed email is a one-shot
  bootstrap event, not a recurring brief. Email-only.
- **Authority-perimeter pre-fetch in the backfill path.** Stage 8's
  orchestrator wiring already runs perimeter fetch inside
  `generateBriefForEvent`, so backfill briefs get perimeters for free.
  No additional wiring needed; do not duplicate.
- **A new `/api/aoi/{id}/backfill` route** (Path C above). Only build
  if Path A (`waitUntil`) proves unreliable — and document the
  switchover in the research log.
- **Schema motion beyond the additive `notifications_log.kind` column +
  index.** No new tables, no columns dropped.

Do NOT touch `db/schema/postgis.ts`, the matcher's existing tests
(beyond the additive `aoiIds` parameter), the dispatcher's brief flow,
the Clerk webhook, the export routes, or any Stage 8 surface.

## Build-without-blocking discipline (per ADR 0006)

- **No new env vars required for production functioning.** The watch-
  confirmed email reuses `RESEND_API_KEY`; the backfill reuses
  `FIRMS_MAP_KEY`. Both handle their unset cases as "skip, log, return
  201."
- AOI POST returns 201 even if the watch-confirmed dispatch or backfill
  fails. Failures are logged, not surfaced to the API caller — the AOI
  was created, the next cron tick will recover.
- The `notifications_log.kind` column defaults to `'brief'` on existing
  rows; queries that filter by `kind='watch_confirmed'` see only the
  Stage 9-onwards rows. No data backfill required.

## Open questions to resolve during build (must answer before merge)

1. **`waitUntil` availability and behavior on this Vercel project.**
   `@vercel/functions` exports `waitUntil`; Next.js 16 also re-exports
   it via `next/server`. Verify the import works and that the AOI POST
   route's logs show the after-response work executing on Vercel
   Preview. If Preview logs show the function instance terminating
   before `waitUntil` completes, fall back to Path C (internal route
   self-call) — document in the research log.
2. **`brief_id` nullability on `notifications_log`.** Inspect the schema
   at branch start. If `NOT NULL`, drop it in this migration. If
   nullable, no change.
3. **Single-AOI matcher scope.** The matcher SQL is in
   `lib/firms/matcher.ts`. Adding an optional `aoiIds` filter must not
   change behavior when omitted. Add a test that runs the matcher with
   no `aoiIds` and asserts identical output to a baseline.
4. **`firstPollAt` truthfulness.** Strawman is `now + 15 min`. If the
   GH Actions cron schedule is `*/15 * * * *` (on the quarter), the
   actual next tick may be in 1–14 min. The honest copy is "by
   {next-quarter-hour-UTC}, usually within 15 minutes." Pick one and
   commit; do not invent a tighter SLA than the cron actually delivers.

## Workflow (per ADR 0006 + ADR 0007)

1. `git checkout -b stage-9-watch-confirmed-and-first-poll-backfill` off
   latest `master`.
2. Build in commits that group naturally (target: 5–7 commits, ≤500 LOC each):
   - `db/migrations/0007_stage9.sql` + `db/schema/index.ts` extension +
     migration test.
   - `lib/notify/watch-confirmed.ts` +
     `lib/notify/watch-confirmed-template.ts` + unit tests.
   - `lib/firms/backfill.ts` + `lib/firms/matcher.ts` `aoiIds` extension
     + unit tests.
   - `app/api/aoi/route.ts` `waitUntil` wiring + after-response tests.
   - Integration test (`tests/aoi-creation.backfill.integration.test.ts`).
   - Optional refactor commit: lift `openJobRun` / `closeJobRun` from
     the cron route into `lib/db/job-runs.ts` so the backfill shares them.
3. `pnpm typecheck && pnpm lint && pnpm test && pnpm build` — all green.
4. `git push origin stage-9-watch-confirmed-and-first-poll-backfill`.
5. Report back to PM_CLAUDE with: branch SHA, all check statuses, draft
   PR description in markdown, the `waitUntil` decision (Path A vs C).
6. PM_CLAUDE opens the PR via
   `gh pr create --draft --base master --label stage-pr:9`.
7. **Auto-merge applies.** Per ADR 0007, once CI is green and the
   reviewer subagent LGTMs, the PR auto-merges. Do not instruct any
   reviewer to "wait for Vanyo".

## Output

1. Branch on origin: `stage-9-watch-confirmed-and-first-poll-backfill`
2. Draft PR description (sections: Summary, What changed, How to test,
   Build-without-blocking notes, `waitUntil` decision, Things to challenge
   in review, Linked: brief 23 / ADR 0006 / ADR 0007 / spec §Flow 1 steps
   5+6 / spec §Acceptance #2 / launch-readiness #2 / research-log
   2026-05-07-v1-stop-line).
3. `pm/research-log/2026-05-XX-stage9-watch-confirmed-and-backfill.md` —
   what shipped, the `waitUntil` decision and why, any deviations from
   this brief, the `firstPollAt` copy decision, open questions for PM.

## Time budget

~3 hours. Stage 9 is smaller than Stage 8 (no external API risk; reuses
existing infra). Sharp edges:

- **`waitUntil` uncertainty on Vercel Hobby.** If the dev agent burns
  more than 30 minutes confirming `waitUntil` semantics, switch to
  Path C (internal route self-call) and move on.
- **Integration test flakiness from after-response timing.** The
  test-only `_runAfterResponseSync` flag is the forcing function — make
  it the default in tests, never in production.
- **Matcher `aoiIds` regression risk.** A wrong `WHERE` clause silently
  breaks the cron's full-bucket match. The "no `aoiIds` → identical
  output" baseline test is non-negotiable.

## Branch + label

- Branch: `stage-9-watch-confirmed-and-first-poll-backfill`
- PR base: `master`
- Label: `stage-pr:9` (PM_CLAUDE applies; informational under ADR 0007)
