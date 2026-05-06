# Brief 20 — Stage 6: rules UI + export

## Why this exists

Stage 6 of the A' pivot. Stages 1–5 just merged: AOIs are created
(Stage 1), FIRMS detections matched and persisted (Stage 2), briefs
generated (Stage 3), emails dispatched (Stage 4), and the user is now
real — Clerk-authenticated, per-user-isolated (Stage 5). Today the only
way to interact with this pipeline is to POST GeoJSON at `/api/aoi`
with a Clerk session cookie. There is no dashboard, no per-AOI editor,
no way to inspect a brief that landed in your inbox, and no way to
walk away with your data.

Stage 6 is the stage where the product becomes a product. Everything
the spec calls a "page" appears here: the dashboard list, the AOI
editor, the rules form, the brief view, and the export endpoints that
satisfy US-6 (portability — table-stakes for the stewardship-user
thesis in `pm/north-star.md`).

This is also the last stage before launch readiness. Stage 7 (cutover)
already merged. Once Stage 6 lands, the remaining work is
content/polish/marketing — not engineering surface.

**Read in order:**
1. `pm/PM_CLAUDE.md`
2. `pm/decisions/0005-problem-chosen-a-prime.md`
3. `pm/decisions/0006-stage-pr-workflow.md` — workflow you follow (branch off `master`, draft PR, PM_CLAUDE opens it)
4. `pm/decisions/0007-*` — auto-merge gate (stage PRs auto-merge once CI is green and the reviewer LGTMs; do NOT instruct anyone to "wait for Vanyo")
5. `pm/blockers.md` — only active blocker is the Clerk webhook signing secret from Stage 5; Stage 6 does NOT introduce new infra blockers (no new env vars, no new third-party accounts).
6. `docs/SPEC-A-prime-v1.md` — §User stories US-2 (rules), US-4 (per-AOI page), US-6 (export); §Core flows Flow 2 (Create AOI), Flow 3 (Configure rules), Flow 7 (Review history); §API surface — the dashboard read paths, `GET /api/aoi/{id}/export?format=geojson|markdown`, `GET /api/brief/{id}`, `GET /api/brief/{id}/share/{token}`; §Acceptance for v1 launch items 7 (positioning line) and 8 (repo + license footer).
7. `docs/pivot-architecture.md` — §3 data model (`aois`, `aoi_rules`, `aoi_events`, `aoi_briefs` are all stable; no schema motion this stage), §6 R-table (R5 storage cap — export must be streamed for large brief sets, not buffered).
8. `pm/briefs/19-stage5-clerk-auth.md` — the `withDb` shape Stage 6 routes plug into; every UI route and every export endpoint funnels through `withDb` and gets `userId` from Clerk. The `<ClerkProvider>` wrap in `app/layout.tsx` is already present; the `ClerkConfigBanner` fallback path stays.
9. `pm/briefs/18-stage4-notification-dispatch.md` — recipient/channel model; the Stage 6 rules form writes the same `aoi_rules.notify_channels` shape the dispatcher reads. Quiet-hours digest merging (US-2 acceptance #3) stays out of scope (Stage 6 is rules-edit + skip-only delivery; the digest job is v1.1).
10. `db/schema/index.ts` — current schema is sufficient for Stage 6; the `aoi_briefs.share_token` and `share_expires_at` columns already exist (Stage 3) and are unused. Stage 6 wires them up.
11. `app/page.tsx`, `app/layout.tsx`, `app/sign-in/[[...rest]]/page.tsx`, `app/sign-up/[[...rest]]/page.tsx` — the only existing UI today; Stage 6 adds `app/dashboard/**` and the brief view, and replaces the "Not ready yet." footer on `app/page.tsx` with a "Sign in to start watching" CTA when Clerk is configured.
12. `app/api/aoi/route.ts`, `app/api/aoi/[id]/route.ts`, `app/api/aoi/[id]/rules/route.ts` — every CRUD path the dashboard needs is already there; Stage 6 calls them from React Server Components and the rules form's client action. No new mutation endpoints.
13. `lib/db/aoi-repository.ts` — read shapes returned by `listAois`, `getAoiById`, `getRulesByAoiId`. Stage 6 may need one new read (`listBriefsForUser` or `listBriefsForAoi`) — see Scope.
14. `lib/notify/dispatch.ts` — the dispatcher reads `notify_channels` and `paused_until`; Stage 6's rules form must produce shapes the dispatcher already understands (don't widen the schema mid-stage).

## Goal

Land — on a `stage-6-rules-ui-and-export` branch off `master` — the
authenticated dashboard surface (list of AOIs, per-AOI editor with
rules form, brief view), the AOI creation flow (GeoJSON upload + paste
only — map drawing deferred to v1.1), the export endpoints for AOIs
(GeoJSON FeatureCollection) and briefs (CSV + Markdown), the public
share-link route for individual briefs (consumes the existing
`aoi_briefs.share_token`), and the route-handler + component test
coverage for all of it. Build-without-blocking discipline holds: when
Clerk is unconfigured, marketing page renders unchanged and `/dashboard`
returns a friendly 503 banner. PR draft markdown ready; PM_CLAUDE opens
the PR.

## Scope (strict)

### No schema migration

`aois`, `aoi_rules`, `aoi_events`, `aoi_briefs` are stable. The
`aoi_briefs.share_token` and `share_expires_at` columns already exist
from Stage 3 (`db/schema/index.ts` lines 249–250) and have always been
nullable — Stage 6 begins populating them via a "share this brief"
toggle on the brief view. **No `db/migrations/0005_*.sql`** unless
something genuinely cannot be expressed in the current shape; if you
discover such a case, stop and report — do not widen mid-stage.

### Dashboard surface (`app/dashboard/**`)

All routes are React Server Components reading via `withDb`. Each
calls the existing repository functions; no parallel DB layer.

#### `app/dashboard/page.tsx` — AOI list

- Uses `withDb` server-side to call `listAois(db, userId)`.
- Renders a table: name, area (ha), region bucket, created-at, last
  brief timestamp, paused indicator.
- Empty state: a CTA card linking to `/dashboard/aoi/new`.
- Header with sign-out button (Clerk's `<UserButton />`).
- Mobile-responsive at 375 px (US-4 acceptance criterion is per-AOI
  page, but the dashboard inherits the same constraint by default).

To get "last brief timestamp" without N+1 queries, add one new
read to `lib/db/aoi-repository.ts`:

```ts
listAoisWithLatestBrief(db, userId): Promise<Array<AoiRow & {
  lastBriefAt: Date | null;
  pausedUntil: Date | null;
}>>
```

Single SQL with `LEFT JOIN LATERAL (SELECT created_at FROM aoi_briefs
WHERE aoi_id = aois.id ORDER BY created_at DESC LIMIT 1) b ON true`
plus `LEFT JOIN aoi_rules`. Two-backend discipline: PGlite supports
`LATERAL`, but if any driver edge-case bites, fall back to a window
function (`ROW_NUMBER() OVER (PARTITION BY aoi_id ORDER BY created_at
DESC)`) — both work on Postgres 16 and PGlite. Document the choice
inline.

#### `app/dashboard/aoi/new/page.tsx` — create AOI

Two tabs (US-1 acceptance: "at least two of the three [upload, draw,
paste] work in v1"):
- **Upload GeoJSON** — `<input type="file" accept=".geojson,application/geo+json,application/json">`. Client reads as text, validates shape (Polygon / MultiPolygon Feature or FeatureCollection-of-one), then POSTs to `/api/aoi`.
- **Paste GeoJSON** — `<textarea>`, same validation + POST path.

**Map drawing is deferred** — explicitly out of scope (see "Out of
scope"). US-1 is satisfied by upload + paste.

Form is a client component (`"use client"`) because file reads + JSON
validation happen in the browser. On success → router push to
`/dashboard/aoi/[id]`.

Reuse `lib/validators/aoi.ts` and `lib/validators/geojson.ts` for the
client-side preview; the server re-validates on POST regardless.

#### `app/dashboard/aoi/[id]/page.tsx` — AOI editor

- `withDb` → `getAoiById(db, userId, id)` → 404 path returns the spec'd
  `not_found` error rendered as a "Not found or not yours" page.
- Renders three sections, each a sub-component:
  1. **Summary** — name (editable inline via PATCH `/api/aoi/[id]`),
     polygon area, region bucket, created-at, archive button (DELETE).
     Polygon visualization is text-only for v1: a small static
     SVG-based bbox sketch and the centroid lat/lon. **No MapLibre.**
     (See "Out of scope".)
  2. **Rules form** — distance buffer (km), min confidence
     (low/nominal/high), min FRP (MW), quiet hours (tz dropdown +
     start/end hour inputs, all optional), paused-until (date-time or
     "not paused"), notification channels (list editor — add/remove
     `{type: "email"|"webhook", target}` rows). Submits via PUT
     `/api/aoi/[id]/rules`. Client component.
  3. **Recent briefs (last 20)** — reverse-chron list, each linking to
     `/dashboard/brief/[briefId]`. Server-side fetch via a new
     repository read `listBriefsForAoi(db, aoiId, userId, { limit: 20 })`
     that joins through `aois` to enforce ownership at the SQL level
     (no separate ownership check round-trip).

Quiet-hours UI shape mirrors the dispatcher's `inQuietHours` reader
(`lib/notify/dispatch.ts` lines 239–257): `{tz, startHour, endHour}`.
Use a small fixed list of common IANA tz strings for v1
(`America/Los_Angeles`, `America/New_York`, `America/Denver`,
`America/Chicago`, `Europe/London`, `Europe/Berlin`, `Europe/Athens`,
`Australia/Sydney`, `UTC`). Free-text tz entry is a v1.1 polish.

#### `app/dashboard/brief/[id]/page.tsx` — brief view

- Read via a new repository function `getBriefByIdForUser(db, userId,
  briefId)` — joins through `aois` to enforce ownership; returns null
  on miss. RSC.
- Renders `rendered_markdown` via `marked` (already a Stage 4 dep) into
  HTML. Sanitize with a tiny allow-list (no script tags; preserve
  `<a>`, `<strong>`, `<em>`, `<code>`, `<pre>`, `<ul>`, `<ol>`, `<li>`,
  `<p>`, `<table>` family, `<h1>`–`<h4>`). DOMPurify is heavy; a
  hand-rolled allow-list using `marked`'s renderer hooks is ~30 LOC.
  Reuse the brief renderer from Stage 3 if it already produces
  pre-sanitized HTML; do not double-render.
- Provenance footer: model id, prompt version, gate reason, latency,
  cost-est USD — all already on `aoi_briefs`.
- "Share this brief" toggle (client component): when on, POST to
  `/api/brief/[id]/share` (new — see below), which mints a
  `share_token` (32 hex bytes) and sets `share_expires_at` to `now() +
  30 days`. Display the resulting public URL
  `https://<host>/brief/share/[token]`. Toggle off DELETEs the same
  endpoint, NULLing both columns. The 30-day TTL matches spec US-4
  acceptance #2; user-selectable TTL (open question 4) is v1.1.

#### `app/brief/share/[token]/page.tsx` — public shared brief

- **Unauthenticated** RSC. Reads via a new helper
  `getBriefByShareToken(db, token, { now })` that returns null on
  miss, expiry, or NULL token.
- Renders the same brief view as the authed page, minus the share
  toggle, the provenance footer's cost numbers (operator-only — keep
  model id and posted-at), and the dashboard chrome.
- **Excluded from middleware-protected paths.** Update
  `app/middleware.ts` (introduced in Stage 5) to add `/brief/share/*`
  alongside `/sign-in/*`, `/sign-up/*`, the marketing root, the cron,
  and the Clerk webhook.

### Marketing page CTA (`app/page.tsx`)

Replace the "Not ready yet" footer with a small CTA block:
- When `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` is set → "Sign in to start
  watching" linking to `/sign-in`.
- When unset → keep the existing footer text (build-without-blocking).
- Add the v1 launch acceptance #7 / #8 footer line: the canonical
  positioning line *verbatim* and a "View source" link to the GitHub
  repo. (Repo URL: confirm from `package.json` "repository" or the
  remote; if missing, leave a `TODO Stage 6:` and append a blocker for
  Vanyo to confirm the public repo URL.)

### Export endpoints (`app/api/export/**`)

Three new routes, all authenticated via `withDb`. Per spec §API surface
the official spec path is `/api/aoi/{id}/export?format=geojson|markdown`
— Stage 6 ships **both** that single-AOI shape AND a portfolio-level
"all my AOIs / all my briefs" shape because US-6 talks about exporting
"my AOIs" plural.

#### `GET /api/aoi/[id]/export?format=geojson|markdown` (per spec)

- `format=geojson` → returns the AOI as a GeoJSON Feature with
  properties `{name, areaHa, regionBucket, createdAt, rules}` (rules
  inlined verbatim; the round-trip is the user's data). Single
  Feature, not a FeatureCollection.
- `format=markdown` → returns a single Markdown document containing the
  AOI metadata header + every brief's `rendered_markdown` concatenated
  in reverse-chron, separated by `---` rules. Includes the
  positioning-line footer and a link back to `/dashboard/aoi/[id]`
  (US-6 acceptance #2).
- Streams the response when the brief count is large (≥ 50 briefs):
  use a `ReadableStream` body and write briefs in batches. For the v1
  100-AOI / 500-brief target this matters precisely once; the streaming
  shape is cheap insurance. If the streaming shape proves fiddly under
  Next 16's response constraints, ship buffered with a hard guard at
  500 briefs (also per spec US-6 acceptance #1's "≤ 500 briefs"
  bound) and document the choice.
- Content-Type: `application/geo+json` and `text/markdown; charset=utf-8`.
- Content-Disposition: `attachment; filename="<aoiName-slug>.<ext>"`.
- 5s budget per US-6 acceptance #1; integration test asserts a 100-brief
  AOI exports in well under that.

#### `GET /api/export/aois.geojson` (portfolio)

- Returns all the user's non-archived AOIs as a GeoJSON
  FeatureCollection. Each Feature carries the same `properties` shape
  as the per-AOI export. Recipients export their full preserve roster
  for grant reporting (US-6 framing).
- Content-Type: `application/geo+json`.

#### `GET /api/export/briefs.csv` (portfolio)

- Returns the user's briefs (last 12 months by default; overridable via
  `?since=YYYY-MM-DD`) as CSV with columns: `brief_id, aoi_id, aoi_name,
  created_at, gate_reason, model, latency_ms, cost_usd_est,
  last_notified_at, summary` (summary extracted from the JSONB
  payload, double-quote-escaped).
- Streamed via a `ReadableStream` row-by-row using an ASYNC iterator
  over a query cursor pattern. PGlite does not support server-side
  cursors; for the test backend, fall back to chunked offset-paged
  reads (LIMIT 500 OFFSET N). Production (Neon) gets real streaming.
- Content-Type: `text/csv; charset=utf-8`.
- Content-Disposition: `attachment; filename="briefs.csv"`.

### Public share endpoint helpers (`app/api/brief/[id]/share/route.ts`)

- `POST` → mints a token (`crypto.randomBytes(32).toString("hex")`),
  sets `share_expires_at = now() + 30 days`, returns
  `{token, expiresAt, publicUrl}`. Idempotent: if a non-expired token
  already exists for this brief, return it (don't churn the URL).
- `DELETE` → NULLs both columns. Returns `{ok: true}`.
- Both gated by `withDb` and an ownership check via the shared
  `getBriefByIdForUser` helper.

### Middleware update (`app/middleware.ts`)

Add `/brief/share/:path*` to the public matcher (alongside `/`,
`/sign-in/*`, `/sign-up/*`, `/api/aoi/poll`, `/api/webhooks/clerk`).
Add `/dashboard/:path*` to the protected matcher. Keep the
build-without-blocking pass-through when `CLERK_SECRET_KEY` is unset.

### Inject points for tests

- New repository functions (`listAoisWithLatestBrief`,
  `listBriefsForAoi`, `getBriefByIdForUser`, `getBriefByShareToken`,
  `setBriefShareToken`, `clearBriefShareToken`) all accept `now?: Date`
  where they touch time, mirroring the Stage 4 pattern.
- The export route handlers accept a `_setTestNow(fn | null)` injection
  so the CSV / Markdown deterministic-output tests don't drift on
  clock.

### Tests

Three layers, mirroring Stages 3–5:

1. **PGlite unit tests** (no Docker, fast):
   - `tests/dashboard/list-aois.test.ts` — `listAoisWithLatestBrief`
     returns the right shape; latest brief join is correct (seed 2 AOIs,
     each with 0/1/3 briefs, assert latest timestamp); excluded by
     `archivedAt IS NOT NULL`; user isolation (Stage 5's repository
     test pattern — seed two users, assert no leak).
   - `tests/dashboard/list-briefs-for-aoi.test.ts` — limit honored;
     ownership enforced (querying user A for user B's AOI returns null,
     not throws).
   - `tests/dashboard/get-brief-for-user.test.ts` — happy path; cross-user
     access returns null; soft-deleted AOI's brief returns null.
   - `tests/dashboard/share-token.test.ts` — mint creates token + sets
     expiry; second mint on same brief returns existing token; clear
     nulls both; `getBriefByShareToken` returns null after expiry;
     returns null for unknown token.
   - `tests/export/aoi-geojson.test.ts` — single AOI Feature shape;
     properties carry rules; ownership enforced (404 for other-user
     AOI).
   - `tests/export/aoi-markdown.test.ts` — ordering reverse-chron;
     positioning-line footer present verbatim; link back to dashboard
     URL present.
   - `tests/export/portfolio-geojson.test.ts` — FeatureCollection
     contains user's non-archived AOIs only.
   - `tests/export/portfolio-csv.test.ts` — header row matches spec;
     CSV-escaping correct for summaries containing `"` and `,`; `since`
     filter honored.
   - `tests/dashboard/middleware.test.ts` — `/dashboard/*` is in the
     protected matcher; `/brief/share/*` is in the public matcher; pass-through
     when Clerk is unconfigured.

2. **`@testcontainers/postgresql` integration test:**
   - `tests/dashboard/full-flow.integration.test.ts` — stub auth as
     `user_alice`, POST AOI, run cron poll (stub FIRMS + AI Gateway +
     Resend), assert dashboard read returns the AOI with the brief
     attached, assert brief view renders, assert the per-AOI Markdown
     export contains the brief's `rendered_markdown`.

3. **Component smoke tests** — Vitest + React Testing Library is
   already set up if Stage 0 wired it; if not, add `@testing-library/react`
   + `jsdom` (small, well-typed). One smoke test per page-level component:
   renders without throwing given a stub data prop. **No Playwright.**
   E2E browser tests are deferred until v1.1 — the ROI for solo-maintainer
   v1 is too low and the surface is small enough that route + component
   tests cover the regression cases that matter.

Default `pnpm test` runs everything. No new live-test scaffold (no new
third-party API in this stage).

## Out of scope for Stage 6 (do NOT build)

- **MapLibre / map rendering anywhere.** Spec US-1 says draw-on-map is
  one of three valid create paths; upload + paste already satisfy the
  "two of three" requirement. Spec US-4 asks for a map on the AOI
  page; Stage 6 ships a text-only summary (centroid + bbox SVG) and a
  `TODO v1.1: MapLibre` marker. Rationale: MapLibre + Deck.GL is a
  meaningful bundle-size and styling cost, and the launch archetype
  (LTA stewardship directors) already runs ArcGIS / QGIS for spatial
  context — the dashboard's job is the watching loop, not the map.
- **Snooze / pause / unsubscribe signed-token links from email** (US-5).
  These need an inbound `POST /api/notifications/webhook/{token}`
  route + a token-mint flow inside the dispatcher. Defer to v1.1; the
  rules-form `paused_until` field gives the user the same control via
  the dashboard, which is the v1 acceptance.
- **Quiet-hours digest merging** (US-2 acceptance #3). The Stage 4
  dispatcher's `// TODO Stage 6:` marker stays — Stage 6 ships
  rules-edit only; the digest cron is v1.1.
- **BYO Gemini key UI / decryption** (US-7). The `gemini_api_key_enc`
  column stays untouched. Open question #2 in the spec defers this
  decision to Vanyo; Stage 6 ships nothing for it.
- **MCP / bearer-token API access** (US-8). v2 work.
- **Authority-perimeter UI** (spec open question #3). Field stays
  `null` in v1.
- **User-selectable share-link TTL** (spec open question #4). v1 is
  fixed 30 days; the schema column allows a future override.
- **Account settings page beyond Clerk's `<UserButton />`.** Email
  preferences, profile editing, account deletion — Clerk owns
  identity; the dashboard owns AOIs. Settings beyond AOIs is v1.1.
- **`/api/me` extension** (display-name, etc.). Stage 5 brief deferred
  this; Stage 6 inherits the deferral.
- **Rate-limit per-user 60 req/min** (spec §API surface). v1 launch
  traffic is light; rate-limiting without a measured looping pattern
  is premature.
- **Stage 4 dispatcher reviewer follow-ups** — see "Bundling
  decision" below. Three small fixes to `lib/notify/dispatch.ts` are
  tracked separately as a Stage 5.5 chore. They do NOT belong in this
  PR.

Do NOT touch `db/schema/postgis.ts`, the `industrial_mask_static`
seed, any FIRMS code paths, `lib/ai/`, the cron route, or the Clerk
webhook handler. Stage 6 is strictly an additive UI + export layer
plus one share-token mint endpoint.

## Bundling decision — Stage 4/5 dispatcher follow-ups

Three reviewer notes from Stages 4 and 5 are pending:
1. Webhook re-runs return `channel_not_implemented` instead of
   `duplicate` (`findExistingTerminalRow` only checks `channel = 'email'`).
2. Missing-user case (no `users` row) does not produce a
   `skipped/no_recipient` log row.
3. Stage 5 JIT-provisioned users with `<userId>@pending.invalid`
   placeholder emails are sent to anyway; should be treated as
   `no_recipient` and skipped.

**Decision: split.** These are dispatcher-internal bugs on a backend
code path that Stage 6 does not touch (Stage 6 reads
`notify_channels` and writes via the Stage 1 PUT route, but the
dispatcher itself is not on the Stage 6 surface). Bundling them adds
two unrelated commits to a UI-heavy diff and complicates reviewer
focus. Track as a separate Stage 5.5 chore brief (~50 LOC + 3 unit
tests, ~30-min job). PM_CLAUDE will queue the chore brief in the next
heartbeat tick.

If during build the dev agent finds a Stage 6 acceptance criterion
that genuinely depends on one of these fixes (e.g. a dashboard test
that exercises the dispatcher and trips over the `pending.invalid`
case), surface immediately — do not silently re-bundle.

## Build-without-blocking discipline (per ADR 0006)

No new env vars. All build-without-blocking surface is inherited from
Stages 3–5:

- `app/dashboard/*` is gated by Stage 5 middleware → 503 banner when
  Clerk is unconfigured (the existing `ClerkConfigBanner` covers it).
- `/brief/share/[token]` is unauthenticated and reads only the share
  token + brief — no Clerk dependency, no Resend / AI Gateway
  dependency. It works in any deploy.
- The export endpoints depend only on Postgres reads (`withDb`); no
  third-party calls.
- Marketing page CTA: when Clerk is unconfigured, falls back to the
  current footer (no broken sign-in link).
- Tests pass with no env vars set.

## Workflow (per ADR 0006 + ADR 0007)

1. `git checkout -b stage-6-rules-ui-and-export` off latest `master`
2. Build in commits that group naturally (target: 7–10 commits, ≤500 LOC each):
   - `lib/db/aoi-repository.ts` extensions: `listAoisWithLatestBrief`,
     `listBriefsForAoi`, `getBriefByIdForUser`, `getBriefByShareToken`,
     `setBriefShareToken`, `clearBriefShareToken` + unit tests
   - `app/dashboard/page.tsx` (list) + `app/dashboard/layout.tsx` (chrome)
   - `app/dashboard/aoi/new/page.tsx` (create flow, client component)
   - `app/dashboard/aoi/[id]/page.tsx` (editor) + sub-components for
     rules form (client) + brief list
   - `app/dashboard/brief/[id]/page.tsx` (brief view) + sanitizer
   - `app/api/brief/[id]/share/route.ts` (mint/clear) + tests
   - `app/brief/share/[token]/page.tsx` (public view) + middleware
     update
   - `app/api/aoi/[id]/export/route.ts` (per-AOI geojson + markdown) +
     tests
   - `app/api/export/aois.geojson/route.ts` + `app/api/export/briefs.csv/route.ts`
     (portfolio) + tests
   - `app/page.tsx` CTA update
   - integration test
3. `pnpm install` (likely `@testing-library/react` + `jsdom` if not
   present), `pnpm typecheck`, `pnpm lint`, `pnpm test` (Docker
   running locally for integration coverage), `pnpm build` — all green
4. `git push origin stage-6-rules-ui-and-export`
5. Report back to PM_CLAUDE with: branch SHA, all check statuses, draft
   PR description in markdown
6. PM_CLAUDE opens the PR via `gh pr create --draft --base master --label stage-pr:6`
7. **Auto-merge applies.** Per ADR 0007, once CI is green and the
   reviewer subagent LGTMs the diff, the PR auto-merges. The
   `stage-pr:6` label is informational. Do not instruct any reviewer
   to "wait for Vanyo".

## Output

1. Branch on origin: `stage-6-rules-ui-and-export`
2. Draft PR description in your reply (sections: Summary, What changed,
   How to test, Build-without-blocking notes, Things to challenge in
   review, Linked: brief 20 / ADR 0006 / ADR 0007 / spec §User stories
   US-2 + US-4 + US-6 / pivot-arch §3)
3. `pm/research-log/2026-05-06-stage6-rules-ui-and-export.md` — what
   shipped, deferrals, deviations from brief, open questions for PM.
   Note any RSC / streaming surprises (Next.js 16 + Cache Components
   mode — see "Time budget" below).

## Time budget

~4 hours (Stage 6 is the largest UI surface in the pivot; a half-step
above the Stage 3/4/5 ~3-hour baseline). If you hit a 20-minute block
on any single error, stop and report. The three known sharp edges:
- **Next.js 16 Cache Components and `<ClerkProvider>`.** The Stage 5
  brief flagged this. The dashboard pages are dynamic per-user and
  must NOT be cached; ensure no `use cache` directives leak in. RSC
  + Clerk's `auth()` should "just work" but if a render boundary
  surprises you, document it.
- **Streaming `ReadableStream` responses from a Next route handler.**
  The Web Fetch `Response` body accepts a `ReadableStream`; Next 16's
  route handler runtime supports it on the Node runtime (which all
  these routes use, per the `pg` driver constraint). If the stream
  closes early or the test framework can't consume it, fall back to
  buffered + the 500-brief hard guard. Document the choice.
- **Markdown sanitization.** `marked` with a hand-rolled allow-list
  is the path; do NOT add DOMPurify (32 KB of jsdom setup) just for
  this. If the allow-list grows past ~50 LOC, consider
  `isomorphic-dompurify` and document.

## Branch + label

- Branch: `stage-6-rules-ui-and-export`
- PR base: `master`
- Label: `stage-pr:6` (PM_CLAUDE applies; informational under ADR 0007)
