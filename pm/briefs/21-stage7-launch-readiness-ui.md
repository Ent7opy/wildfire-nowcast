# Brief 21 — Stage 7: launch-readiness UI

## Why this exists

The 2026-05-07 product review (`pm/product-reviews/2026-05-07.md`)
concluded that Stages 0–6 shipped a working backend pipeline and a
working dashboard, but the v1 surface is **not yet shippable to a
non-Vanyo user**. Two acceptance items in `docs/SPEC-A-prime-v1.md` —
US-4's "polygon on MapLibre, last 90 days of detections as points" and
US-5's "snooze-24h / pause / unsubscribe links (signed token, no
login)" — were quietly deferred during Stage 6 with `TODO v1.1` markers
that the user will see literally rendered on screen
(`app/dashboard/aoi/[id]/page.tsx:55` prints
`TODO v1.1: MapLibre` to the user). At the same time the brief view
leaks operator telemetry (model id, prompt version, gate reason,
latency-ms, cost-est USD) into the stewardship reader's eyeline
(`app/dashboard/brief/[id]/page.tsx:43-47`), which the stewardship
positioning explicitly rejects.

Stage 7 is **not** new product surface. It is the closure of three
things the spec already requires for v1, plus two operational items
the review elevated:

1. The map (US-4) and a polygon draw control (US-1's third option),
   so a non-Vanyo first user can both create an AOI without a
   pre-exported `.geojson` file and verify what is actually being
   watched.
2. Signed-token snooze / pause / unsubscribe email endpoints (US-5),
   reusing the share-token plumbing pattern from Stage 6.
3. Cut operator telemetry from the user-facing brief footer.
4. A "Did this brief help?" feedback link in every email — one
   thesis-validation signal on a tool that today has zero feedback
   channel (review §4 and §5 #6).
5. A 14-day prune for `firms_detections`, called from the cron route,
   to honor the `docs/pivot-architecture.md` R5 mitigation.

This is the last stage before the launch-acceptance checklist in
`pm/launch-readiness.md` flips to all-passing. After Stage 7 merges,
the remaining work is content / outreach / one strategic decision on
authority-perimeter fetch (Stage 8, separate brief).

**Read in order:**
1. `pm/PM_CLAUDE.md`
2. `pm/decisions/0005-problem-chosen-a-prime.md`
3. `pm/decisions/0006-stage-pr-workflow.md` — workflow you follow
   (branch off `master`, draft PR, PM_CLAUDE opens it)
4. `pm/decisions/0007-*` — auto-merge gate (CI green + reviewer LGTM
   merges; do not instruct anyone to "wait for Vanyo")
5. `pm/product-reviews/2026-05-07.md` — the review this brief
   operationalizes (sections §2, §3, §5 #1, #2, #4, #6, #8)
6. `pm/launch-readiness.md` — the spec-acceptance map this stage is
   trying to flip green
7. `pm/blockers.md` — Vercel-Hobby clause is now an active blocker;
   does not gate this stage's merge but does gate the
   announcement-post in launch acceptance #9
8. `docs/SPEC-A-prime-v1.md` — §User stories US-1 (third creation
   path: draw on map), US-4 (map on AOI page + polygon visible + last
   90 days of detections), US-5 (snooze/pause/unsubscribe via signed
   tokens, no login); §Core flows Flow 4 step 5 (email body shape);
   §Acceptance for v1 launch items 1, 2, 5
9. `docs/pivot-architecture.md` — §6 R-table R5 storage cap (drives
   the prune step); §3 data model (`firms_detections.detected_at` is
   the prune column)
10. `pm/briefs/20-stage6-rules-ui-and-export.md` — the brief whose
    explicit deferrals (MapLibre, snooze/pause/unsubscribe) Stage 7
    now closes. Same `withDb` + repository + RSC pattern.
11. `app/dashboard/aoi/[id]/page.tsx`, `app/dashboard/aoi/new/page.tsx`,
    `app/dashboard/brief/[id]/page.tsx` — the three pages this stage
    edits. Note the literal `TODO v1.1: MapLibre` on the AOI page and
    the operator-telemetry footer on the brief page.
12. `lib/share/token.ts`, `lib/share/url.ts`,
    `app/api/brief/[id]/share/route.ts`,
    `app/brief/share/[token]/page.tsx` — the existing signed-token
    plumbing pattern. Stage 7's new notify-action endpoints follow
    the same shape: mint a 32-byte hex token at email-send time,
    persist to DB with an expiry, redeem via a public route.
13. `lib/notify/dispatch.ts`, `lib/notify/markdown.ts`,
    `lib/notify/resend.ts` — where the new signed links are injected
    into outbound email bodies and where the feedback link is
    appended.
14. `app/api/aoi/poll/route.ts` — the cron entry point. Stage 7 adds
    a single prune step at the top of the run (or as a final step;
    your call — doesn't matter as long as it runs once per tick).
15. `db/schema/index.ts` — `aoiRules.pausedUntil` and
    `aoiRules.notifyChannels` are already the right shapes; no
    structural changes there. New tables: `notify_action_tokens` and
    `brief_feedback`. See "Schema additions" below.

## Goal

Land — on a `stage-7-launch-readiness-ui` branch off `master` — a v1
surface a non-Vanyo first user can complete end-to-end:

- A MapLibre map on `/dashboard/aoi/[id]` rendering the AOI polygon
  with the last 90 days of FIRMS detections plotted as points (US-4).
- A MapLibre + polygon draw control on `/dashboard/aoi/new` as a
  third tab alongside Upload and Paste (US-1's third option).
- Signed-token snooze-24h / pause / unsubscribe links inside every
  outbound notification email, backed by `/api/notify/<action>/[token]`
  endpoints that require no login (US-5).
- A "Did this brief help? yes / no" pair of signed links in every
  email, backed by `/api/notify/feedback/[token]?v=yes|no` and a new
  `brief_feedback` table.
- The brief view's operator-telemetry footer hidden from end users.
- A prune step in the cron route that deletes `firms_detections`
  older than 14 days.

PR draft markdown ready; PM_CLAUDE opens the PR.

## Scope (strict)

### 1. Map on the AOI page (`app/dashboard/aoi/[id]/page.tsx`)

- Add `maplibre-gl` (peer-deps clean; no react wrapper required —
  initialize via `useEffect` in a small client component
  `app/dashboard/_components/aoi-map.tsx`).
- Tile source: **MapLibre demo style** (`https://demotiles.maplibre.org/style.json`).
  Free, attribution-required. Add the attribution control. Vector tiles
  are fine for v1 — we are showing a polygon, not raster basemaps.
  When (not if) v1.1 needs a real basemap, the swap is a one-line
  style URL change. Document this explicitly in a comment.
- Render the AOI polygon as a filled GeoJSON layer (low-opacity fill
  + 2px stroke).
- Fetch the last 90 days of `firms_detections` that were matched to
  this AOI (i.e. detections whose `aoi_events.aoiId` joined back to
  the polygon, in the 90-day window) and plot each as a circle marker
  sized by FRP. New repository function:
  ```ts
  listMatchedDetectionsForAoi(db, {
    userId, aoiId, sinceDays = 90
  }): Promise<Array<{ lat: number; lon: number; frpMw: number | null;
                      detectedAt: Date; satellite: string }>>
  ```
  Joins `firms_detections` → `aoi_events` (matched detections only)
  → `aois` (ownership). 90-day cap honored even if a detection's
  underlying event is older.
- Map zooms to fit the AOI bbox on first load. No interactive draw
  on this page.
- Remove the `TODO v1.1: MapLibre` text from the page.
- **Bundle-size discipline:** `maplibre-gl` is ~250 KB gzipped. Load
  the map component via `next/dynamic` with `ssr: false` so it is
  not in the dashboard's initial JS bundle.

### 2. Map + polygon draw on the create page (`app/dashboard/aoi/new/page.tsx`)

- Add a third tab `Draw` alongside `Upload` and `Paste`.
- The draw tab mounts `aoi-map.tsx` in editable mode with a
  Polygon-draw control. Use `@mapbox/mapbox-gl-draw` (works with
  MapLibre via the standard adapter
  `@watergis/maplibre-gl-export`-style pattern — or use
  `terra-draw` which is MapLibre-native and lighter). **Pick one;
  document the choice.** Either is acceptable — the dev agent's
  judgment on which has fewer transitive deps wins.
- On polygon completion, the drawn shape is converted to a GeoJSON
  Feature and POSTed to `/api/aoi` exactly like the Upload and Paste
  paths. No new server endpoint.
- Same name input + same client-side validation
  (`lib/validators/geojson.ts`).
- Same dynamic import discipline so the map JS only loads when the
  Draw tab is active.

This satisfies SPEC US-1 acceptance: "at least two of the three"
becomes all three.

### 3. Snooze / pause / unsubscribe signed-token endpoints

#### Schema addition — `notify_action_tokens`

```ts
export const notifyActionTokens = pgTable("notify_action_tokens", {
  token: text("token").primaryKey(),       // 32-byte hex
  aoiId: uuid("aoi_id").notNull().references(() => aois.id, { onDelete: "cascade" }),
  briefId: uuid("brief_id").references(() => aoiBriefs.id, { onDelete: "set null" }),
  action: text("action").notNull(),        // "snooze" | "pause" | "unsubscribe" | "feedback"
  channel: text("channel").notNull(),      // "email" — the channel the link was minted for
  target: text("target").notNull(),        // recipient address (the email)
  expiresAt: timestamp("expires_at", { withTimezone: true }).notNull(),
  redeemedAt: timestamp("redeemed_at", { withTimezone: true }),
  redeemedValue: text("redeemed_value"),   // for feedback: "yes" | "no"
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});
```

- Index on `(aoiId, action)` for the dispatcher's "have I already
  minted this for this brief?" check.
- 30-day expiry for snooze/pause/unsubscribe; 90-day expiry for
  feedback (review-window slack).
- Tokens are minted **inside the dispatcher** at email-send time —
  one fresh quartet (snooze, pause, unsubscribe, feedback) per
  outbound email. Do NOT reuse tokens across emails: a user who
  forwards an email should not be granting the recipient permission
  to pause their AOI.

#### New routes

All under `app/api/notify/`. Each is unauthenticated (the token IS the
auth), idempotent on the redemption side (re-clicking returns the same
"already done" page, not an error), and works without Clerk being
configured.

- `GET /api/notify/snooze/[token]` — sets
  `aoi_rules.paused_until = max(now + 24h, current paused_until)`,
  marks token redeemed, returns a tiny HTML page "Snoozed Foo
  Preserve until {ts}. Resume in dashboard." with a link to
  `/dashboard/aoi/[id]`.
- `GET /api/notify/pause/[token]` — sets
  `aoi_rules.paused_until = now() + 100 years` ("indefinite" — the
  user resumes via the dashboard). Same redemption page shape.
- `GET /api/notify/unsubscribe/[token]` — removes the **target email
  address** from `aoi_rules.notify_channels` for this AOI. Other
  channels (webhooks added later, secondary email recipients) remain.
  If after removal `notify_channels` is empty, the AOI is paused
  (same as Pause action) so the user is not silently in a state
  where polling continues but no one is told.
- `GET /api/notify/feedback/[token]?v=yes|no` — inserts a row into
  `brief_feedback`, marks the token redeemed with `redeemed_value =
  v`. Idempotent: re-clicking the same link with the same `v`
  returns the same thank-you page; clicking with the opposite `v`
  flips the recorded value (rationale: a user who mis-clicked
  shouldn't be locked into the wrong answer).

All four use `GET` (not `POST`) because email clients open links via
GET; this means each route must be safe against bot prefetching. Two
mitigations:
- The token is a 32-byte secret bearer — random prefetchers cannot
  guess it; only the recipient's own mail client / scanner can see
  it. This is the same threat model the existing share token
  operates under.
- The redemption page is HTML, not a 302 — Gmail's link scanner that
  GETs every link will record a redemption, but every action is
  recoverable from the dashboard, and a scanner-induced snooze
  resolves itself in 24h. Document the trade-off in a comment on the
  route handler.

#### Schema addition — `brief_feedback`

```ts
export const briefFeedback = pgTable("brief_feedback", {
  id: uuid("id").primaryKey().default(sql`gen_random_uuid()`),
  briefId: uuid("brief_id").notNull().references(() => aoiBriefs.id, { onDelete: "cascade" }),
  helpful: boolean("helpful").notNull(),
  recipientToken: text("recipient_token").notNull(),  // FK-by-string to notify_action_tokens.token
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
}, (t) => [
  uniqueIndex("brief_feedback_brief_token_uniq").on(t.briefId, t.recipientToken),
]);
```

The unique index makes the "click yes then click no" flip a single-row
update, not a duplicate insert.

#### Dispatcher changes (`lib/notify/dispatch.ts`)

- After building the recipient list and before calling `sendEmail`,
  mint four tokens for this `(brief, channel, target)` triple and
  insert them into `notify_action_tokens`.
- Render the four URLs into the email body. Use the existing
  `publicShareUrl` host pattern (`NEXT_PUBLIC_APP_URL`).
- Email body (Markdown, appended below the existing brief content):
  ```
  ---
  Was this brief useful? [Yes]({yes_url}) · [No]({no_url})
  · [Snooze 24h]({snooze_url})
  · [Pause this AOI]({pause_url})
  · [Unsubscribe]({unsub_url})
  ```
  The links are appended by a new helper `lib/notify/footer.ts`
  (small; deterministic; tested). The dispatcher passes the four URLs
  as a struct; `footer.ts` formats them.
- Idempotency: the existing `findExistingTerminalRow` check on
  `notifications_log` still gates re-sends; if a brief has already
  been sent to this target, no new tokens are minted.

#### Markdown renderer (`lib/notify/markdown.ts`)

- Currently does not handle `[label](url)` link syntax (per the file's
  own comment listing supported tags). Extend it to handle exactly
  this case: a single anchor on a line, no nesting. Tests assert
  escaping: a malicious URL with `"` is escaped; the label is escaped.
- Per `pm/briefs/20-stage6-rules-ui-and-export.md` "Time budget"
  decision, extend `lib/notify/markdown.ts` rather than introducing
  `marked` + DOMPurify.

### 4. Cut operator telemetry from the brief footer

In `app/dashboard/brief/[id]/page.tsx` (and any analogous public-share
page), remove the user-facing display of:
- `brief.model`
- `brief.promptVersion`
- `brief.gateReason`
- `brief.latencyMs`
- `brief.costUsdEst`

Keep `brief.createdAt` as "Posted: {date}" — that's user-relevant.

The columns themselves stay on `aoi_briefs`; this is purely a UI cut.
A future `/admin/brief/[id]` page is **out of scope** for Stage 7.

### 5. 14-day prune for `firms_detections`

In `app/api/aoi/poll/route.ts`, before (or after — order doesn't
matter, but document the choice) the per-bucket fan-out, run:

```sql
DELETE FROM firms_detections
WHERE detected_at < now() - interval '14 days';
```

Wrapped in a small helper `pruneOldDetections(db, { now, retentionDays = 14 })`
in `lib/firms/prune.ts`. Counted into the parent `job_runs` row as a
new column `detectionsPruned` (schema addition; nullable; integer).

**Test that brief generation is unaffected.** Specifically: brief
generation depends on `aoi_events` (which references the matched
detection by event row, not by `firms_detections.id`) and the
gate-reason logic in `lib/ai/gate.ts`. Verify by reading the gate code
that no path reaches back into `firms_detections` for rows older than
14 days. If any such path exists, surface immediately — do not silently
extend the retention window.

### Tests

Three layers, mirroring Stage 6:

1. **PGlite unit tests:**
   - `tests/notify/action-tokens.test.ts` — mint inserts a row;
     redemption marks redeemed; re-redemption returns same response;
     expired token returns null.
   - `tests/notify/snooze.test.ts` — `paused_until` advances by 24h;
     re-snoozing extends not shortens; pre-existing later
     `paused_until` is preserved.
   - `tests/notify/unsubscribe.test.ts` — removes the email channel;
     leaves other channels; auto-pauses if channels empty after.
   - `tests/notify/feedback.test.ts` — yes inserts row; flip to no
     updates the same row (unique index); double-yes is idempotent.
   - `tests/notify/markdown-link.test.ts` — `[label](url)` renders;
     `"` in URL is escaped; nested links rejected as plain text.
   - `tests/notify/footer.test.ts` — footer markdown carries all four
     URLs in stable order; deterministic.
   - `tests/notify/dispatch-tokens.test.ts` — dispatcher mints four
     tokens per outbound email and writes them into
     `notify_action_tokens`; on re-dispatch (idempotency-skip), no
     new tokens are minted.
   - `tests/firms/prune.test.ts` — deletes rows older than 14 days;
     leaves rows newer; returns deleted count; idempotent at zero.
   - `tests/dashboard/aoi-map-data.test.ts` — `listMatchedDetectionsForAoi`
     returns the right shape, honors the 90-day window, ownership-checks.

2. **`@testcontainers/postgresql` integration test:**
   - `tests/notify/full-action-flow.integration.test.ts` — stub auth
     as `user_alice`, POST AOI, run cron poll (stub FIRMS + AI Gateway
     + Resend with a recording sender), capture the email body,
     extract the four URLs, GET each, assert the resulting
     `aoi_rules` / `notifyChannels` / `brief_feedback` state.

3. **Component smoke tests:**
   - `aoi-map.tsx` smoke-renders given a stub `Polygon` and an empty
     detections array (jsdom does not run WebGL — assert that the
     component mounts without throwing and the map container element
     is present; do not assert map tiles render).
   - The new Draw tab on `/dashboard/aoi/new` mounts without throwing.
   - The brief page no longer renders any of the operator telemetry
     strings (`Model:`, `Gate reason:`, `latency`, `cost-est`).

Default `pnpm test` runs everything. New deps: `maplibre-gl`,
`@mapbox/mapbox-gl-draw` OR `terra-draw` (dev agent picks one).

### Schema migration

Yes, a new migration is required for this stage:
`db/migrations/0006_stage7.sql`:
- `CREATE TABLE notify_action_tokens (…)`.
- `CREATE TABLE brief_feedback (…)`.
- `ALTER TABLE job_runs ADD COLUMN detections_pruned integer`.

Use Drizzle's `pnpm db:generate` to scaffold, then hand-edit the SQL
to add the partial indexes that Drizzle's index DSL doesn't express.

## Out of scope for Stage 7 (do NOT build)

- **Authority-perimeter fetch** (review §5 #5). Stage 8.
- **`data_freshness` field on the AOI page** + FIRMS-429 honesty
  (review §5 #7). Stage 8.
- **Second-archetype outreach plan** (review §5 #11). PM-only docs;
  no code stage.
- **`/admin` page surfacing the cut operator telemetry.** Out of
  scope; the columns remain on `aoi_briefs` for future use.
- **Real basemap tiles (Stadia / Mapbox / Maptiler).** Stage 7 ships
  with the MapLibre demo style. Real basemap requires a paid key OR
  an attribution-only OSM raster source decision Vanyo has not yet
  made; do not silently sign up for a third-party tile account.
  Track as a v1.1 polish.
- **User-selectable share-link TTL** (spec open question 4). Still
  fixed 30 days for shares; 30 days for action tokens.
- **Mobile gesture polish on the map.** MapLibre's defaults are fine.
- **Rate-limit per-user on action endpoints.** Tokens are
  bearer-secret; rate-limiting public-by-design GETs is premature.
- **Onboarding "first poll scheduled" confirmation email** (review
  §3 missing-but-needed list, item 4). The Flow 1 step 5 confirmation
  email is a separate small chore — track as a follow-up; do not
  bundle into Stage 7's UI-heavy diff.
- **Backfill on first AOI** (Flow 1 step 6). Same rationale: separate
  small chore.
- **Cookie-based "I already snoozed" UI on `/api/notify/snooze`.**
  The redemption HTML page is sufficient.

Do NOT touch `db/schema/postgis.ts`, the `industrial_mask_static`
seed, the FIRMS client (only `firms/prune.ts` is new), `lib/ai/`, the
Clerk webhook handler, the dashboard chrome (Stage 6's layout), or
the export endpoints (Stage 6's outputs). Stage 7 is strictly
additive: new tables, new routes, new map component, plus three
small surgical edits (cron prune step, dispatcher footer injection,
brief-footer telemetry cut).

## Build-without-blocking discipline (per ADR 0006)

No new third-party env vars. The MapLibre demo style is a public URL
with no key. New surface degrades cleanly:

- `/api/notify/<action>/[token]` works without Clerk being
  configured (token IS the auth).
- The map components require no env vars; they fail gracefully (and
  the AOI page still renders polygon metadata) if the WebGL context
  cannot be acquired (jsdom, very old browsers).
- The dispatcher only mints action tokens when it is actually
  sending an email; no env vars trigger an extra third-party call.
- The cron prune step runs unconditionally; if `firms_detections` is
  empty (fresh deploy) it is a no-op.

Tests pass with no env vars set.

## Workflow (per ADR 0006 + ADR 0007)

1. `git checkout -b stage-7-launch-readiness-ui` off latest `master`
2. Build in commits that group naturally (target: 8–10 commits, ≤500 LOC each):
   - `db/migrations/0006_stage7.sql` + `db/schema/index.ts` additions
     (`notifyActionTokens`, `briefFeedback`, `jobRuns.detectionsPruned`)
   - `lib/firms/prune.ts` + cron-route hook + tests
   - `lib/notify/markdown.ts` link extension + tests
   - `lib/notify/footer.ts` (new helper) + tests
   - `lib/notify/dispatch.ts` token-mint + footer-render integration + tests
   - `app/api/notify/snooze/[token]/route.ts` + `pause` + `unsubscribe`
     + `feedback` + repository helpers + tests
   - `lib/db/aoi-repository.ts`: `listMatchedDetectionsForAoi` + tests
   - `app/dashboard/_components/aoi-map.tsx` (client component, dynamic-imported)
   - `app/dashboard/aoi/[id]/page.tsx`: mount the map, drop the TODO line
   - `app/dashboard/aoi/new/page.tsx`: third tab + draw integration
   - `app/dashboard/brief/[id]/page.tsx`: cut operator telemetry footer
   - integration test
3. `pnpm install` (`maplibre-gl` + draw library), `pnpm typecheck`,
   `pnpm lint`, `pnpm test` (Docker running locally for integration),
   `pnpm build` — all green
4. `git push origin stage-7-launch-readiness-ui`
5. Report back to PM_CLAUDE with: branch SHA, all check statuses,
   draft PR description in markdown
6. PM_CLAUDE opens the PR via
   `gh pr create --draft --base master --label stage-pr:7`
7. **Auto-merge applies** per ADR 0007.

## Output

1. Branch on origin: `stage-7-launch-readiness-ui`
2. Draft PR description in your reply (sections: Summary, What
   changed, How to test, Build-without-blocking notes, Things to
   challenge in review, Linked: brief 21 / product review
   2026-05-07 / spec US-1+US-4+US-5 / launch-readiness items 1+2+5)
3. `pm/research-log/2026-05-07-stage7-launch-readiness-ui.md` — what
   shipped, deferrals, deviations from brief, open questions for PM.
   Note any MapLibre / draw-library / token-redemption-UX surprises.

## Time budget

~5 hours (largest stage in the pivot — three independent surfaces:
map, action tokens, prune). If you hit a 20-minute block on any single
error, stop and report. Known sharp edges:

- **MapLibre + Next.js 16 RSC.** The map component is client-only;
  `next/dynamic({ ssr: false })` is mandatory. Hydration mismatches
  are the most likely failure mode; if the map's container has SSR
  output, hydration breaks.
- **GET-vs-POST for action endpoints.** The trade-off is documented
  above; if the dev agent thinks POST-with-meta-refresh is cleaner,
  surface and report — do not silently switch.
- **Draw library bundle size.** Whichever of `@mapbox/mapbox-gl-draw`
  vs `terra-draw` lands, document why.
- **Brief feedback "yes then no" flip.** The unique index forces an
  UPSERT path; PGlite's `ON CONFLICT … DO UPDATE` works, but verify
  before committing.

## Branch + label

- Branch: `stage-7-launch-readiness-ui`
- PR base: `master`
- Label: `stage-pr:7` (PM_CLAUDE applies; informational under ADR 0007)
