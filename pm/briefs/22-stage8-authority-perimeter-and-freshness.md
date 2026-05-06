# Brief 22 — Stage 8: authority-perimeter tool-call + data-freshness honesty

## Why this exists

Stage 7 (cutover / launch readiness) just merged. The product-reviewer pass on
2026-05-07 named two thesis-adherence gaps the launch UI does not resolve:

1. **Briefs feel "L1.5", not L2** (review §1, §3 "Built but underdeveloped").
   The schema has `authority_perimeter` and `weather_note`, but
   `lib/ai/generate.ts:93` hardcodes `authorityPerimeter: null` and the prompt
   builder faithfully tells the model to leave the field null. The thesis's
   load-bearing sample paragraph ("the authority perimeter posted 90 minutes
   ago covers the detection cluster") is structurally unreachable today
   because the input is never fetched. SPEC open question 3 punted this to
   v1.1; the review escalated it to launch-blocking for thesis adherence.

2. **The system is silently dishonest about its own freshness**
   (review §4, §5 #7). The cron is real and it records per-bucket
   `job_runs` rows, but the AOI page surfaces *briefs*, not *poll outcomes*.
   A user whose bucket is rate-limited or whose cron tick failed sees an empty
   brief list and has no way to distinguish "nothing happening" from "we
   stopped watching." For a stewardship tool whose contract is "we are
   watching your place," this is a quiet trust break.

Stage 8 closes both gaps. The third item in this stage is PM-only: identify
second-archetype outreach contacts (review §5 #11) so post-launch week 2 isn't
a blank page if LTA WRN doesn't bite.

**Read in order:**

1. `pm/PM_CLAUDE.md` — doctrine, especially "no fabricated data"
2. `pm/decisions/0006-stage-pr-workflow.md` — branch / draft PR / label workflow
3. `pm/decisions/0007-*` — auto-merge gate (CI green + reviewer LGTM merges; no
   "wait for Vanyo" instructions)
4. `pm/blockers.md` — sole pre-existing blocker is the Clerk webhook signing
   secret (Stage 5 holdover); Stage 8 may add new blockers if any of the three
   authority endpoints turn out to require keys (see "Open questions").
5. `pm/product-reviews/2026-05-07.md` — particularly §1, §4, and §5 items
   #5, #7, #11.
6. `docs/SPEC-A-prime-v1.md` — §LLM brief format (the schema this brief
   improves), §Open questions question 3 (authority perimeters — Stage 8 is
   the v1.1 it referenced, brought forward), §Acceptance for v1 launch.
7. `docs/pivot-architecture.md` — §3 data model (no schema motion this stage
   except an additive `job_runs` widening — see Scope), §6 R-table R3 (FIRMS
   rate limits — the freshness surface is the user-visible mitigation).
8. `lib/ai/generate.ts` — orchestrator; the place that today writes
   `authorityPerimeter: null` (line 93). The Stage 8 fanout happens here.
9. `lib/ai/gateway.ts` — `generateBriefViaGateway` calls `generateObject` from
   `ai@^6.0.175`. **Critical:** `generateObject` does NOT accept a `tools`
   parameter; tool-calling is on `generateText` / `streamText`. See
   "Open questions" for the implementation-pattern decision.
10. `lib/ai/prompt.ts` — `BriefContext.authorityPerimeter` shape + the
    "do not invent values" rule. Stage 8 keeps the rule and starts populating
    the field.
11. `lib/ai/schema.ts` — `BriefAuthorityPerimeterSchema` (source / posted_ts /
    contains_detection). The shape does not change in v1.
12. `app/api/aoi/poll/route.ts` — cron route; freshness recording must wrap
    each per-bucket attempt at start AND completion.
13. `app/dashboard/aoi/[id]/page.tsx` — where the freshness banner lands.
    The existing `BBox: ... TODO v1.1: MapLibre` line stays untouched
    (different stage's deferral).
14. `db/schema/index.ts` — `job_runs` (lines 313–328) already records per-bucket
    children with `bucket`, `started_at`, `finished_at`, `status` (`ok` /
    `partial` / `error` / `running`), and `error`. **No new table required.**
    Stage 8 adds two nullable columns and one index.
15. `pm/briefs/20-stage6-rules-ui-and-export.md` for tone, structure, and the
    auto-merge / build-without-blocking discipline established for stage briefs.

## Goal

Land — on a `stage-8-authority-perimeter-and-freshness` branch off `master` —
two product-visible improvements:

1. **Authority-perimeter context in briefs.** The brief generator fetches
   the most recent authority-published perimeter near the detection from the
   region-appropriate public GeoJSON source (NIFC for US, ICNF for Portugal,
   CWFIS for Canada). The fetched perimeter is folded into the prompt and the
   model populates `context.authority_perimeter` with real values
   (`source`, `posted_ts`, `contains_detection`) instead of nulls.
2. **Per-bucket freshness honesty on the AOI page.** Each AOI surface shows
   "last polled N minutes ago" and a yellow warning when the most recent
   poll attempt for that AOI's bucket failed, was rate-limited, or is stale
   (>30 min since last successful tick).

Test coverage at the levels Stages 3–6 established. Build-without-blocking
discipline holds: if any authority endpoint is unreachable, the field stays
null and the brief still ships. PR draft markdown ready; PM_CLAUDE opens the
PR.

The third deliverable, `pm/outreach-plan-v1.md`, is PM-only, lands in the
same chore PR as this brief if Vanyo prefers, or has already been written
alongside this brief — it does not require dev-agent work.

## Scope (strict)

### Schema motion (additive only)

`db/migrations/0006_stage8.sql`:

```sql
ALTER TABLE "job_runs"
  ADD COLUMN "outcome" text,           -- success | rate_limited | network_error | timeout | partial
  ADD COLUMN "retry_pending" boolean NOT NULL DEFAULT false;

CREATE INDEX "job_runs_bucket_started_at_idx"
  ON "job_runs" ("bucket", "started_at" DESC)
  WHERE "bucket" IS NOT NULL;
```

Drizzle-side: extend `jobRuns` in `db/schema/index.ts` with the two new
columns (`outcome: text("outcome")`, `retryPending: boolean("retry_pending").notNull().default(false)`).

**Why no new `firms_polls` table:** `job_runs` already records per-bucket
children with the right shape (Stage 2). A second table would duplicate the
data and force the freshness query to UNION across two sources. The product
review's "`firms_polls` table or extend `job_runs`" question resolves to
**extend `job_runs`** based on schema inspection — both options were
acceptable to the reviewer; the existing shape is closer.

The existing `status` column ("ok" / "partial" / "error" / "running") is
operator-facing and stays. `outcome` is the *user-facing* taxonomy
(`success` / `rate_limited` / `network_error` / `timeout` / `partial`) that
maps to the UI banner copy. Two columns, two audiences. The mapping is a
pure-TS function, tested.

### Authority-perimeter fetch (`lib/ai/authority/`)

New module. Three small files keep responsibility tight:

#### `lib/ai/authority/sources.ts`

Static catalog of public authority GeoJSON endpoints. Format:

```ts
export type AuthoritySource = {
  id: "nifc" | "icnf" | "cwfis";
  name: string;             // "NIFC WFIGS" | "ICNF" | "CWFIS"
  /** ArcGIS / GeoServer / static URL — CONFIRM in "Open questions" before merge. */
  url: string;
  /** Bucket prefixes this source serves. 5°×5° prefix match against `region_bucket`. */
  bucketPrefixes: string[];
};
```

Stage 8 ships **only sources whose public, key-free GeoJSON endpoint is
confirmed** (see "Open questions"). If any source needs a key, surface as a
blocker — do NOT hardcode the URL and silently 404. **No fabricated endpoints.**

#### `lib/ai/authority/fetch.ts`

```ts
export type AuthorityPerimeter = {
  source: string;       // catalog `name`
  postedTs: string;     // ISO 8601 from the GeoJSON feature
  containsDetection: boolean;
  /** Original feature for debugging — NOT persisted, NOT shown to LLM. */
  rawFeatureId?: string;
};

export async function fetchAuthorityPerimeter(args: {
  lat: number;
  lon: number;
  radiusKm: number;     // default 25
  regionBucket: string;
  now?: Date;
  fetchImpl?: typeof fetch;   // test injection
  timeoutMs?: number;          // default 10s
}): Promise<AuthorityPerimeter | null>;
```

Behaviour:

- Look up `regionBucket` against `sources.ts`. If no source covers it,
  return `null` (e.g. an AOI in Australia or Brazil today).
- `fetch` the source GeoJSON with `AbortSignal.timeout(timeoutMs)`.
- Parse the FeatureCollection. Filter to features whose centroid is within
  `radiusKm` of `(lat, lon)` (haversine; no PostGIS round-trip — the data
  volume per fetch is small). Among those, pick the most recent by the
  source's published-timestamp property (per-source mapping in `sources.ts`,
  e.g. NIFC's `attr_PolygonDateTime`).
- `containsDetection` is computed via a point-in-polygon test against the
  detection lat/lon (use `@turf/boolean-point-in-polygon`; turf is already a
  repo dependency from `lib/firms/matcher.ts`'s fallback path).
- Build-without-blocking: any of {404, 5xx, network error, timeout, parse
  error, no features in radius} returns `null`. **No retry, no backoff in
  v1** — the next 15-min cron tick is the retry. Emit a `console.warn` with
  the source id and error category so cron logs are diagnosable.

#### `lib/ai/authority/tool.ts`

The Vercel AI SDK tool definition that wraps `fetchAuthorityPerimeter` for
LLM tool-calling:

```ts
import { tool } from "ai";
import { z } from "zod";

export const fetchAuthorityPerimeterTool = (deps: {...}) => tool({
  description: "Fetch the most recent authority-published fire perimeter near (lat, lon)...",
  parameters: z.object({
    lat: z.number(), lon: z.number(), radius_km: z.number().default(25),
  }),
  execute: async ({ lat, lon, radius_km }) => {
    const r = await fetchAuthorityPerimeter({...});
    return r ?? { source: null, posted_ts: null, contains_detection: null };
  },
});
```

**Critical implementation question (resolved here, not deferred):** the
existing `lib/ai/gateway.ts` calls `generateObject`, which in `ai@^6.0.175`
**does not accept a `tools` parameter** — tools are exclusive to
`generateText` / `streamText`. Two paths exist:

- **Path A (preferred):** keep `generateObject` for the structured-output
  contract. Pre-fetch the perimeter in the orchestrator (`lib/ai/generate.ts`)
  before building the prompt. Pass the result into `BriefContext.authorityPerimeter`
  exactly the way `priorEvents` is already passed. Pros: no SDK gymnastics,
  one network round-trip, schema enforcement preserved. Cons: technically
  "the orchestrator decides when to call," not "the model decides."
- **Path B (literal LLM tool-call):** add a `generateText`-with-tools step
  before `generateObject`. The model receives the event facts, decides
  whether to call `fetchAuthorityPerimeter`, the tool runs server-side, the
  result is added to the message history, then a second `generateObject`
  call produces the schema-valid brief. Pros: matches the product review's
  "LLM tool-call" framing. Cons: two LLM calls per brief (cost ~2x), more
  ways for the model to misbehave (skip the tool, call it with wrong args),
  more SDK surface to test.

**Decision:** Ship **Path A** in Stage 8. Document Path B as a v1.1 follow-up
in the research log. Rationale: the product-review #5's load-bearing claim
is "this data should be in the brief," not "the model must literally invoke
the tool." Path A delivers the user-visible improvement at half the cost
and one-tenth the integration risk; Path B is a correctness-of-mechanism
improvement we can layer on once it has measurable value (e.g. when the
model sometimes shouldn't fetch, like for archived AOIs).

If during build the dev agent finds a clean way to do Path B without
doubling the gateway call (e.g. a single `generateObject` call with a
`prepareStep` hook that pre-runs tools — confirm against the AI SDK v6
docs), prefer it. Otherwise Path A. Either way, the **`fetchAuthorityPerimeterTool`
helper is still defined in `tool.ts`** so the v1.1 swap is a one-call change
and the test surface stays.

#### Orchestrator wiring (`lib/ai/generate.ts`)

- New private helper `gatherAuthorityPerimeter(loaded, deps)` that calls
  `fetchAuthorityPerimeter` with the AOI centroid + the nearest detection's
  `region_bucket`. Skips the call if `nearestDetection` is null (no point to
  test against) or if the bucket isn't in the source catalog.
- The result populates `BriefContext.authorityPerimeter` instead of the
  hardcoded `null` on line 93. Shape mapping: `source` → `source`,
  `postedTs` → `postedTs`, `containsDetection` → `containsDetection`.
- The fetch failure path returns `null` and the brief generates exactly as
  it does today — no new error branch.
- `GeneratorDeps` gains an optional `fetchPerimeter` injection mirroring
  the existing `gateway` injection, for tests.

#### Persistence

The Stage 3 schema already persists the fetched perimeter inside the
`payload` JSONB blob — `BriefContextSchema.authority_perimeter` is part of
the Brief shape that's stored in `aoi_briefs.payload`. **No separate column
required.** (The product-review prompt's "the dormant column on the schema
since Stage 3" is incorrect; what's dormant is the *field within the
JSONB*, not a top-level column. Brief will say so plainly.)

### Data-freshness on the AOI page

#### Cron route changes (`app/api/aoi/poll/route.ts`)

For each per-bucket child run:

1. **At start** of the bucket attempt: INSERT `job_runs` row with
   `status='running'`, `outcome=NULL`, `retry_pending=false`,
   `started_at=now()`, `bucket=<bucket>`. (The route already does the
   INSERT pattern; Stage 8 just adds the new fields.)
2. **At completion**, UPDATE the same row with `finished_at=now()`,
   `status=...` (existing), and one of:
   - `outcome='success'` on a clean fetch + match.
   - `outcome='rate_limited'`, `retry_pending=true` on FIRMS HTTP 429.
   - `outcome='network_error'`, `retry_pending=true` on fetch reject /
     non-2xx.
   - `outcome='timeout'`, `retry_pending=true` on `AbortError`.
   - `outcome='partial'` on success-with-some-AOIs-failed (matches existing
     `status='partial'` semantics).
3. The pure-TS mapping function `firmsResultToOutcome(result)` lives in
   `lib/firms/freshness.ts` and is unit-tested.

**`retry_pending` is a *signal*, not a *promise*.** v1 does not implement
intra-tick retry; the next 15-min cron is the retry. The flag tells the UI
to render "(retrying)" instead of "(failed)"; if the user sees the same
flag two ticks in a row they at least know we know.

#### Freshness query (`lib/db/freshness.ts`)

```ts
export type AoiFreshness = {
  bucket: string;
  lastPolledAt: Date | null;
  outcome: "success" | "rate_limited" | "network_error" | "timeout" | "partial" | null;
  retryPending: boolean;
  isStale: boolean;        // computed: lastPolledAt < now - 30min AND outcome === 'success'
};

export async function getAoiFreshness(
  db: AppDb,
  args: { aoiId: string; userId: string; now?: Date }
): Promise<AoiFreshness | null>;
```

- Joins `aois` → `job_runs` on bucket. Picks the most recent `job_runs` row
  for the AOI's bucket where `status != 'running'` (we want completed
  attempts; in-flight runs aren't yet a freshness signal).
- `null` return when no completed runs exist (brand-new AOI, first tick
  pending).
- Uses the new `job_runs_bucket_started_at_idx` index for an O(1) lookup.
- Two-backend discipline: pure SQL, runs identically on Neon and PGlite.

#### AOI page surface (`app/dashboard/aoi/[id]/page.tsx`)

A new section above "Recent briefs" (after the metadata `<dl>`):

```
┌────────────────────────────────────────────────────────┐
│ Last polled: 47 minutes ago                            │
│ [yellow banner if degraded:                            │
│   "Last attempt: rate-limited — retrying next tick"   │
│   "Last attempt: network error — retrying next tick"  │
│   "Polling delayed — last successful tick over 30 min  │
│    ago"                                                │
│   "First poll pending — usually within 15 minutes"]    │
└────────────────────────────────────────────────────────┘
```

Implementation:
- New server-component `<FreshnessBanner aoiId userId />` rendered between
  the metadata `<dl>` and `<RulesForm>`.
- Reads via `getAoiFreshness`. Pure server-side — no client JS.
- Renders relative time using a small in-repo `formatRelative(now, then)`
  helper (`lib/ui/relative-time.ts`); avoid pulling `date-fns` for one
  formatter.
- Color: the existing CSS variables (`--muted` for normal, an
  app-defined `--warn` for degraded). If `--warn` doesn't exist yet,
  add `--warn: oklch(0.85 0.13 80)` (warm yellow) to `app/globals.css`
  alongside the existing palette.
- Honest about its own honesty: when `lastPolledAt` is null (no completed
  run), show "First poll pending — usually within 15 minutes" rather than
  silently rendering nothing.

**Copy is launch-facing and matters.** Use these exact strings unless the
dev agent has a specific reason to change them; report deviations in the
research log.

#### Inject points

- `getAoiFreshness` accepts `now?: Date` for deterministic testing.
- `<FreshnessBanner>` accepts a `__testNow?: Date` prop (server component
  prop, not exposed in production callers); the page-level caller never
  passes it, tests do.

### Tests

Mirroring Stages 3–6:

1. **PGlite unit tests:**
   - `tests/ai/authority/fetch.test.ts` — happy path with stubbed
     `fetchImpl` returning a small FeatureCollection; radius filter (one
     feature inside, one outside, only the inside one wins); most-recent
     selection (two inside, picks newer `postedTs`); 404 → null; timeout →
     null; malformed JSON → null; bucket not in catalog → null without
     making a network call (assert `fetchImpl` not called).
   - `tests/ai/authority/tool.test.ts` — tool definition's `execute`
     returns the `{ source, posted_ts, contains_detection }` shape on hit
     and the all-null shape on miss; never throws. (Future-proofs Path B.)
   - `tests/ai/generate.authority.test.ts` — orchestrator integration:
     stub gateway with a Brief that uses the perimeter values; assert the
     persisted `aoi_briefs.payload.context.authority_perimeter` carries the
     fetched values (not nulls). Counterpart: stub `fetchPerimeter` to
     reject → assert brief still generates with all-null perimeter.
   - `tests/firms/freshness.test.ts` — `firmsResultToOutcome` mapping for
     each FIRMS error category; happy path → 'success'.
   - `tests/db/freshness.test.ts` — `getAoiFreshness` returns the most
     recent completed run for the AOI's bucket; ignores rows with
     `status='running'`; computes `isStale` correctly (boundary at
     exactly 30 min); cross-AOI isolation (AOI A's freshness doesn't leak
     B's bucket).
   - `tests/ui/freshness-banner.test.tsx` — RTL render: happy path
     (no banner), each degraded variant (correct copy + `--warn` styling),
     `lastPolledAt: null` ("First poll pending"), boundary at 30 min.

2. **`@testcontainers/postgresql` integration test:**
   - `tests/aoi-poll.freshness.integration.test.ts` — run the cron route
     end-to-end with a stubbed FIRMS client that returns `429` for one
     bucket and `200` for another. Assert the two `job_runs` rows have the
     correct `outcome` / `retry_pending` values. Then call `getAoiFreshness`
     for AOIs in each bucket and assert the right banner state.

3. **Live test gate:** add `tests/ai/authority/fetch.live.test.ts`,
   gated on `AUTHORITY_PERIMETER_LIVE=1`, that hits each confirmed source
   with a known-fire-region lat/lon and asserts a non-null result. Skipped
   in CI by default; run manually before merge to catch endpoint drift.
   (Mirrors the `FIRMS_LIVE` gate pattern from Stage 2 if it exists; if
   not, document the gate inline.)

Default `pnpm test` runs everything except live. No new third-party env vars
required for the test suite.

## Out of scope for Stage 8 (do NOT build)

- **Slack / Discord webhook delivery.** Still Stage 6's deferred item; not
  bundled here.
- **Per-user rate limits on briefs.** SPEC §API surface mentions; v1 launch
  traffic is too low to justify yet.
- **BYO Gemini key UI / decryption.** SPEC US-7; v1.1.
- **MCP / agent-consumable API.** Spec US-8 / candidate E; ships ~2–3 weeks
  after A' v1 per ADR 0004.
- **Authority sources beyond NIFC / ICNF / CWFIS.** Australia (NAFI),
  Brazil (Programa Queimadas / INPE), Greece (Civil Protection / 112) —
  add in v1.1 once v1 has confirmed-working endpoints to compare against.
- **A weather fetch.** `BriefContext.weather` stays null this stage; the
  product review noted weather is a separate gap. Keep scope tight.
- **Path B (true two-step LLM tool-call).** Documented as a v1.1
  follow-up; Path A ships in Stage 8 (see §Authority-perimeter fetch).
- **Retry-with-backoff inside a single cron tick.** `retry_pending` is a
  user-visible signal only; the next tick is the retry.
- **Map / draw / snooze-link / brief-feedback / detection-prune.** All
  separate product-review recommendations (#1, #2, #6, #8); each gets its
  own brief if Vanyo prioritizes. Not in Stage 8.
- **Schema migration beyond the additive `job_runs` columns + index.**
  No new tables, no columns dropped, no `aoi_briefs` motion. The persisted
  authority perimeter lives inside the existing `payload` JSONB.

Do NOT touch `db/schema/postgis.ts`, the `industrial_mask_static` seed, the
matcher, the dispatcher, the Clerk webhook, or any export route.

## Build-without-blocking discipline (per ADR 0006)

- **No new env vars required for production functioning.** Authority sources
  are public, key-free GeoJSON. Confirmed list below — surface a blocker if
  any turns out to require a key.
- `AUTHORITY_PERIMETER_LIVE=1` gates the optional live test only; CI default
  is unset.
- If the AI Gateway is unconfigured, the brief skips with `config_missing`
  exactly as today — Stage 8 does not change that path.
- If an authority endpoint is down at brief-generation time, the field is
  null and the brief still ships. No throw, no skip.
- The freshness banner renders sensibly for AOIs that pre-date Stage 8's
  schema change: rows without `outcome` (NULL) are treated as `null` outcome
  → "First poll pending" banner until the next tick records a real outcome.

## Open questions to resolve during build (must answer before merge)

1. **NIFC public GeoJSON endpoint.** Likely candidate:
   `https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson`
   (NIFC WFIGS Interagency Perimeters, the National Interagency Fire Center
   ArcGIS service the agency itself publishes). Dev agent must `curl` it as
   the first step of authority work; if it 401s or 403s, surface a blocker
   and fall back to the next-best public source.
2. **ICNF endpoint.** Less obvious; the Portuguese authority (ICNF — Instituto
   da Conservação da Natureza e das Florestas) publishes fire info via the
   ANEPC SGIF (Sistema de Gestão de Informação de Incêndios) but the public
   GeoJSON path is not stable. Dev agent should attempt
   `https://fogos.icnf.pt/` adjacency endpoints; if no key-free GeoJSON exists,
   **surface as a blocker** and ship Stage 8 with NIFC + CWFIS only.
3. **CWFIS endpoint.** Canadian Wildland Fire Information System publishes
   active perimeters as a public WMS / ArcGIS feature service at
   `https://cwfis.cfs.nrcan.gc.ca/` — confirm GeoJSON output is available
   key-free.
4. **AI SDK v6 tool-call pattern with `generateObject`.** Resolved above —
   ship Path A (orchestrator pre-fetch); document Path B as v1.1.
5. **`firms_polls` vs extend `job_runs`.** Resolved above — extend `job_runs`
   (it already has the per-bucket child shape from Stage 2).

## Workflow (per ADR 0006 + ADR 0007)

1. `git checkout -b stage-8-authority-perimeter-and-freshness` off latest
   `master`.
2. Build in commits that group naturally (target: 6–8 commits, ≤500 LOC each):
   - `db/migrations/0006_stage8.sql` + `db/schema/index.ts` extension + a
     migration test
   - `lib/ai/authority/{sources,fetch,tool}.ts` + unit tests
   - `lib/ai/generate.ts` orchestrator wiring + the `gatherAuthorityPerimeter`
     test
   - `lib/firms/freshness.ts` (mapping fn) + `app/api/aoi/poll/route.ts`
     instrumentation + the integration test
   - `lib/db/freshness.ts` + `lib/ui/relative-time.ts` + unit tests
   - `app/dashboard/aoi/[id]/page.tsx` `<FreshnessBanner>` + RTL test +
     `app/globals.css` `--warn` token
3. **Before merge, run the live test once locally:**
   `AUTHORITY_PERIMETER_LIVE=1 pnpm test tests/ai/authority/fetch.live.test.ts`.
   If any source fails, surface as a blocker. Do not silently disable.
4. `pnpm typecheck && pnpm lint && pnpm test && pnpm build` — all green.
5. `git push origin stage-8-authority-perimeter-and-freshness`.
6. Report back to PM_CLAUDE with: branch SHA, all check statuses, draft PR
   description in markdown, the live test outcome.
7. PM_CLAUDE opens the PR via
   `gh pr create --draft --base master --label stage-pr:8`.
8. **Auto-merge applies.** Per ADR 0007, once CI is green and the reviewer
   subagent LGTMs, the PR auto-merges. Do not instruct any reviewer to
   "wait for Vanyo".

## Output

1. Branch on origin: `stage-8-authority-perimeter-and-freshness`
2. Draft PR description (sections: Summary, What changed, How to test,
   Build-without-blocking notes, Live-endpoint confirmation matrix, Things
   to challenge in review, Linked: brief 22 / ADR 0006 / ADR 0007 / spec
   §LLM brief format / spec §Open questions Q3 / pivot-arch §3 / product
   review 2026-05-07 §5 #5 + #7).
3. `pm/research-log/2026-05-XX-stage8-authority-perimeter-and-freshness.md` —
   what shipped, deferrals (Path B, weather, ICNF if blocked), live-endpoint
   confirmation matrix, open questions for PM, any deviations from this
   brief.

## Time budget

~3.5 hours (Stage 8 is smaller than Stage 6 but with one external-API risk
in the authority fetch). If you hit a 20-minute block on any single error,
stop and report. Sharp edges:

- **Authority endpoints are not under our control.** The single biggest
  Stage 8 risk is one of the three sources requiring auth, redirecting,
  paginating, or returning a non-FeatureCollection. The live test is the
  forcing function. If two of three sources work, Stage 8 still ships
  (the third is filed as a blocker for v1.1).
- **AI SDK v6 + `generateObject` + tools.** Settled to Path A above; if
  you discover Path B is cheap (one call via `prepareStep`), prefer it,
  but do not invent a pattern that isn't documented.
- **PGlite + the new `WHERE bucket IS NOT NULL` partial index.** PGlite
  supports partial indexes; verify in the migration test. If it doesn't,
  drop the partial clause (the index is small enough that a full-table
  variant is fine for v1).

## Branch + label

- Branch: `stage-8-authority-perimeter-and-freshness`
- PR base: `master`
- Label: `stage-pr:8` (PM_CLAUDE applies; informational under ADR 0007)
