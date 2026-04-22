# 2026-04-22 — Stage 2: FIRMS poll, AOI matcher, GH Actions cron (dev agent log)

**Brief:** `pm/briefs/16-stage2-firms-poll-and-cron.md`
**Branch:** `stage-2-firms-cron` (off `pivot/a-prime`)
**Status:** Local typecheck/lint/test/build all green. 69/69 tests pass, including 7 PostGIS integration tests against `imresamu/postgis:16-3.5-alpine` via @testcontainers/postgresql. PR not opened (PM_CLAUDE owns that step per ADR 0006).

## What shipped

1. **Schema additions (`db/schema/index.ts`)** — `firms_detections`, `aoi_events`, `industrial_mask_static`, `job_runs`. Drizzle TS surface is the friendly view; the canonical PostGIS DDL lives in the SQL migration alongside.
2. **SQL migrations** — `db/migrations/0001_stage2.sql` (Neon: PostGIS geometry types, GIST indexes, dedupe unique-indexes) and `0001_stage2.test.sql` (PGlite: GeoJSON-as-TEXT). Both idempotent (`IF NOT EXISTS` everywhere).
3. **Migration runner** — `scripts/db-migrate.ts` now applies every `*.sql` file in lexical order, skipping `*.test.sql` variants. PGlite test fixture (`db/test/pglite.ts`) layers Stage 1 + Stage 2 test SQL.
4. **FIRMS client (`lib/firms/client.ts`)** — Single `fetchAreaCsv` function. Lazy `FIRMS_MAP_KEY` read (returns typed `config_missing` when unset). Exponential backoff with jitter on 429/5xx (3 attempts). In-process token bucket: 6 req/min cap.
5. **CSV parser** — Handles VIIRS canonical headers and remaps MODIS `brightness`/`bright_t31` into the same canonical fields. `emptyArea: true` for the no-data sentinel and header-only responses.
6. **Bucket coalescing (`lib/firms/buckets.ts`)** — `getActiveBuckets(db)` aggregates non-archived AOIs by `region_bucket`; `bucketToBbox(key)` is the round-trip inverse of Stage 1's `regionBucketFromLonLat`. Round-trip tested on six places (Athens, Lisbon, SF, Darwin, Cape Town, Longyearbyen).
7. **Dedupe hash (`lib/firms/dedupe.ts`)** — sha256 over (aoi_id, bucket, rounded centroid 0.01°, source, UTC-day-index), 32 hex chars. Deterministic, sensitive to all five inputs.
8. **Matcher (`lib/firms/matcher.ts`)** — Two paths gated on `db.usePostGIS`. Production: `INSERT ... ON CONFLICT DO NOTHING` for detections, ST_Intersects against the static mask sets `is_industrial_static` inline, `ST_DWithin(...::geography, ::geography, distance_buffer_km*1000)` finds matches. Confidence gate handles VIIRS l/n/h and MODIS 0–100. Per-poll match window uses `db's now()` for clock-skew safety. UPSERT path for re-detections in the same dedupe window; INSERT for new events.
9. **`/api/aoi/poll` route (`app/api/aoi/poll/route.ts`)** — POST with `Authorization: Bearer <CRON_SECRET>` + constant-time compare. Optional `{bucket}` body to scope. 503 when CRON_SECRET / FIRMS_MAP_KEY / DATABASE_URL unset. `maxDuration = 60`, `runtime = "nodejs"`. Records parent + per-bucket `job_runs` rows; one bucket failing → parent run = `partial`, others continue.
10. **GitHub Actions cron (`.github/workflows/firms-poll.yml`)** — `workflow_dispatch` only at merge time; `schedule:` block commented out. Vanyo flips by un-commenting two lines after CRON_SECRET wiring. `concurrency: { group: firms-poll, cancel-in-progress: false }`. One curl retry on transient failure; non-retryable on 401/400/503.
11. **Industrial mask seed** — 68 polygons in `db/seeds/industrial-mask-stage2.json` (Persian Gulf, North African, West African, Bakken, Permian, Russian Arctic gas flares; Houston/Baton Rouge/Wilmington refineries; Asian steel mills; ten persistently active volcanoes). Sources cited in JSON header (Elvidge 2016, World Bank GGFR, Smithsonian GVP, EOSDIS NRT FAQ). `pointBoxToPolygon` does an equirectangular box buffer; pole-safe.
12. **Seed script + `pnpm db:seed:industrial`** — `scripts/db-seed-industrial-mask.ts` is idempotent (unique on `(kind, name)`).
13. **PostGIS testcontainer harness (`db/test/testcontainer.ts`)** — Lazy-imports `@testcontainers/postgresql`. Probe `dockerAvailable()` for daemon reachability; `tryStartPostgisContainer()` swallows image-pull failures and lets tests skip cleanly. Defaults to `imresamu/postgis:16-3.5-alpine` (multi-arch); override with `WFN_POSTGIS_IMAGE`.
14. **Tests** — 69 total, all green:
    | File | Tests | Notes |
    |---|---|---|
    | `tests/firms-client.test.ts` | 11 | parser, retries, throttle, config-missing, MODIS column mapping |
    | `tests/firms-buckets.test.ts` | 7 | round-trip 6 places + malformed-key handling |
    | `tests/firms-dedupe.test.ts` | 5 | determinism, day-window roll, sub-bin nudge, sensitivity |
    | `tests/industrial-seed.test.ts` | 3 | shape validation, centroid-in-polygon, polar guard |
    | `tests/poll-route-auth.test.ts` | 6 | env gates, bearer auth, 401/400/503 paths, empty-bucket happy path |
    | `tests/firms-matcher.integration.test.ts` | 5 | spatial: in-buffer hit, out-of-buffer miss, industrial mask suppression, idempotent re-poll, in-window event extension |
    | `tests/poll-route.integration.test.ts` | 2 | end-to-end happy path + partial-run on FIRMS error |
    | (Stage 1 carried over) | 30 | unchanged |
15. **Route injection point** — `_setTestFirmsFetch` lets the integration suite stub the FIRMS client without hitting the live NASA endpoint. Production leaves it null. Documented inline.

## Deferred (per brief)

- LLM brief generation → Stage 3
- Notifications / Resend → Stage 4
- UI for AOI events → Stage 5
- Clerk auth → Stage 5 (still using `STUB_USER_ID` everywhere)
- Full NASA STA mask ingest → future stage (the static seed is the v1 stand-in)
- TTL prune cron for `firms_detections` rows older than 14 days → was implied by spec §3.4 ("rows older than 14 days are pruned by the cron") but not in the brief's strict scope. Calling it out as Open Question 2 below; current matcher `inserted_at >= pollStart` keeps the table from being a correctness liability, but storage will grow until the prune lands.

## Deviations from the brief

1. **Match-time scope: `inserted_at >= pollStart` (DB clock), not `detected_at >= now() - 24h`.** Brief said "from this poll" implicitly; my first cut filtered on `detected_at >= now() - 24h`. That broke idempotence (a re-poll re-evaluates the same row and re-extends the event). Switched to a poll-start timestamp captured from the DB's `now()` at the start of `matchDetectionsToAois`. ON CONFLICT DO NOTHING then makes a re-poll a true no-op. Documented inline.
2. **Testcontainer image: `imresamu/postgis:16-3.5-alpine` instead of upstream `postgis/postgis:16-3.5`.** Upstream is amd64-only; the alpine fork is multi-arch. CI Ubuntu runners on amd64 are unaffected; this matters only for arm64 dev boxes (Vanyo's Mac). Override via `WFN_POSTGIS_IMAGE` env. Added `tryStartPostgisContainer` so a future image-pull failure skips integration tests cleanly with a console warning rather than failing CI.
3. **Two integration test files instead of the brief's three.** Brief listed `firms-client.test.ts`, `firms-matcher.integration.test.ts`, `poll-route.integration.test.ts` — that's what shipped; I additionally added `firms-buckets.test.ts`, `firms-dedupe.test.ts`, `industrial-seed.test.ts`, `poll-route-auth.test.ts` to keep the unit shape per behavior small and parallelisable.
4. **No explicit `vitest.config.ts` projects split for integration vs. unit.** The integration tests gate themselves at the `describe.skip` level via `dockerAvailable()`, and individually skip via `ctx.skip()` if `tryStartPostgisContainer()` returns null. Default `pnpm test` runs the full set; absence of Docker degrades gracefully with visible console warnings rather than red CI. Acceptable for Stage 2; can split if Stage 6+ adds more slow tests.
5. **Vercel Cron suggestion declined.** The Vercel deployment-skill's auto-validation suggested using `vercel.json` crons instead of GitHub Actions. Architecture says no (Hobby tier 2-cron cap; Vercel Cron is the documented R2 hot-backup, not primary). Held the line.

## Test coverage

| Area | Coverage |
|---|---|
| FIRMS client URL building, parsing, backoff, throttle | unit (mocked fetch) |
| Dedupe hash determinism + sensitivity to all five inputs | unit |
| Bucket round-trips for 6 globally distributed places | unit |
| Industrial seed JSON shape + polygon containment | unit |
| Route auth: missing/invalid/expired bearer, env-var gates | unit (PGlite) |
| Spatial matcher: in-buffer hit, out-of-buffer miss | integration (PostGIS) |
| Industrial mask suppression at insert time | integration |
| Dedupe idempotence on identical re-poll | integration |
| Event-extension on same-window re-detection (count++, peak FRP) | integration |
| End-to-end POST → events written → job_runs closed ok | integration |
| Partial-run bookkeeping when one bucket FIRMS-fails | integration |

What's NOT covered (acceptable per brief):
- Live FIRMS API integration (forbidden — would burn rate limit and require key in tests)
- Multi-bucket fan-out under FIRMS rate-limiting (synthetic; would require throttle harness)
- TTL prune (function not yet built; see Open Q 2)
- Vercel-specific Fluid Compute warm-isolate behaviour for the token bucket

## Things to challenge in review

1. **Match-time window: `inserted_at >= pollStart` vs. `detected_at >= ...`.** Decision rationale above. The alternative (track inserted IDs through `RETURNING id` and pass them into the match query) is more explicit but harder to read in SQL. Open to switching if review prefers the explicit ID path.
2. **Industrial mask is a 68-polygon stand-in.** Spec calls for the full NASA STA mask layer; that's a future job. Risk: a real wildfire near (e.g.) the Bakken or near a flare at a not-listed location could be over-suppressed. Mitigation: the seed kinds (`gas_flare` / `refinery` / `industrial` / `volcanic`) are stored on every row so a future Stage 6 industrial-mask QA UI can show suppressed detections. Sources cited inline.
3. **Token bucket lives at module scope.** Inside one Vercel warm isolate it's meaningful; cold start / multi-region fan-out resets it. We compensate by also gating per-bucket FIRMS calls (one per bucket per cron tick = max ~50/tick at 100 AOIs spread across 50 tiles, well under FIRMS's 5k/10min cap).
4. **`_setTestFirmsFetch` injection point.** Module-scope let to keep tests honest without pulling in a DI framework. `_setTestDb` follows the same pattern from Stage 1. If review wants the production code to *not* know about the test hook, the alternative is to refactor the route to take its FIRMS fetcher via a factory — a heavier rewrite that buys little.
5. **Cron workflow ships disabled.** `schedule:` is a single-line uncomment for Vanyo. Documented in the workflow file header; separate reminder in `pm/blockers.md` Stage 2 entries.

## Open questions for PM

1. **TTL prune timing.** Spec calls for `firms_detections` rows older than 14 days to be pruned. Brief did not include the prune in scope. Options: (a) ship a cron entrypoint `/api/aoi/prune` next stage; (b) inline a `DELETE FROM firms_detections WHERE inserted_at < now() - INTERVAL '14 days'` at the end of every `/api/aoi/poll` (cheap, but overhead per tick); (c) Neon's free tier is generous enough that we live with growth through Stage 5 and add prune in Stage 6. Default proposal: (c). Awaiting decision.
2. **Vercel Cron hot backup (R2 risk register item).** When should we wire the Hobby-tier free Vercel Cron job at `/api/cron/heartbeat` as a redundancy for GH Actions outage? Probably Stage 4 or 5 once notifications can detect "no successful poll in 30 min" via job_runs. Worth confirming.
3. **`MAX_FIRMS_DAY_RANGE` env var.** Currently hardcoded to 1 day per poll (matches "every 15 min" cadence). When we need to backfill (e.g. after Vanyo flips the cron toggle and we want to seed history), the brief did not call for an env. Should `dayRange` become an env-tunable, or do we add a separate `/api/aoi/backfill` route in a later stage?

## Local verification (run by the agent)

```
pnpm install                     # added @testcontainers/postgresql + testcontainers
pnpm typecheck                   # clean
pnpm lint                        # clean
pnpm test                        # 69/69 green (PGlite + PostGIS testcontainer)
pnpm build                       # 6 routes registered (4 from Stage 1 + /api/aoi/poll)
```

Docker Desktop was running locally; the postgis-alpine image pulled in ~30 s on the first run. CI Ubuntu runner has Docker pre-installed and runs the full integration suite by default.

## Time

~2.5 h wall-clock (within the 3 h budget).
