# Brief 16 — Stage 2: FIRMS poll, AOI matcher, GitHub Actions cron

## Why this exists

Stage 2 of the A' pivot. Lands the data flow that turns FIRMS detections into AOI events and a brief-worthy event queue. This is the first stage with a live external dependency (NASA FIRMS) and the first cron-driven pipeline.

**Read in order:**
1. `pm/PM_CLAUDE.md`
2. `pm/decisions/0005-problem-chosen-a-prime.md`
3. `pm/decisions/0006-stage-pr-workflow.md` — workflow you must follow (branch off `pivot/a-prime`, draft PR, PM_CLAUDE opens it)
4. `pm/blockers.md` — Stage 2 entries: Docker for testcontainers, CRON_SECRET. Build-without-blocking applies.
5. `docs/SPEC-A-prime-v1.md` — §Data model + §API surface (binding for the new endpoints)
6. `docs/pivot-architecture.md` — §3 (data model includes the Stage 2 tables you must add) + the architecture diagram in the Amendment / §1
7. `pm/research-log/2026-04-21-stage1-aoi-crud.md` + `db/schema/index.ts` — to understand the existing two-backend (Neon+PostGIS / PGlite) repository pattern

## Goal

Land — on a `stage-2-firms-cron` branch off `pivot/a-prime` — the schema tables, FIRMS client, bucket coalescing, AOI matcher, `/api/aoi/poll` endpoint, GitHub Actions cron workflow, and the integration tests against a real PostGIS testcontainer. PR draft markdown ready; PM_CLAUDE opens the PR.

## Scope (strict)

### Schema additions (new migration `db/migrations/0001_stage2.sql` + Drizzle schema in `db/schema/`)

Per `docs/pivot-architecture.md` §3, add the following tables (deferred from Stage 1):

- **`firms_detections`** — short-TTL (~14 days) cache of raw FIRMS pixels relevant to active AOIs. Columns per spec: `id`, `source` (e.g. `VIIRS_NOAA20_NRT` / `MODIS_NRT`), `detected_at`, `geom (Point, 4326)`, `frp_mw`, `confidence`, `daynight`, `acq_date`, `acq_time`, `bright_ti4`, `bright_ti5`, `scan`, `track`, `version`, `is_industrial_static` (boolean, set at insert via NASA STA mask lookup if available, else null), `bucket`, `inserted_at`. Unique constraint on `(source, acq_date, acq_time, geom)` for idempotent dedupe. GIST index on `geom`. B-tree on `(bucket, detected_at)`.
- **`aoi_events`** — per-AOI matched events with dedupe key. Columns: `id`, `aoi_id` (FK), `first_seen_at`, `last_seen_at`, `nearest_distance_km`, `detection_count`, `peak_frp_mw`, `dedupe_hash` (24h content hash over (aoi_id, bucket, rounded coords, source) — used to gate duplicate brief generation), `status` (`'new' | 'open' | 'closed'`), `closed_at`. Unique on `(aoi_id, dedupe_hash)`.
- **`industrial_mask_static`** — small static GeoJSON catalog from NASA Static Thermal Anomalies (STA) Mask. Columns: `id`, `kind` (e.g. `gas_flare` / `industrial` / `volcanic`), `name`, `geom (Polygon, 4326)`, `source_url`, `loaded_at`. Seed with a small fixture file at first (~50 well-known polygons covering global gas-flare hotspots) — full STA ingest is a future job.
- **`job_runs`** — observability for the cron. Columns: `id`, `job_name` (e.g. `'firms-poll'`), `bucket` (nullable), `started_at`, `finished_at`, `status` (`'ok' | 'partial' | 'error'`), `firms_request_count`, `detections_inserted`, `events_created`, `error` (text, nullable). Indexed on `(job_name, started_at desc)` for "show me the last N runs" UIs in later stages.

Hand-author SQL (same pattern as Stage 1 — `0001_stage2.sql` for Neon, `0001_stage2.test.sql` for PGlite). PostGIS-specific bits (geometry types, GIST) live in the SQL; Drizzle TS gets the same columns minus the spatial DSL gaps.

### TypeScript FIRMS client (`lib/firms/client.ts`)

- Single function: `fetchAreaCsv({ source, bbox, dayRange })` returning a typed array of detections.
- Source enum: `VIIRS_NOAA20_NRT`, `VIIRS_SNPP_NRT`, `MODIS_NRT`. Default for v1: `VIIRS_NOAA20_NRT` (best resolution, most current).
- Builds the URL: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/<KEY>/<SOURCE>/<bbox>/<dayRange>`.
- Reads `FIRMS_MAP_KEY` from `process.env` lazily — function returns a typed `FirmsConfigError` if missing (does not throw at import time; build-without-blocking).
- Parses the CSV (rows: `latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight`).
- Implements simple exponential backoff on 5xx + 429 (3 attempts, jitter, max 4s).
- No request more often than 6/min globally — guard with a tiny in-process token bucket (since Vercel functions reuse instances under Fluid Compute, this is meaningful).
- Pure function, no DB writes. The matcher is responsible for persistence.

### Bucket coalescing (`lib/firms/buckets.ts`)

- `getActiveBuckets(db)` → `string[]` — distinct `region_bucket` values from non-archived AOIs, ordered by AOI count desc (so heaviest buckets get fresh data first if we ever rate-limit).
- `bucketToBbox(bucket)` → `[minLon, minLat, maxLon, maxLat]` — inverse of `lib/geo/region-bucket.ts` from Stage 1. Use the same `5°×5°` grid; produce the bbox string FIRMS wants (`minLon,minLat,maxLon,maxLat`).
- Sanity test: round-trip a few AOI centroids (e.g. Athens, Lisbon, San Francisco) through `regionBucket → bbox → contains(centroid)`.

### Detection → AOI matcher (`lib/firms/matcher.ts`)

- `matchDetectionsToAois(db, { bucket, detections })` returns `{ insertedDetections, eventsAffected }`.
- For each detection in the bucket: insert into `firms_detections` (idempotent via the unique constraint — use `INSERT ... ON CONFLICT DO NOTHING`).
- Compute `is_industrial_static` at insert time via a `ST_Intersects` against `industrial_mask_static` (PostGIS path) or a turf.js point-in-polygon (PGlite path). Cache the mask in-process for the duration of one poll.
- For each AOI in the bucket, query `ST_DWithin(aoi.polygon::geography, det.geom::geography, aoi_rules.distance_buffer_km * 1000)` joined to detections from this poll. Roll matches up into one row per (aoi_id, dedupe_hash):
  - `dedupe_hash = sha256(aoi_id + bucket + floor(centroid_lat*100)/100 + floor(centroid_lon*100)/100 + 24h_window_index)` — same window of activity = same event.
  - Existing row in `aoi_events` with same hash + `status != 'closed'` → UPSERT to extend `last_seen_at`, bump `detection_count`, update `peak_frp_mw`. **No new brief queued.**
  - No existing row → insert with `status = 'new'`. **This is what Stage 3's brief generator picks up.**
- Skip industrial detections (`is_industrial_static = true`) entirely from the matcher — they don't trigger events.
- Skip detections with `confidence` below the AOI's `aoi_rules.min_confidence`.

### `/api/aoi/poll` route (`app/api/aoi/poll/route.ts`)

- POST endpoint. Body: optional `{ bucket: string }` to poll one specific bucket; if absent, polls all active buckets.
- Auth: requires `Authorization: Bearer <CRON_SECRET>` header. Reject 401 otherwise. The secret comes from `process.env.CRON_SECRET`.
- Build-without-blocking: if `CRON_SECRET` is unset → 503 with a typed message; if `FIRMS_MAP_KEY` is unset → 503. The endpoint must be safe to deploy before Vanyo wires the secrets.
- Writes a `job_runs` row at start, updates it at finish (status, counts, error).
- Returns `{ runs: [{ bucket, detections, events_created, duration_ms }], total_duration_ms }`.
- 60s function timeout config (Hobby max). Tolerate one bucket failing without aborting the whole poll.

### GitHub Actions cron (`.github/workflows/firms-poll.yml`)

- `schedule: cron: '*/15 * * * *'`
- One job: `curl -X POST -H "Authorization: Bearer $CRON_SECRET" "$VERCEL_URL/api/aoi/poll"` with timeout + retry once on transient failure.
- Use `secrets.CRON_SECRET` and `vars.VERCEL_URL` (or hardcode the wildfire-nowcast.vercel.app URL since that's stable for the production deploy — branch previews are not cron'd).
- **Initially `on: workflow_dispatch:` only (no schedule)** — the `schedule:` line is commented out. Reason: until Vanyo confirms `CRON_SECRET` is set on both Vercel and GitHub, an active cron will spam 401 errors. Vanyo flips the toggle by un-commenting one line after the secret is wired.
- Workflow must `concurrency: cancel-in-progress: false` and `group: firms-poll` so a slow run doesn't get pre-empted.

### Tests

Two parallel test infrastructures co-existing:

1. **PGlite (existing)** for non-spatial unit tests (FIRMS client parsing, bucket math, dedupe-hash function, route auth-failure paths).
2. **NEW: `@testcontainers/postgresql`** + `postgis/postgis:16-3.5` image for spatial integration tests:
   - `tests/firms-matcher.integration.test.ts` — spin up container once per file, run migrations, seed a few AOIs + an industrial-mask polygon, feed in synthetic FIRMS detections, assert the right events get created with the right `dedupe_hash`, assert dedup on a second poll, assert industrial-mask suppression.
   - `tests/firms-client.test.ts` — pure parser test using a recorded FIRMS CSV fixture in `tests/fixtures/firms-sample.csv`. Do NOT hit FIRMS in tests.
   - `tests/poll-route.integration.test.ts` — end-to-end through the route handler against the testcontainer, with `FIRMS_MAP_KEY` mocked via `process.env` injection + the FIRMS client stubbed via `vi.mock`.

Add `vitest.config.ts` projects or a separate config so the slow integration tests can run conditionally (pass an env var `INTEGRATION=1` to include them; default `pnpm test` includes integration in CI but skips locally if Docker isn't available — use a `beforeAll` Docker-availability check that calls `it.skip` with a clear message).

### Industrial-mask seed

- Small JSON fixture at `db/seeds/industrial-mask-stage2.json` (≤100 polygons covering known global flares — Persian Gulf, Bakken, Niger Delta, etc.). Cite source URLs in a header comment. This is a stand-in for the full NASA STA layer; future stage can do the proper ingest.
- Seed runs as part of the Stage 2 SQL migration (idempotent `INSERT ... ON CONFLICT DO NOTHING`).

## Out of scope for Stage 2 (do NOT build)

- LLM brief generation (Stage 3)
- Notifications / Resend (Stage 4)
- UI for AOI events (Stage 5)
- Auth (Stage 5 — keep `STUB_USER_ID` everywhere)
- Full NASA STA mask ingest job
- Any UI changes — this stage adds zero pages

Do NOT touch existing `api/`, `ml/`, `ingest/`, `ui/`, `models/`, `configs/` directories.

## Build-without-blocking discipline (per ADR 0006)

The PR must be mergeable and the preview must build green even though Vanyo may not yet have:
- Docker running locally (CI is fine; integration tests skip locally with a clear message)
- `CRON_SECRET` on Vercel (route returns 503 with a typed message)
- `FIRMS_MAP_KEY` on Vercel (route returns 503; FIRMS client returns typed error from `fetchAreaCsv`)

The cron workflow ships with `schedule:` commented out so it doesn't fire until Vanyo flips the toggle.

## Workflow

1. `git checkout -b stage-2-firms-cron` off `pivot/a-prime` (latest)
2. Build in commits that group naturally (target: 5–8 commits, ≤500 LOC each)
3. `pnpm install`, `pnpm exec tsc --noEmit`, `pnpm exec eslint .`, `pnpm test` (with Docker running locally if you want integration coverage), `pnpm build` — all green
4. `git push origin stage-2-firms-cron`
5. Report back to PM_CLAUDE with: branch SHA, all check statuses, draft PR description in markdown
6. PM_CLAUDE opens the PR via `gh pr create --draft --base pivot/a-prime`

## Output

1. Branch on origin: `stage-2-firms-cron`
2. Draft PR description in your reply (sections: Summary, What changed, How to test, Build-without-blocking notes, Things to challenge in review, Linked)
3. `pm/research-log/2026-04-22-stage2-firms-cron.md` — what shipped, deferrals, deviations from brief, open questions for PM. (Write this file directly; PM_CLAUDE will commit it if your sandbox blocks the write.)

## Time budget

~3 hours. If you hit a 20-minute block on any single error, stop and report.
