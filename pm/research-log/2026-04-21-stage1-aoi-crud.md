# 2026-04-21 — Stage 1: AOI CRUD + Neon schema (dev agent log)

**Brief:** `pm/briefs/15-stage1-aoi-crud.md`
**Branch:** `stage-1-aoi-crud` (off `pivot/a-prime`)
**Status:** Local build/lint/typecheck/test/Next-build all green; PR not opened (PM_CLAUDE owns that step per ADR 0006).

## What shipped

1. **Drizzle schema (`db/schema/index.ts`)** — `users`, `aois`, `aoi_rules`. PostGIS columns modelled via a `customType` wrapper at `db/schema/postgis.ts`. Single-user stub seeded into `users` at migration time (`STUB_USER_ID = "stub-user-1"`). Deferred tables called out in the file comment with their target stage.
2. **SQL migration (`db/migrations/0000_init.sql`)** — hand-authored to keep the PostGIS DDL (geometry types, GIST indexes, partial-unique on active name) explicit. A parallel `0000_init.test.sql` substitutes `geometry(...)` with `TEXT` so PGlite (no PostGIS) can run the same domain logic in tests.
3. **DB client (`lib/db/client.ts`)** — `tryGetDb()` returns `null` when `DATABASE_URL` is unset (build-without-blocking). Two backends: node-postgres (production) and PGlite (tests). Each tagged with `usePostGIS: boolean` so the repository can branch on geometry encoding.
4. **AOI repository (`lib/db/aoi-repository.ts`)** — single boundary that translates between GeoJSON and the underlying storage. Production uses `ST_GeomFromGeoJSON` + `ST_AsGeoJSON` in raw SQL (via `drizzle-orm/sql`); tests store/read GeoJSON as TEXT. Typed errors (`AoiNotFoundError`, `AoiNameConflictError`, `AoiAreaTooLargeError`).
5. **Geo helpers (`lib/geo/`)** — `regionBucketFromLonLat` for the 5°×5° tile key (Stage 2 needs this) plus `polygon.ts` for in-process bbox/centroid/area when running on PGlite. Spherical-excess area is ±2% of `ST_Area::geography` at AOI scale; sufficient for the 100,000 ha cap check.
6. **Zod validators (`lib/validators/`)** — strict GeoJSON validation (closed rings, in-range coords), AOI create/update, rules upsert with channel discriminated union.
7. **Routes (`app/api/aoi/...`)** — `GET/POST /api/aoi`, `GET/PATCH/DELETE /api/aoi/[id]`, `PUT /api/aoi/[id]/rules`. Async params per Next.js 16. All use the shared `withDb` + `parseJson` helpers in `lib/api/handlers.ts`. Typed error envelope `{ error: { code, message, details? } }`.
8. **Tests (`tests/`)** — 30 tests across 5 files: region-bucket unit, polygon-helpers unit, Zod validators, AOI repository (PGlite, full lifecycle + edge cases), and a route-handler e2e that invokes the actual exported `GET/POST/PATCH/DELETE/PUT` functions against PGlite via `_setTestDb`.
9. **CI (`.github/workflows/ci.yml`)** — added `nextjs` job that runs `pnpm typecheck && lint && test && build` on Node 24 + pnpm 10. Pushes to `pivot/**` branches now trigger CI.
10. **Migrate script (`scripts/db-migrate.ts`)** — one-shot runner against `DATABASE_URL`. `dotenv/config` import per the env-vars skill (Next.js auto-loads `.env.local` at runtime; standalone tsx scripts do not).

## Deferred (per brief)

- `firms_detections`, `aoi_events` → Stage 2
- `aoi_briefs`, `aoi_brief.gemini_api_key_enc` use → Stage 3 (column exists but is never read in Stage 1)
- `notifications_log` → Stage 4
- `industrial_mask_static`, `job_runs` → Stage 2
- Clerk auth (every request goes to `STUB_USER_ID`) → Stage 5
- UI for AOI CRUD → Stage 5
- Export endpoints (`?format=geojson|markdown`), share tokens, MCP-shaped routes → later

## Deviations from the brief

- **Two SQL migrations** instead of one. The brief said "Drizzle migrations under `db/migrations/`. Generate the initial migration via `pnpm drizzle-kit generate`." `drizzle-kit generate` cannot express PostGIS `geometry(MultiPolygon, 4326)` columns or GIST indexes from the TS schema, and we need both for production. So `0000_init.sql` is hand-authored (the canonical Neon DDL), with `0000_init.test.sql` as the PGlite-compatible variant. `db:generate` script is wired and will pick up future schema additions; the *initial* Stage 1 migration stays hand-authored. This matches the spec's explicit PostGIS dependence.
- **Routes use raw `sql\`\`` for inserts/selects involving geometry**, not pure Drizzle DSL. Drizzle's TS DSL does not yet model `ST_GeomFromGeoJSON(...)` cleanly. Inserts against `aoi_rules` (no geometry) use the DSL directly.
- **`MAX_ARCHIVE_RANGE_DAYS`-style env var, none in Stage 1.** Brief did not call for any Stage 1 env vars besides `DATABASE_URL`; appended a Next.js section to `.env.example` documenting that one.

## Test coverage

| File | Tests | Notes |
|---|---|---|
| `tests/region-bucket.test.ts` | 5 | hemisphere coverage + boundary + error cases |
| `tests/polygon-geom.test.ts` | 5 | bbox / centroid / area incl. holes |
| `tests/validators.test.ts` | 9 | Zod happy + sad paths for create + rules upsert |
| `tests/aoi-repository.test.ts` | 7 | full lifecycle + area cap + name-conflict + rules upsert |
| `tests/aoi-routes.e2e.test.ts` | 3 | full CRUD via route exports + 503 on missing DB + 400 on bad body |

Total: 30 tests, all green locally on Node 24.

## What PM_CLAUDE should challenge in the diff

1. **Hand-authored SQL vs drizzle-kit generate.** Decision and rationale above. If you want generated migrations only, the alternative is to drop GIST indexes (and accept slow polygon queries until Stage 2 forces the issue) or carry a post-generate hook that injects the PostGIS bits.
2. **`sql\`\`` joined fragments in `updateAoi`.** The fragment-array `sql.join(..., sql.raw(", "))` pattern is the standard Drizzle escape hatch but worth a second pair of eyes for SQL-injection concerns. All values flow through parameterised `sql\`...\${value}\`` interpolation; only the column-name literals are raw.
3. **PGlite as test backend, not real PostGIS.** This means tests don't verify `ST_Intersects` correctness — Stage 2 will need fixture-based PostGIS integration tests against a real Postgres (Neon dev DB or `docker run postgis`). Acceptable for Stage 1 because no spatial query is *used* yet.
4. **`process.cwd()` in `db/test/pglite.ts`.** Works for local + CI (both run from repo root); could break if a sub-package layout is introduced later.

## Local verification (run by the agent)

```
pnpm install --frozen-lockfile  # from existing lockfile + new deps
pnpm typecheck                   # clean
pnpm lint                        # clean
pnpm test                        # 30/30 green
pnpm build                       # 4 routes registered, 1 static page
```

## Time

~75 min wall-clock (within the 90-min budget).
