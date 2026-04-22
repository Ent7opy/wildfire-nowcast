# Brief 15 — Stage 1: AOI CRUD + Neon schema

## Why this exists

Stage 1 of the A' pivot. First stage that adds real backend code. Lands the data model + the CRUD surface that every later stage depends on.

**Read in order:**
1. `pm/PM_CLAUDE.md`
2. `pm/decisions/0005-problem-chosen-a-prime.md`
3. `pm/decisions/0006-stage-pr-workflow.md` — this brief follows the per-stage PR workflow
4. `pm/blockers.md` — note the "build-without-blocking" pattern; do NOT wait on Vanyo for Neon
5. `docs/SPEC-A-prime-v1.md` — §Data model, §API surface (these are binding)
6. `docs/pivot-architecture.md` — §3 collapsed data model (binding SQL schema)

## Goal

Land the AOI domain — schema, migrations, CRUD routes, unit tests — on a `stage-1-aoi-crud` branch off `pivot/a-prime`. Open a draft PR back to `pivot/a-prime`. Build / lint / typecheck must be green locally before pushing.

## Scope (strict)

**Build:**
- Add Drizzle ORM (TypeScript-native, free-tier-friendly, no codegen step). Use `drizzle-kit` for migrations.
- Translate the SQL schema in `docs/pivot-architecture.md` §3 into Drizzle schema definitions (`db/schema/`). For Stage 1 the in-scope tables are: `users`, `aois`, `aoi_rules`. Defer `firms_detections`, `aoi_events`, `aoi_briefs`, `notifications_log` to later stages — but **leave a comment in the schema file** noting the planned tables and which stage adds them.
- Drizzle migrations under `db/migrations/`. Generate the initial migration via `pnpm drizzle-kit generate`.
- PostGIS support: Neon supports PostGIS. The `polygon`, `bbox`, `centroid` columns must be PostGIS `geometry` types. Drizzle has community PostGIS support — use the `drizzle-orm/pg-core` `customType` pattern if needed.
- Single-user stub for Stage 1: every API call is treated as `STUB_USER_ID = "stub-user-1"`. The `users` table gets one row at migration time. Auth (Clerk) lands in Stage 5.
- API routes per `docs/SPEC-A-prime-v1.md` §API surface — Stage 1 implements:
  - `POST   /api/aoi`              — create AOI (accepts GeoJSON polygon in body)
  - `GET    /api/aoi`              — list current user's AOIs
  - `GET    /api/aoi/:id`          — read one AOI (with rules)
  - `PATCH  /api/aoi/:id`          — update name / polygon
  - `DELETE /api/aoi/:id`          — soft-delete (set `archived_at`)
  - `PUT    /api/aoi/:id/rules`    — upsert rules (distance threshold, quiet hours, channels)
- Validation: Zod schemas for every request body / response payload. Co-located with the route handler.
- Error handling: typed error responses (`{ error: { code, message } }`). No raw 500s leaking.
- Unit tests for the route logic using PGlite (in-memory PostgreSQL) so tests run anywhere without a real DB. Use Vitest (already standard in this kind of stack).
- Integration test: a single end-to-end test that creates → reads → updates → deletes an AOI through the route handlers.

**Do NOT:**
- Touch any files in `api/`, `ml/`, `ingest/`, `ui/`, `models/`, `configs/`, `Makefile`, `docker-compose.yml`, `railway.toml`, `Dockerfile*`
- Add UI pages yet (Stage 5 owns the UI)
- Add the FIRMS poll, the LLM brief, or notifications (Stages 2/3/4)
- Wire Clerk or any auth (Stage 5)
- Open a non-draft PR (PM_CLAUDE opens the PR; you push the branch and report)

## Build-without-blocking discipline

Vanyo has not yet provisioned Neon. Your code MUST:
- Build green without `DATABASE_URL` set
- Run all unit + integration tests against PGlite in-memory (no real DB needed)
- At runtime, when `DATABASE_URL` IS set, route handlers connect to Neon and operate normally
- When `DATABASE_URL` is NOT set, route handlers return a typed `503 service_unavailable` error with a helpful message (`"DATABASE_URL not configured; this is expected during pre-Neon-setup development"`). This makes the Vercel preview deploy fail gracefully rather than crash.

## Framework guidance (binding per Vercel plugin context)

- Next.js 16 App Router. API routes in `app/api/.../route.ts`.
- Server Components by default; this brief has no UI so no `'use client'` needed.
- **Do NOT use `@vercel/postgres`** — sunset. Use `drizzle-orm` against the standard `pg` / `postgres` driver pointed at Neon's pooled connection string.
- Node.js 24 runtime (default).
- pnpm, not npm.

## Workflow (per ADR 0006)

1. Branch from `pivot/a-prime`: `git checkout -b stage-1-aoi-crud`
2. Build the work in commits that are individually meaningful (≤300 lines per commit ideally; not strict)
3. `pnpm install`, `pnpm exec tsc --noEmit`, `pnpm exec eslint .`, `pnpm test`, `pnpm build` — all green locally before pushing
4. `git push origin stage-1-aoi-crud`
5. **Do NOT open the PR yourself.** Report back to PM_CLAUDE with: branch SHA, all check statuses, a draft PR description in markdown (sections: Summary, What changed, How to test, Linked: brief 15 / ADR 0005 / spec §API surface)
6. PM_CLAUDE opens the PR via `gh pr create` and watches CI

## Output

1. **Branch on origin:** `stage-1-aoi-crud` with as many commits as natural
2. **Draft PR description in markdown** in your reply
3. **Write `pm/research-log/2026-04-21-stage1-aoi-crud.md`** (≤700 words): what shipped, what was deferred, any deviations from the spec / brief, test coverage notes, anything PM_CLAUDE should challenge in the diff

## Time budget

~90 min. If you hit a 15-min block on any single error, stop and report.
