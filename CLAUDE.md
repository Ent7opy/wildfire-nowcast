# CLAUDE.md

Guidance for Claude Code working on the Wildfire Nowcast A' pivot.

## North Star (current)

Wildfire Nowcast is a **free, open, AI-native fire intelligence agent for stewardship-motivated users** — conservation trusts, Natura 2000 site managers, Firewise communities, Indigenous fire crews, LTER field scientists, environmental journalists — that watches their specific polygons and explains, in context, what is happening to their place.

The full thesis lives in `pm/north-star.md`. Read it before making product decisions. The pre-pivot "globally deployable hourly ground truth for incident commanders" framing is **retired** (ADR 0005).

## What changed and why (read this before exploring)

The repo is mid-pivot. Q2 2026 re-scope from an IDC-demo system (detection + spread + weather + chatbot, FastAPI + Vite + PostGIS + Redis) to a narrow stewardship agent on free-tier infra (Next.js 16 + Drizzle + Neon + Vercel cron + AI Gateway). The legacy stack has been removed; only the A' implementation remains on `master`.

History:
- `pm/decisions/0001`–`0005` — research and pivot synthesis.
- `pm/decisions/0006` — stage-PR workflow (binding for all dev work).
- `docs/SPEC-A-prime-v1.md` — current product spec.
- `docs/pivot-architecture.md` — current architecture.

## Where strategy and mechanism live

| File | Purpose |
|---|---|
| `pm/PM_CLAUDE.md` | Operating doctrine. Decision rules, escalation triggers, maturity gate. **Read first.** |
| `pm/north-star.md` | Working product thesis. |
| `pm/decisions/` | Append-only ADRs. |
| `pm/backlog.md` | Candidate problems + stage status. |
| `pm/blockers.md` | Vanyo handoff queue (account creation, secrets, decisions). |
| `pm/briefs/` | Versioned dev-agent prompts per stage. |
| `pm/research-log/` + `pm/signals/` | Condensed research / raw evidence split. |
| `loop.md` | Autonomous heartbeat protocol — what the agent loop does each tick. |
| `.claude/agents/` | Subagent definitions: `pm`, `dev`, `reviewer`, `scout`, `cutover`. |
| `AGENTS.md` | Hard rules for any agent (no fabricated data, hard stops, push back on shortcuts). |

## Stack

- **Next.js 16** (App Router) — `app/`, `lib/`, hosted on Vercel.
- **Drizzle ORM** + **PostgreSQL 16 / PostGIS 3.5** on **Neon** (autoscale-to-zero).
- **PGlite** for unit tests; **`@testcontainers/postgresql`** with `postgis/postgis:16-3.5` for spatial integration tests.
- **GitHub Actions cron** for FIRMS polling (`.github/workflows/firms-poll.yml`).
- **AI Gateway** (`@ai-sdk/google` via Vercel AI Gateway) for brief generation.
- **Resend** for notification email dispatch.
- **Clerk** for auth + per-user AOIs.
- **TypeScript 5**, **vitest**, **eslint 9**, **pnpm 10**.

## Stage status (snapshot — confirm against `pm/backlog.md`)

- Stage 0 — Next.js scaffold ✅ merged.
- Stage 1 — AOI CRUD + Drizzle schema + PGlite tests ✅ merged.
- Stage 2 — FIRMS poll + AOI matcher + GitHub Actions cron ✅ merged.
- Stage 3 — Brief generation (AI Gateway, structured output) ✅ merged.
- Stage 4 — Notification dispatch (Resend) ✅ merged.
- Stage 5 — Auth (Clerk) + per-user AOIs ✅ merged.
- Stage 6 — Rules UI / export ✅ merged.
- Stage 7 — Launch readiness UI ✅ merged.
- Stage 8 — Authority perimeter + freshness ✅ merged.
- Stage 9 — Watch-confirmed email + first-AOI backfill ✅ merged.

## Commands

```bash
pnpm install
pnpm dev          # next dev — http://localhost:3000
pnpm typecheck
pnpm lint
pnpm test         # vitest
pnpm build        # next build
pnpm db:generate  # drizzle-kit generate (after schema changes)
pnpm db:migrate   # tsx scripts/db-migrate.ts
```

Spatial integration tests need Docker running locally for `@testcontainers/postgresql`. CI on GitHub Actions has Docker pre-installed.

## How to do work

1. Read `pm/PM_CLAUDE.md` (doctrine), `pm/decisions/0006-stage-pr-workflow.md` (workflow), the relevant brief in `pm/briefs/`, and `loop.md`.
2. Branch off `master`. Naming: `stage-N-<short>` for stage work, `chore/<short>` for chores.
3. Implement only what the brief says.
4. Locally: `pnpm typecheck && pnpm lint && pnpm test && pnpm build`.
5. Push, draft a PR body, hand off to the orchestrator (which calls `gh pr create`).
6. Stage PRs always require Vanyo's review — they bypass the auto-merge gate by design.

## Conventions

- **No dev-time scaffolding for hypothetical futures.** Don't add error handlers, validators, or fallbacks for paths that can't happen — `lib/firms/client.ts` only checks `FIRMS_MAP_KEY` because it's the build-without-blocking edge.
- **Comments are rare.** See top-level "Doing tasks" guidance — only comment when the *why* is non-obvious.
- **Two-backend repository pattern.** Code that hits the DB must work against both Neon+PostGIS (production) and PGlite (unit tests). Spatial code falls back to a turf.js path when the underlying connection isn't PostGIS — see `lib/firms/matcher.ts` for the pattern.
- **No `--no-verify`, no force-push.** Per `loop.md` invariants.
