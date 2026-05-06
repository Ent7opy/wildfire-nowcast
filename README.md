# Wildfire Nowcast

> A free, open, AI-native fire intelligence agent for stewardship-motivated users — conservation trusts, Natura 2000 site managers, Firewise communities, Indigenous fire crews, LTER field scientists, environmental journalists — that watches their specific polygons and explains, in context, what is happening to their place.

Full thesis: [`pm/north-star.md`](pm/north-star.md). Pivot history: [`pm/decisions/`](pm/decisions/).

## Status

Mid-pivot (Q2 2026). The pre-pivot stack (FastAPI + Vite + Redis + Railway) has been removed. Active work is on the Next.js / Drizzle / Neon / Vercel-cron implementation.

| Stage | Description | State |
|---|---|---|
| 0 | Next.js 16 scaffold | merged |
| 1 | AOI CRUD + Drizzle schema | merged |
| 2 | FIRMS poll + AOI matcher + Actions cron | merged |
| 3 | Brief generation (AI Gateway) | next |
| 4 | Notification dispatch (Resend) | pending |
| 5 | Auth (Clerk) | pending |
| 6 | Rules UI / export | pending |

Authoritative status: [`pm/backlog.md`](pm/backlog.md).

## Stack

Next.js 16 · Drizzle ORM · Neon (Postgres 16 + PostGIS 3.5) · Vercel · GitHub Actions cron · TypeScript 5 · vitest · pnpm 10.

## Quick start

```bash
pnpm install
cp .env.example .env.local   # fill in FIRMS_MAP_KEY at minimum
pnpm dev                     # http://localhost:3000
```

For Neon, FIRMS, and other secret setup steps Vanyo must do, see [`pm/blockers.md`](pm/blockers.md).

## Commands

```bash
pnpm typecheck
pnpm lint
pnpm test          # vitest, includes PGlite + (with Docker) PostGIS testcontainer
pnpm build
pnpm db:generate   # after schema changes
pnpm db:migrate
```

## How the project is run

Solo-operated, with a Claude Code agent harness driving day-to-day work. The harness reads [`loop.md`](loop.md) each tick and dispatches to subagents in [`.claude/agents/`](.claude/agents/) (`pm`, `dev`, `reviewer`, `scout`). Stage PRs always require Vanyo's review; chores can auto-merge under the gate in `loop.md`.

Doctrine for human + AI agents: [`AGENTS.md`](AGENTS.md), [`pm/PM_CLAUDE.md`](pm/PM_CLAUDE.md).

## License

TBD — non-profit / donations-compatible. See ADR 0003 for the constraint.
