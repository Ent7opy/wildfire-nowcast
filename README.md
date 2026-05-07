# Wildfire Nowcast

> A free, open, AI-native fire intelligence agent for stewardship-motivated users — conservation trusts, Natura 2000 site managers, Firewise communities, Indigenous fire crews, LTER field scientists, environmental journalists — that watches their specific polygons and explains, in context, what is happening to their place.

Full thesis: [`pm/north-star.md`](pm/north-star.md). Pivot history: [`pm/decisions/`](pm/decisions/).

## Status

Post-pivot (Q2 2026). The pre-pivot stack (FastAPI + Vite + Redis) has been removed. The A' implementation — AOI CRUD, FIRMS polling, AI-Gateway brief generation with authority-perimeter (NIFC / CWFIS) context, Resend dispatch, Clerk auth, rules UI / export, and a MapLibre-based dashboard view of AOIs with detection overlay — is on `master`.

Authoritative stage / backlog status: [`pm/backlog.md`](pm/backlog.md).

## Stack

Next.js 16 · React 19 · Drizzle ORM · Neon (Postgres 16 + PostGIS 3.5) · Clerk auth (Svix-verified webhooks) · Vercel AI Gateway (`@ai-sdk/gateway`) · Resend · MapLibre GL · Tailwind 4 · GitHub Actions cron · TypeScript 5 · vitest · pnpm 10.

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

Solo-operated, with a Claude Code agent harness driving day-to-day work. The harness reads [`loop.md`](loop.md) each tick and dispatches to subagents in [`.claude/agents/`](.claude/agents/):

- `pm` — owns the `pm/` workspace (briefs, backlog, blockers, ADR drafts, research-log condensation).
- `dev` — implements one stage or chore per run on a dedicated branch.
- `reviewer` — adversarial PR review against brief acceptance criteria; posts LGTM or BLOCKER.
- `scout` — background chores (dead-code audit, dependency bumps, doc drift, link sweeps); capped at one PR per UTC day.
- `cutover` — one-shot Phase 0 agent that landed the A' pivot on master; retired post-merge.
- `product-reviewer` — high-level product critic on thesis fit, flow coherence, and feature scope.

Stage PRs always require Vanyo's review; chores can auto-merge under the gate in `loop.md`.

Doctrine for human + AI agents: [`AGENTS.md`](AGENTS.md), [`pm/PM_CLAUDE.md`](pm/PM_CLAUDE.md).

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Donations-compatible; see ADR 0003 for the constraint that drove the choice.
