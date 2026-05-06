# Blockers — what Vanyo can unblock

Single source of truth for the small handful of things PM_CLAUDE cannot do directly: account creation, secret-setup on platforms without exposed APIs, local CLI installs.

Update mode: PM_CLAUDE adds entries; Vanyo marks `[x]` when done; PM_CLAUDE removes resolved entries during ADR / synthesis turns.

---

## Active

No active blockers. Stage 3 (brief generation) is unblocked — see
`pm/briefs/17-stage3-brief-generation.md`. New Stage 3+ blockers will land
here as PM_CLAUDE discovers them while drafting the next stage's brief.

---

## Resolved (for the record)

- [x] **2026-05-06** — Stage 1 Neon Postgres provisioned. `DATABASE_URL` set on Vercel `wildfire-nowcast` (Preview scope). Stage 1 PR merged to master.
- [x] **2026-05-06** — Stage 2 local Docker confirmed running; `@testcontainers/postgresql` integration tests pass locally for Vanyo. CI on GitHub Actions has Docker pre-installed; Stage 2 PR is merged. Verified by the cron workflow `firms-poll.yml` shipping with `schedule:` toggle gated on this secret being live.
- [x] **2026-05-06** — Stage 2 `CRON_SECRET` set on Vercel (Preview + Production) and as a GitHub Actions repo secret. The `/api/aoi/poll` route's bearer-token check (`app/api/aoi/poll/route.ts`) and the `firms-poll.yml` workflow's pre-flight `CRON_SECRET` guard are in place; Vanyo can now uncomment the `schedule:` lines when ready.
- [x] **2026-05-06** — Stage 3 Vercel AI Gateway enabled; `AI_GATEWAY_API_KEY` set on Vercel (Preview + Production). Codebase does not yet reference the var (expected — Stage 3 has not started; build-without-blocking pattern means Stage 3 dev work introduces the read).
- [x] **2026-05-06** — Stage 4 Resend account created; `RESEND_API_KEY` set on Vercel (Preview + Production). Used in Stage 4.
- [x] **2026-05-06** — Stage 5 Clerk free-tier project created; `CLERK_PUBLISHABLE_KEY` + `CLERK_SECRET_KEY` set on Vercel (Preview + Production). Used in Stage 5.
- [x] **2026-05-06** — Vercel CLI installed locally. Convenience for PM_CLAUDE verification of preview deploys, env vars, function logs.
- [x] **2026-04-22** — `FIRMS_MAP_KEY` moved from Railway → Vercel env vars. Confirmed by Vanyo.
- [x] **2026-04-21** — `.claude/pm/` → `pm/` move. Vanyo's global gitignore excluded `.claude/`; resolved by moving PM workspace to a project-native path.
