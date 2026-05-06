# Blockers — what Vanyo can unblock

Single source of truth for the small handful of things PM_CLAUDE cannot do directly: account creation, secret-setup on platforms without exposed APIs, local CLI installs.

Update mode: PM_CLAUDE adds entries; Vanyo marks `[x]` when done; PM_CLAUDE removes resolved entries during ADR / synthesis turns.

---

## Active

### Vercel project Framework Preset misconfigured

- [ ] **Set Vercel project Framework Preset to "Next.js"** (currently auto-detect / Vite-era setting from before cutover)
  - Symptom: every preview deploy fails with `No Output Directory named "dist" found after the Build completed.` and earlier with `The specified Root Directory "ui" does not exist`. The repo has no `vercel.json` and `next.config.ts` exists at the root, so Vercel should auto-pick Next.js — but the project setting is sticky from the legacy Vite/`ui/` stack.
  - Two fix paths:
    - **Dashboard:** Vercel → wildfire-nowcast → Settings → Build & Development Settings → Framework Preset → Next.js. Clear Root Directory or set to repo root.
    - **In-code (preferred, survives mis-clicks):** add a `vercel.ts` per the platform's current recommendation and commit it. Loop can do this autonomously if greenlit.
  - **Currently blocking three PRs:** [#388](https://github.com/Ent7opy/wildfire-nowcast/pull/388) (Stage 3, LGTM'd), [#389](https://github.com/Ent7opy/wildfire-nowcast/pull/389) (this PR, pm chore), [#390](https://github.com/Ent7opy/wildfire-nowcast/pull/390) (dead-export cleanup, LGTM'd). Branch protection requires the Vercel check, so none can merge until the preset is fixed.

**When:** As soon as possible — loop is idle until this clears.

### Autonomy proposal awaiting decision

- [ ] **Decide on the autonomy package** proposed by the orchestrator at 15:36 UTC 2026-05-06 (in chat, not on a PR):
  - Fix Vercel preset via CLI / commit `vercel.ts`
  - Flip `allow_auto_merge=true` on the GitHub repo (currently off, so auto-merge can't even queue)
  - Amend ADR 0006 to scope "stage PRs require Vanyo" more narrowly
  - Narrow the auto-merge gate's exclusion list in `loop.md`

**Why:** Loop hit three consecutive idle ticks today (15:00 → 15:55 UTC) entirely because every productive PR funnels back to Vanyo by design. Without a decision on this package, the loop will continue idling whenever it produces work.

**When:** Whenever Vanyo is ready. Until then, expect more idle ticks.

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
