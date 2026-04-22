# Blockers — what Vanyo can unblock

Single source of truth for the small handful of things PM_CLAUDE cannot do directly: account creation, secret-setup on platforms without exposed APIs, local CLI installs.

Update mode: PM_CLAUDE adds entries; Vanyo marks `[x]` when done; PM_CLAUDE removes resolved entries during ADR / synthesis turns.

---

## Active

### Stage 1 — Neon Postgres

- [ ] **Create Neon free-tier project** at https://console.neon.tech
  - Any region (suggest `aws-eu-central-1` for EU latency, or `aws-us-east-1` for US)
  - Project name: `wildfire-nowcast` (or any — only the connection string matters)
- [ ] **Add `DATABASE_URL` to Vercel `wildfire-nowcast` project** (Preview environment scope only for now; Production scope when we get to Stage 7 cutover)
  - Connection string from Neon dashboard, the "pooled" variant (`-pooler` host) for Vercel functions
  - Vercel dashboard → wildfire-nowcast project → Settings → Environment Variables

**Why:** Stage 1 dev agent builds the AOI schema + CRUD against PGlite in-memory (so dev doesn't block on this). The Vercel preview deploy will be live the moment this env var lands. Without it, the preview will just show DB-connection errors on `/api/aoi/*` routes — fine for now, but pasting the env var lights it up.

**When:** Any time before the Stage 1 PR is merged. Not blocking the agent.

---

### Stage 2 — FIRMS MAP_KEY migration

- [ ] **Move `FIRMS_MAP_KEY` from Railway → Vercel env vars** (Preview + Production scopes)
  - Same key value as Railway currently uses
  - NASA's MAP_KEY rate limit is 5,000 transactions / 10 min — we'll be well under via bucket coalescing

**When:** Before Stage 2 dev work begins.

---

### Stage 3 — Vercel AI Gateway

- [ ] **Enable Vercel AI Gateway** on your team
  - Vercel dashboard → AI Gateway → Get started
  - Free tier includes $5/mo credit
- [ ] **Add `AI_GATEWAY_API_KEY` to Vercel `wildfire-nowcast` project** (Preview + Production)

**When:** Before Stage 3 dev work begins.

---

### Stage 4 — Resend (notifications)

- [ ] **Create Resend free-tier account** at https://resend.com
  - Free: 3,000 emails/mo, 100/day
- [ ] **Verify a sender domain** (or use Resend's `onboarding@resend.dev` for testing)
- [ ] **Add `RESEND_API_KEY` to Vercel** (Preview + Production)

**When:** Before Stage 4 dev work begins.

---

### Stage 5 — Clerk (auth)

- [ ] **Create Clerk free-tier project** at https://clerk.com
  - Free: 10,000 MAU
- [ ] **Add `CLERK_PUBLISHABLE_KEY` + `CLERK_SECRET_KEY` to Vercel** (Preview + Production)

**When:** Before Stage 5 dev work begins. Stages 1–4 use a single-user stub (`STUB_USER_ID = "stub-user-1"`); no auth needed for those.

---

### Convenience (non-blocking but unlocks better verification)

- [ ] **Install Vercel CLI locally** — `npm i -g vercel`

**Why:** Lets PM_CLAUDE verify Vercel preview deploys, pull env vars locally, run `vercel logs` against running deployments, check function execution times. Right now PM_CLAUDE trusts the git integration blindly — fine for Stage 0 but increasingly useful as functions, crons, and AI Gateway calls go live in later stages.

**When:** Whenever convenient.

---

## Resolved (for the record)

- [x] **2026-04-21** — `.claude/pm/` → `pm/` move. Vanyo's global gitignore excluded `.claude/`; resolved by moving PM workspace to a project-native path.
