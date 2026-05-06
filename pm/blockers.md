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

**Why:** Stage 1 dev agent built the AOI schema + CRUD against PGlite in-memory. The Vercel preview deploy lights up the moment this env var lands.

**When:** Any time. Not blocking dev.

---

### Stage 2 — local Docker for integration tests

- [ ] **Confirm Docker is running locally** — `docker ps` should return a table (empty is fine)
  - Most likely already true: existing `docker-compose.yml` for the legacy Railway stack means Docker is installed
  - If "Cannot connect to Docker daemon": just open Docker Desktop
  - If not installed at all: `brew install --cask docker` (macOS)

**Why:** PGlite (Stage 1's in-memory test DB) does not include PostGIS. Stage 2's FIRMS-to-AOI spatial matcher uses `ST_DWithin` / `ST_Intersects`, which need real PostGIS. Stage 2 dev agent adds `@testcontainers/postgresql` to dev deps; tests spin up a `postgis/postgis:16-3.5` container per test run (~10s startup, then fast). GitHub Actions Ubuntu runners have Docker pre-installed so CI needs no changes.

**When:** Before Stage 2 PR opens. Non-blocking for the agent — it can write the testcontainer-using tests; Vanyo only needs Docker running to execute them locally.

---

### Stage 2 — CRON_SECRET (cheap, but required)

- [ ] **Generate a CRON_SECRET random string** (e.g. `openssl rand -hex 32`)
- [ ] **Add `CRON_SECRET` to Vercel `wildfire-nowcast` project** env vars (Preview + Production scopes)
- [ ] **Add the same `CRON_SECRET` as a GitHub Actions repository secret** at https://github.com/Ent7opy/wildfire-nowcast/settings/secrets/actions

**Why:** GitHub Actions cron will POST to `/api/aoi/poll` every 15 min. The function must reject anyone else hitting that endpoint, otherwise we burn FIRMS rate limit and Neon CU-hours. Standard pattern: shared secret in an `Authorization: Bearer <CRON_SECRET>` header.

**When:** Before Stage 2 PR merges. Until then the cron workflow is disabled at the YAML level (commit on/off toggle).

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

- [x] **2026-04-22** — `FIRMS_MAP_KEY` moved from Railway → Vercel env vars. Confirmed by Vanyo.
- [x] **2026-04-21** — `.claude/pm/` → `pm/` move. Vanyo's global gitignore excluded `.claude/`; resolved by moving PM workspace to a project-native path.
