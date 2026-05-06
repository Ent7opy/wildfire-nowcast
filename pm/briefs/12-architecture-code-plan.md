# Brief 12 — Architecture-to-code plan

## Why this exists

Agent 09 produced the reference architecture + cut list. This brief turns it into a concrete, ordered, safe migration sequence a solo developer can land over Q2 2026 without the system ever being "half-migrated" for more than a day.

**Read in order:**
1. `pm/PM_CLAUDE.md`
2. `pm/decisions/0005-problem-chosen-a-prime.md`
3. `pm/research-log/2026-04-21-free-tier-architecture.md` — BINDING reference architecture + cut list
4. `pm/research-log/2026-04-21-repo.md` — current repo inventory
5. `CLAUDE.md` + `docker-compose.yml` + `railway.toml` + `api/migrations/versions/`

## Goal

Produce `docs/pivot-architecture.md` — the ordered, safe rebuild plan. Concrete enough that a developer (human or agent) can execute it sequentially.

## Sections required

### 1. End-state architecture (restate, don't re-derive)
One-page summary of the target stack (Next.js 16 on Vercel, Neon Postgres, GH Actions cron, Vercel AI Gateway → Gemini Flash-Lite, Resend/webhooks, optional Cloudflare R2). Reuse the diagram from agent 09.

### 2. Starting state (current repo, honest)
Link to `research-log/2026-04-21-repo.md`. Note: 64 Alembic revisions, Docker Compose with 8 services, ~40k LOC to cut, 10 open "never ran in prod" issues.

### 3. The collapsed data model
≤10 tables. Draft the SQL / Prisma schema. Specify PostGIS geometry columns. Map every table to its purpose in the A' flow.

### 4. Migration sequence (the core deliverable)
A numbered, ordered, safe sequence of commits / PRs. Each step:
- **Goal** (one line)
- **Files touched** (add / delete / rename)
- **Reversibility** (can I revert in 1 commit if wrong? yes / no / with-caveats)
- **Verification** (what must be green before moving on)

Proposed phasing (agent should refine):
- **Stage 0** — set up parallel Next.js project skeleton in the repo under `apps/nextjs-ui/` (or similar) without touching existing code. Vercel preview green.
- **Stage 1** — port AOI CRUD. Neon schema migration. Verify CRUD in preview.
- **Stage 2** — port FIRMS ingest as a Vercel function + GH Actions cron. Verify end-to-end detection match.
- **Stage 3** — add LLM brief generation via AI Gateway. Verify schema output.
- **Stage 4** — add notifications (Resend + webhook). Verify delivery.
- **Stage 5** — land UI (map, AOI list, brief history). Verify against v1 spec acceptance criteria.
- **Stage 6** — BIG CUT — delete the subsystems listed as CUT in agent 09's table, in the order agent 09's dependency graph implies. Run tests after each subsystem deletion.
- **Stage 7** — retire Railway services + docker-compose + migrate DNS. Document rollback.

### 5. Dependency cut graph
Which CUT items unblock other cuts? E.g., cutting spread forecasting first enables cutting the weather / fuels / LFMC / lightning / terrain ingests. Draw the order.

### 6. Risk register
Top 5 things that can go wrong. For each: detection, mitigation, rollback.

### 7. Rollback playbook
If v1 is not ready by end of Q2, what's the minimum-regret pause state? (Hint: main branch keeps the original Docker stack until Stage 7 is green; new stack lives on a separate Vercel project until cutover.)

### 8. Concrete file inventory
Attach (as an appendix) a table that mirrors agent 09's cut list but with **current LOC**, **target LOC**, and **ticket title** for each line item. This becomes the execution checklist.

## Constraints

- ≤ 5,000 words.
- No hand-waving. "Migrate X" must include the file paths.
- Every Stage must be testable (green CI) before the next Stage begins.
- Solo-maintainer discipline: no Stage takes more than 5 days of solo work.
- `pm/**` and `docs/` both permitted for Write.

## Time budget

~50 min.
