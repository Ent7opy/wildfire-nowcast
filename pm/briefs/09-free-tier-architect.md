# Brief 09 — Free-tier architect + solo-footprint estimator

## Why this exists

Candidate A + D + E together are only viable if they run at $0–10/month on a solo maintainer's free-tier infrastructure. This brief produces the concrete architecture sketch that proves (or disproves) that feasibility, and the concrete cut list for the current repo.

**Read first:**
1. `pm/PM_CLAUDE.md`
2. `pm/decisions/0003-nonprofit-and-free-infra-constraints.md`
3. `pm/research-log/2026-04-21-repo.md` — current repo inventory + cut candidates
4. `pm/backlog.md` — candidate A description
5. `CLAUDE.md` + `docker-compose.yml` — current stack ground truth

## Goal

Two things:

1. **A concrete free-tier architecture** for A + D + E with a realistic cost model at 50 users / 100 AOIs / 1-notification-per-week-per-AOI scale. Name the specific services (Vercel, Neon, Cloudflare, GitHub Actions, AI Gateway, etc.) and each tier's limits.
2. **A concrete cut list** — which files and subsystems in the current repo get deleted, retained, or migrated. Table form, ranked by confidence.

## Method

- Use WebFetch / WebSearch to get current free-tier limits (they drift; don't trust memory): Vercel hobby, Neon free, Cloudflare R2 free, GitHub Actions free, Vercel AI Gateway, Vercel Blob, Railway (as current baseline for comparison), Fly.io, Supabase.
- Use `gh issue list`, Glob, Grep, Read for the repo inventory (read-only).
- Compute: per-AOI FIRMS API call frequency × average payload × storage × LLM tokens per reasoned-brief × notifications. Show the math.
- Assume: FIRMS allowed rate (check current terms), LLM cost at Gateway prices for a small model (Haiku-tier) and a medium model (Sonnet-tier).
- Compare against current Railway cost for the existing docker-compose stack (look at `railway.toml`, `railway.ingest.toml`).

## Output

**`pm/research-log/2026-04-21-free-tier-architecture.md`** — ≤1200 words (this one can run longer):
- `## Reference architecture (A + D + E)` — components, data flow, where state lives, where compute runs, auth model, how AOI subscriptions become cron jobs, how LLM reasoning is gated, rate limits.
- `## Cost model` — monthly cost at (a) 10 users/20 AOIs, (b) 50 users/100 AOIs, (c) 500 users/1000 AOIs. Include LLM costs separately. Compare to current stack.
- `## Free-tier risk table` — services, their free-tier limits, what blows up first as usage grows, migration path when it does.
- `## Cut list (from current repo)` — table: path / subsystem, action (KEEP / CUT / MIGRATE / REWRITE), rationale, confidence.
- `## Things we'd need to build new` — components that don't exist yet.
- `## Verdict` — achievable / marginal / infeasible, with caveats.

**`pm/signals/2026-04-21-free-tier-raw.md`** — service tier URLs, pricing snapshots with dates, FIRMS rate-limit citation.

## Constraints

- **No hand-waving.** "Should be cheap" is not an answer. Show the math.
- If a service has changed recently (e.g., Vercel Fluid Compute pricing), use the current public docs. Cite the URL with date of access.
- If the target is infeasible at some scale, say at what scale it breaks and what would need to change.

## Time budget

~45 min. This is the most load-bearing Phase 2 brief.
