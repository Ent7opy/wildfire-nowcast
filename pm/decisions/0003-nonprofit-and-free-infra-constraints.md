# 0003 — Non-profit positioning + hobbyist-level infrastructure cost

**Date:** 2026-04-21
**Status:** Accepted
**Stakeholder:** Vanyo

## Context

Wildfire Nowcast is the first tool inside the Earth Tools portfolio. Vanyo has stated that Earth Tools is non-profit by intent; the maximum acceptable revenue model is voluntary donations ("buy me a coffee") once the tool is genuinely useful. In addition, because the tool may see extended low-usage periods during its lifetime, it must run at hobbyist-level infrastructure cost — target $0–10/month total — so it does not become a financial burden.

## Decision

Two constraints become binding on all pivot candidates:

1. **Non-profit / donation-only.** No paid tiers, no enterprise SKU, no B2B revenue model as a dependency. Donations and optional sponsorships only.
2. **Free-tier-first infrastructure.** Target $0–10/month total operating cost at realistic early-stage usage. Prefer: Vercel free tier, Neon Postgres autoscale-to-zero, GitHub Actions cron, Cloudflare R2 / Vercel Blob, serverless / on-demand compute. Reject: always-on Docker Compose workers, continuous global ingest firehoses, Redis-dependent always-on queues.

The "maturity gate" in `PM_CLAUDE.md` is extended: every candidate must show an explicit free-tier infra plan that passes basic cost math before it can be `chosen`.

## Consequences

### Immediate candidate impact (see `backlog.md`)

- **Candidate B (wildfire risk for insurance) — REJECTED.** Insurance is an inherently commercial, regulated, licensed vertical. Non-profit framing is incompatible. Removed from shortlist.
- **Candidate A (fire-aware AOI agent) — reframed.** The "B2B operator" framing is dropped. New framing: *conservation NGOs, Indigenous fire stewards, protected-area managers, small municipalities, researchers, WUI homeowners, diaspora, journalists*. The agentic AOI-monitoring pattern survives; only the buyer archetype changes. Architecture must be free-tier (per-AOI on-demand queries, no continuous global ingest).
- **Candidate D (FIRMS-done-right substrate) — reinforced.** Open-source library fits non-profit perfectly. No change needed.
- **Candidate E (MCP / agent-consumable API) — reinforced.** Free tier + rate limits + donations fits naturally. No paid tier required for v1.
- **Candidate C (Greece-first Mediterranean) — still viable, smaller.** B2g concerns are less acute if product is free; trust moat remains real. Keep on list but de-prioritize.
- **Candidate F (AI-disinformation triage) — still niche, still open.** Fits non-profit framing. Keep as background option.

### Architectural impact on current repo

The current docker-compose stack (Postgres+PostGIS + Redis + titiler + pg_tileserv + worker + ingest_scheduler + api + ui) is **not** free-tier compatible. This reinforces findings from `research-log/2026-04-21-repo.md`: the pivot will likely replace most of the compute-side stack, not extend it. Subsystems flagged as "over-built" in Phase 1 are now also flagged as "incompatible with $0–10/month target."

### Positive side effects

1. **Removes the "will anyone pay" risk entirely** from candidate selection.
2. **Aligns with Earth Tools' stated mission** (open ecological intelligence).
3. **Forces architectural discipline** — free-tier operation is an excellent forcing function against scope creep.
4. **Open-source credibility compounds into Vanyo's end-of-May 2026 Accedia talk** on AI tools and dev processes.
5. **Watch Duty precedent ($25/yr voluntary donations) validates the "useful + free + donated to"** model in the wildfire space specifically.

## Phase 2 adjustment

Phase 2 agent briefs (see `briefs/07–10`) incorporate these constraints explicitly. No candidate will be allowed to pass the adversarial critique without a free-tier architecture sketch and a non-profit user/distribution story.
