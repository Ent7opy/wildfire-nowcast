# 0004 — Phase 2 synthesis and revised product thesis

**Date:** 2026-04-21
**Status:** Proposed — awaiting Vanyo sign-off to become Accepted

## Context

Phase 2 ran 4 parallel agents (ADR 0003): two adversarial critics (A, D), one free-tier architect, one user-archetype scout. All 4 returned. Findings below, plus the reconciled product thesis.

## Findings

### D dies. Cleanly. Fold as a byproduct.
Agent 08's strongest attack lands: **NASA released the Static Thermal Anomalies (STA) Mask + Detections layers in March 2025** — global industrial / gas-flare / volcanic / landfill masking is now first-party. WFN's `industrial_coverage.py` is a reimplementation of a first-party layer released 13 months ago. The other four attacks reinforce: FIRMS API v4.0.66 has built-in `data_availability` + `missing_data` endpoints (drift narrative is stale); audience <200 devs globally; pyronear's own wrapper is 20-month-stale; GEE already is the substrate. D is not a product. When A ships, extract internal FIRMS + STA code, publish as a 5-hour byproduct library, and move on.

### A's architecture is feasible — but the B2B framing is dead.
Agent 07 landed a critique the reframe didn't survive: **Watch Duty's 2025 annual report** shows 74 staff across NA + Europe, a Pro tier, and an active B2B utility-AOI partnership with Overstory. Every utility-flavored archetype of ADR 0003 is already WD territory.

However — agent 07's attack targets the *utility / B2B operator* framing. Agent 10's evidence does *not* support the claim that WD reaches **stewardship-motivated users** (Natura 2000 site managers, LTER PIs, Indigenous Peoples Burning Network, land trusts, Firewise community leads). WD's DNA is *consumer public-safety alerts + enterprise B2B utility*. Stewardship users live in academic / conservation / sovereignty networks (EUROPARC, LTA, IPBN, LTER, ARSET) that WD does not currently reach and is structurally unlikely to chase.

### Agent 09 proves the math; agent 10 proves the audience.
- **Infra:** ~$0/month at 50u/100 AOIs (target), ~$20–25/mo at 500u/1000 AOIs, with bucket-coalesced polling + 5% LLM gate. Vercel + Neon autoscale-to-zero + GitHub Actions cron + AI Gateway Flash-Lite. Cut list: ~40k LOC CUT, ~3k MIGRATE, ~2k KEEP.
- **Audience:** Real, named, publicly-documented. Top 3 archetypes: (1) Protected-area managers (Natura 2000 140k ha burned 2025, LTER field sites, UNESCO biosphere reserves); (2) WUI homeowners via NFPA Firewise (1,500+ pre-organized communities); (3) Conservation NGOs with land holdings (Sonoma Land Trust, Midpen, TNC chapters, National Trust UK, Land Trust Alliance Wildfire Resilience Network). Long-tail that matters disproportionately: Indigenous fire stewards (IPBN, FNESS, Kimberley LC) for legitimacy; researchers for academic moat; journalists (Cart, Hagerty) for distribution amplification.
- **Distribution insight:** users discover via peer networks (LTA WRN newsletter, LTER, IPBN, NFPA Firewise, AGU), not App Store / Google Ads. One excellent LTA-WRN newsletter post > any consumer funnel.

## Revised thesis — Candidate A' ("Fire Stewardship Agent")

**One-line:** *A free, open, AI-native fire intelligence agent for people whose relationship to land is stewardship — conservation trusts, protected-area managers, Indigenous fire crews, Firewise communities, LTER field scientists — that watches their specific polygons and explains what is happening to their place, in context.*

### What changes from A → A'

| | A (original) | A' (revised) |
|---|---|---|
| Audience | "Anyone with a place to protect" (incl. utilities, infra, insurance cat teams) | Stewardship-motivated: land trusts, Natura 2000 managers, Firewise communities, Indigenous fire stewards, LTER PIs, field-station researchers, journalists |
| Value prop | Alert me fast when a fire threatens my AOI | Help me *understand* what's happening to my place — detections + authority context + weather + local history, explained |
| AI shape | L1 (AOI Watchkeeper) — reasoned alert (template-beatable per critique) | L2 (Situation Brief Agent) — multi-source synthesis over the user's site, grounded in their history. Non-templatable. |
| Primary risk | Watch Duty | *Not* Watch Duty in this audience. Primary risk is academic / conservation peer-network reach. |
| Distribution | B2B sales-ish | LTA WRN + LTER + IPBN + Firewise + EUROPARC newsletters, AGU posters, Substack bylines |
| Revenue | Donations (faces WD + Fogos.pt ceiling) | Donations — and the audience has existing donate-to-mission behaviour (land trusts, NPR-style stewardship orgs) |
| Infra | $0/mo target | Unchanged; agent 09 architecture holds |

### The AI-native claim now actually holds

L1-style alerts fail the "strip the AI layer" test (a template beats them). L2-style situation briefs do not — multi-event, cross-authority, cross-history synthesis grounded in the user's specific polygon and its past behaviour is genuinely non-templatable. Stewardship users want *depth* more than *speed*. A well-designed brief ("this detection is 14 km N of your reserve, wind is 240° at 28 km/h, current fuel moisture in this stand is below the post-2023-Mati-fire regrowth threshold, Natura 2000 management authority for this site is X with bulletin Y posted 90 min ago, similar spread behaviour in August 2023 near Evia produced outcome Z") is the product. The fast alert is a side feature.

### Candidate layering (revised)

- **A' — Fire Stewardship Agent.** Primary product. First milestone by end of Q2 2026.
- **E — MCP / agent-consumable wildfire data surface.** Thin wrapper over A''s backend. Ships as a side artifact 2–3 weeks after A' v1. Free tier + BYO key.
- **D — companion FIRMS + STA client library.** 5-hour extract from A''s internals, one PyPI publish, no roadmap. Honest positioning.
- **C — Mediterranean / Greek distribution.** Not a separate candidate. Reached *through* archetypes 3 (Natura 2000) and 5 (Mediterranean WUI communities) already covered by A'. The Greek 2025 signal becomes a launch-timing accelerator, not a product.
- **F — AI-disinformation triage.** Parked. Revisit if a named partner (NWT Fire, BC Wildfire, a journalism org) asks.

## What this means for the Accedia end-of-May 2026 talk

Talk is about AI tools + dev processes, not product demo. This revised thesis *strengthens* the talk: the story is "the original IDC system was 3 things done acceptably; the swarm + AI-native pivot cut it to one thing done well, deleted more code than it added, and produced a stewardship tool shipping on free-tier infrastructure for $0/mo." That narrative is more interesting to an internal AI-tools audience than any demo of the old scope would have been.

## Decision requested from Vanyo

1. **Accept A' as the chosen candidate.** If yes, I write ADR 0005 (problem chosen) and begin Phase 3 (design + spec).
2. **Accept the E/D/C/F layering above** (E as thin side artifact, D as byproduct, C as distribution accelerator, F parked).
3. **Confirm that "stewardship-motivated, non-profit, free, AI-depth-over-alert-speed" is the positioning** — this is the sentence the Accedia talk, the repo README, the `earth-tools.org` page, and the LTA WRN newsletter post all share.

If yes → I will:
- Rewrite `north-star.md` with A''s thesis as accepted
- Mark D, B, C, E, F statuses in `backlog.md`
- Kick Phase 3 with 3 agents: (a) spec writer for A' v1 (minimum shippable), (b) architect-to-code plan for the Next.js + Vercel + Neon rebuild with the cut list from agent 09, (c) launch-post drafter for the LTA WRN newsletter (to be held until product exists, but drafted now so the target audience shapes the product)

If "not yet" → happy to brainstorm. The most pressure-testable thing to challenge is the "WD won't reach stewardship users" claim in agent 07's Attack 1 — if you think WD's European expansion is broader than agent 10 found, we should dig deeper before committing.
