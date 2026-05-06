# 0002 — Phase 1 synthesis

**Date:** 2026-04-21
**Status:** Accepted
**Supersedes:** nothing; partially annotates `docs/competitive-brief.md`

## Context

Phase 1 research swarm (6 agents, ADR 0001) returned. All six condensed logs and raw signals are on disk under `research-log/` and `signals/`. Write-permission denials in 4 of 6 sub-agent sandboxes were worked around by PM persisting inline-delivered content.

## Convergent findings (evidence from ≥2 independent agents)

### F1 — Watch Duty is un-competeable in its lane (Reddit + Twitter + Geography)
Watch Duty (8M users, US/CA, non-profit, 100K req/s @ <20ms during LA 2025) is treated as civic infrastructure: users pay $25/yr unprompted, founder explicitly refuses acquisition, brand is *ideologically* anti-software ("the magic isn't in the software, it's in the people"). **Corollary:** any pivot that targets US/CA public fire-alert space loses on trust before it loses on features. Lane is taken.

### F2 — The AI-native slot in wildfire tooling is genuinely empty (GitHub + Twitter + AI-leverage)
Pyronear, simfire, ELMFIRE, ForeFire, bcgov/wps, WMT — **zero LLM/agent surface**. WildfireGPT (17 stars, Argonne) is pure RAG Q&A, not tool-using, not real-time, academically criticized for lacking real-time integration and specialized training. No agentic / tool-composing / durable-monitoring product exists in the fire vertical. The chatbot-on-map anti-pattern is explicitly present in our own code (`AIChatAssistant.tsx` 482 LOC) and must go.

### F3 — FIRMS is universal but consumers keep reinventing the same plumbing (GitHub + Repo + Geography)
Endpoint drift, SSL issues, dedup, freshness SLA, industrial false-positive masking — every FIRMS consumer hand-rolls this (datadesk/nasa-wildfires issue trail, pyronear, Pantanal devs, academic projects). **WFN already has the only open drift-monitored FIRMS ingest with industrial masking — real adjacency.** The moat is nowhere in detection itself (FIRMS solves it globally) but in the clean, normalized, auditable substrate on top.

### F4 — The denoiser is NOT the proven moat (Repo)
Latest gate `20260305_140954/gate_report.json` — `"pass": false`, event precision 0.124, global F1 0.22. Issue #318: never registered in production. Issue #320: 95% drop rate in prod. The competitive brief (March 2026) treats "denoising" as the wedge; evidence says it's regressed and unvalidated. **Any pivot thesis leaning on denoiser quality as the wedge must re-validate first or choose a different wedge.**

### F5 — Mediterranean + Iberian geography is genuinely under-served (Geography + Reddit-lite + Repo)
2025 was EU's worst season ever. GR/PT/ES post-mortems name the *same* gap: no interoperable operational common picture. Greek Patras 2025 incident shows 112 trust is fracturing. Fogos.pt is volunteer-built proof of citizen-grade demand in PT. WFN already has MeteoAlarm + Balkans hindcast configs. **But b2g/civil-protection sales cycles kill solo operators.** Promising for product, hard as business.

### F6 — Post-fire + insurance is a large un-addressed pain (Reddit)
"13,000 destroyed, 7 rebuilt in LA County." r/California consistently argues CalFire FHSZ under-calibrates real risk (Altadena, Palisades burned outside "high hazard" zones). Homeowners pushed into CA FAIR plan. **Competitive brief does not address this.** Adjacent products exist (ZestyAI Z-FIRE, CoreLogic, Verisk Firecast) but none are live-event-anchored or regulator-auditable with rationale.

### F7 — Scope ran ahead of operation (Repo)
10 open issues, all 2026-04-04, all "never run in prod." Zero user-reported issues. Zero feature-request issues. `grep TODO/FIXME/XXX` returns zero matches. Feature work stopped ~2 weeks ago; recent commits are bugfix cadence on IDC-demo surfaces (archive scrubber, forecast SSE). **Permission to delete is real and extensive.**

## What the competitive brief got wrong (annotated, not deleted)

The brief (`docs/competitive-brief.md`, March 2026) recommends "ground truth for active fires — detection you can trust, spread you can act on." After Phase 1 evidence:

- "Detection you can trust" — **contradicted**. Our denoiser regressed (F4). The positioning claim is currently false.
- "Spread you can act on" — **too broad for solo-maintenance**; competes with Technosylva/NOAA/Copernicus (F7 and repo cost).
- Audience ("ICs + dispatchers + researchers + emergency managers") — too broad; each has different workflow and trust model.

Brief will be marked partially superseded; positioning recommendation is not binding on the pivot.

## Candidate problem directions (see `backlog.md` for details)

Six candidates identified. Ranked by PM gut-feel:

| # | Candidate | Strength | Main risk |
|---|---|---|---|
| A | **Fire-aware AOI agent for B2B operators** (utilities / insurers / infra / protected-areas) | Builds on WFN's proven primitives (FIRMS ingest, AOI watchlist); avoids WD moat; globally deployable; clear AI leverage | Technosylva could move downmarket |
| D | **"FIRMS done right" as data substrate / library** | Confirmed unmet plumbing need; WFN already has it | Hard to commercialize; better as layer beneath A + E |
| B | **Wildfire risk intelligence for insurance** | Huge market, strong Reddit signal, FHSZ mis-calibration is public | Regulatory barriers; ZestyAI + CoreLogic competitive |
| E | **Wildfire MCP / agent-consumable API** | No existing competitor; small but growing ecosystem | Depends on D being done; MCP protocol risk |
| C | **Greece-first Mediterranean second source** | Strongest human-need signal; WFN has Balkans configs | b2g sales cycle kills solo |
| F | **AI-disinformation / provenance triage for fire imagery** | Real emerging pain (NWT 2023, CA 2025) | Niche, episodic, unclear buyer |

## Recommendation to Vanyo

**Proposed shape:** combine A + D. Candidate D ("FIRMS done right") becomes the technical substrate — we already have it working, and it has a real open-source audience. Candidate A (B2B AOI agent with agentic monitoring + reasoned alerts) becomes the first commercial product on top of it. Candidate E (MCP / agent-consumable API) is the long-term surface for other AI apps. This is coherent, uses exactly what WFN already does well, avoids Watch Duty head-on, has clear AI leverage, and is honest about solo-maintainability.

Phase 2 proposal:
- 2 adversarial-critique agents (one targeting A, one targeting D) to try to kill each candidate
- 1 solo-operator-footprint agent to honestly estimate maintenance cost + cut list
- 1 buyer-persona agent to find 3–5 real named companies or agencies that fit the B2B AOI profile and evaluate whether any would actually pay

Pending Vanyo sign-off in `decisions/0003-problem-selection.md`.
