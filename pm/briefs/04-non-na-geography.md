# Brief 04 — Non-North-America Geographic Gap Scout

## Why this exists

Watch Duty owns US/Canada. ArcGIS / NIFC / CWFIS dominate agency use in North America. But the Mediterranean, Australia, Amazon, and Southeast Asia face serious fire problems with far weaker tooling, and Wildfire Nowcast's architecture is globally deployable. This may be where the real gap is.

Read `pm/PM_CLAUDE.md` first.

## Goal

Identify underserved fire-active regions: who's there, what they use, what's missing, and whether a globally-deployable, AI-native tool would find an audience.

## Method

WebSearch + WebFetch + Playwright as needed. No need to exhaust — breadth-first scan is fine.

**Regions to cover:**
- **Mediterranean Europe** (Greece, Portugal, Spain, Italy, France, Croatia, Turkey) — frequent catastrophic fires, EFFIS regional, patchy national tools
- **Australia** — strong state-level tools (Fires Near Me NSW, VicEmergency, WA EmergencyWA), NAFI for northern Australia; what do they lack?
- **Amazon / South America** — Brazil (INPE / Queimadas), Bolivia, Paraguay — huge fire activity, minimal operational tooling for non-agency users
- **Southeast Asia** — Indonesia (peat fires, transboundary smoke — ASEAN concern), Malaysia, Thailand
- **Sub-Saharan Africa** (briefly) — large burned area, least tooling, but also least purchasing power — note and move on

**For each region, capture:**
- Official / authoritative tools (name, scope, maturity, public accessibility)
- Popular non-official tools people actually use
- Major recent fire events and what the after-action discussion revealed about tooling gaps
- Language barriers — is the tooling only in local languages? Is English-only tooling failing locals?
- Data availability — FIRMS covers globally; what about local ground-truth / perimeters?

**Good sources:**
- EFFIS / JRC publications (EU)
- Academic papers on fire monitoring in region
- Regional press coverage during major events
- Reddit and forum threads from affected residents
- NGO and research-institute websites (CSIRO AU, INPE BR, JRC EU)

## Constraints

- Don't invent. If a region's data is thin, say so.
- Be wary of vendor marketing pages — note what they claim vs. what's verifiable.
- Distinguish "used by authorities" from "used by the public / researchers".

## Output (exact paths)

**1. `pm/research-log/2026-04-21-non-na-geography.md`** — ≤800 words:
- `## Regional scan` — one subsection per region: tools in use, gaps, recent event lessons
- `## Cross-regional patterns` — what's missing everywhere outside North America?
- `## Language / access barriers`
- `## Most promising geographic beachheads` — 1–3 regions where a globally-deployable AI-native tool would plausibly find traction, with reasoning
- `## Coverage notes`

**2. `pm/signals/2026-04-21-non-na-raw.md`** — raw bullets, cited.

## Time budget

~30 min.
