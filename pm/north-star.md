# Wildfire Nowcast — Working North Star

**Last updated:** 2026-04-21 (proposed — awaiting ADR 0005 sign-off)
**Status:** Proposed thesis per ADR 0004. Until Vanyo accepts, the current direction is "tested but not committed."

## Thesis (proposed)

**Wildfire Nowcast is a free, open, AI-native fire intelligence agent for people whose relationship to land is stewardship — conservation trusts, protected-area managers, Indigenous fire crews, Firewise communities, LTER field scientists, environmental journalists — that watches their specific polygons and explains, in context, what is happening to their place.**

## Why this

- **It is not a Watch Duty clone.** WD owns US consumer public-safety alerts + enterprise B2B utility. Stewardship users live in conservation / academic / sovereignty networks WD does not currently reach and is structurally unlikely to chase.
- **It is AI-native, not AI-decorative.** Fast threshold alerts can be template-generated. Multi-source situation briefs anchored in a site's own fire history cannot. Stewardship users want depth over speed, which is exactly what LLMs do well.
- **It fits Earth Tools' non-profit posture.** Donations-compatible (land trusts already donate to mission orgs). Runs at ~$0/month on free-tier infra (Vercel + Neon autoscale-to-zero + GitHub Actions cron + AI Gateway Flash-Lite) per agent 09's architecture.
- **The audience is real and named** (agent 10). Natura 2000 site managers (140k ha burned in 2025 — record), 1,500+ pre-organized NFPA Firewise communities, ~1,300 US accredited land trusts, IPBN/FNESS/Kimberley LC Indigenous fire crews, ~28 US LTER sites + 500+ global field stations, ~200 dedicated wildfire-beat journalists.
- **Distribution exists.** Peer networks that already meet about fire — Land Trust Alliance Wildfire Resilience Network, LTER, IPBN, NFPA Firewise, EUROPARC, IJNR, SEJ. One newsletter post into any of these beats a consumer funnel.

## What Wildfire Nowcast is explicitly *not*

- Not a Watch Duty competitor (alert app for general public)
- Not a NOAA / Copernicus / NIFC replacement (authoritative forecasting / incident management)
- Not an ArcGIS layer set (enterprise GIS for agencies)
- Not a Technosylva competitor (enterprise spread simulator)
- Not a utility-vegetation-management tool (that's WD × Overstory)
- Not insurance cat modeling (regulated / commercial / out of scope per ADR 0003)
- Not a general-purpose environmental dashboard

## The AI-leverage bet

A well-designed stewardship brief looks like:

> *"VIIRS NOAA-20 detection 14 km N of [AOI: Pinewoods Preserve], 04:17 UTC. Wind 240° @ 28 km/h, pushing toward reserve. Authority perimeter (PT-ICNF) posted 90 min ago covers the detection cluster. Fuel moisture in this stand is below the post-2023-Mati regrowth threshold. Two comparable spread events near Evia (Aug 2023, Jul 2024) produced {outcome}. No immediate action recommended; next brief in 3h or on polygon breach."*

That paragraph is non-templatable. It is what stewardship users hire the tool to do.

## Milestones (if accepted)

- **Q2 2026 end:** A' v1 shipping on earth-tools.org/wildfire. One archetype well-served (leading candidate: conservation land trusts via LTA WRN). ~$0/month infra.
- **End of May 2026:** Accedia talk on AI tools + dev process — with a live narrow product to reference, not the retired IDC scope.
- **Q3 2026:** Second archetype (likely Natura 2000 site managers via EUROPARC, once Mediterranean season starts).
- **Q4 2026:** MCP surface (candidate E) published as thin side artifact; D extracted as a ~1-day byproduct PyPI publish.

## Retired candidates

See `backlog.md`:
- **B** — wildfire risk for insurance — rejected (non-profit incompatible, ADR 0003)
- **D (standalone)** — FIRMS-done-right as product — rejected (NASA STA closed the moat, ADR 0004). Surviving as byproduct library.
- **C (standalone)** — Greece-first Mediterranean — folded into A' distribution (Mediterranean Natura 2000 sites are an archetype-3 slice)
- **F** — AI-disinformation triage — parked (revisit with a named partner)
