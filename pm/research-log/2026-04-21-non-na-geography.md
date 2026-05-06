# Non-North-America geographic gap scan (2026-04-21)

**Agent:** 04 — Non-NA geography scout
**Method:** WebSearch + WebFetch across regional press, academic AARs, agency sites, Play Store reviews, Copernicus/JRC/EFFIS bulletins.

## Regional scan

### Mediterranean Europe (GR, PT, ES, IT, FR, HR, TR, CY, Balkans)
**In use:** EFFIS (EU-wide, authoritative, free), Copernicus EMS for damage mapping, national systems (Greece Civil Protection + 112 Cell Broadcast; Portugal ICNF SGIFR + IPMA; Spain nationwide state-led). Commercial: Technosylva Wildfire Analyst (primarily utilities/agencies). Citizen layer: Google Search/Maps wildfire notifications (launched 2024 across 14+ countries, shallow); volunteer-built apps like fogos.pt in PT.
**Gaps:** 2025 was EU's worst season ever (1.08M ha, 2× avg; 22 simultaneous very-large fires in PT/ES in Aug). Independent evaluations after Mati 2018, North Evia 2021, Evros 2023 repeatedly cite the same failure mode: *absent interoperable operational common picture between fire brigade, forest service, civil protection, and mayors* (OECD/Goldammer, IAWF, CTIF). 2025 Patras: Greek Civil Protection Minister publicly attacked a mayor for telling residents to *ignore* 112 alerts — trust in the official channel is visibly fraying. EFFIS itself concedes it misses small fires and has cloud/smoke latency.
**Language:** Greek, Portuguese, Spanish, Italian, Turkish, Croatian — national apps are local-language-only; Google layer translates but doesn't integrate local dispatch/evac data.

### Australia
**In use:** NSW RFS Fires Near Me + Hazards Near Me, VicEmergency, EmergencyWA, MyFireWatch (Landgate), NAFI (northern rangelands, CSIRO/CDU heritage), Digital Atlas near-real-time bushfire extents, bushfire.io aggregator.
**Gaps:** Play Store reviews (Nov 2025) cite 10-hour update lag and watch-zone geometry limits (5 km minimum, no polygons — useless in mountainous terrain). State-level fragmentation persists; federal app is a veneer over state feeds. NAFI's audience is pastoral/science, not incident commanders or residents.

### Amazon / South America (BR, BO, PY, CL, AR)
**In use:** INPE BDQueimadas (BR, strong archive, MODIS/VIIRS/AVHRR), FIRMS everywhere else, Copernicus EMSR + JRC GWIS for major events, SENAPRED/CONAF in Chile.
**Gaps:** INPE explicitly acknowledges understory Amazon fires are invisible to MODIS/VIIRS. BO/PY/PE lack national citizen apps — 2024 Pantanal season (340k+ hotspots in Sept, Bolivia declared national disaster, IACHR/OAS human-rights report) was tracked almost entirely via NASA FIRMS + international press. Chile Feb 2024 Valparaíso/Viña del Mar fire (130+ dead) post-mortem focused on water infrastructure, but communication failures well-documented.

### Southeast Asia (ID, MY, TH)
**In use:** Indonesia BMKG "Info BMKG" app, SiPongi+ (MinEnv & Forestry), BNPB data portal. Regional: ASEAN Haze Monitoring System (intent), Thailand-operated Fire-Danger Rating System for Upper SEA, Second ASEAN Haze-Free Roadmap 2023–2030.
**Gaps:** Problem is political/enforcement (East Asia Forum 2024: "framework misses the forests for the trees"), not satellite coverage. Transboundary haze is a known ASEAN concern but monitoring is not unified across borders, and peat-fire detection (sub-canopy smouldering) is unsolved.

### Sub-Saharan Africa
**In use:** CSIR AFIS (SA origin, near-real-time, free), GFED4s for research. Angola Aug 2024: 6% of land area burned in a week.
**Gaps:** Massive burned area, minimal operational tooling, but most fire is deliberate land management — not a catastrophic-loss market. Purchasing power low. Noted and skipped.

## Cross-regional patterns

1. **FIRMS floor is universal** — raw active-fire detection is effectively solved globally. The moat is nowhere in detection.
2. **The real gap is the operational layer between satellite hotspot and human decision**: no common operating picture across agencies, no confidence scoring, no spread outlook the public or a mayor can read, no integration with evac orders or road closures. Greece, Australia, Chile, Portugal — same story.
3. **Commercial vendors (Technosylva) sell to utilities + top-tier agencies only.** Mid-tier buyers (small-country civil protection, regional forestry, NGOs, Indigenous fire managers) are unserved.
4. **Google's wildfire layer is pan-global but deliberately shallow** — boundaries + notifications, no workflow, no SLA. It sets a ceiling on "just a map" businesses but leaves room above.
5. **Watch Duty is explicitly US-only** (all 50 states Dec 2025, no international announcement). The Watch-Duty-shaped hole outside the US is real.
6. **Trust in official channels is eroding visibly** in Greece (2025 Patras/112) and contested in Australia (app review complaints). An independent, AI-native second source that's actually fast could find traction.

## Language / access barriers

- Official national tools: Greek, Portuguese, Spanish, Italian, Turkish, Croatian, Indonesian, Portuguese (BR), Spanish (CL) — all local-language. English-only tooling will fail locals.
- EFFIS is EN-only and very institutional in tone; unusable for a resident or small-town mayor.
- INPE portals have partial English. BMKG/SiPongi are Indonesian-dominant.
- **LLM translation + AI-native summarization is an obvious unlock:** a single fire-event summary translated into 12 languages on the fly is something no incumbent does.

## Most promising geographic beachheads

1. **Mediterranean Europe, Greece-first.** Biggest recurring-catastrophe market outside NA; 2025 was unprecedented; repeated AARs name the *same* gap (no shared operational picture); trust in 112 is visibly fracturing; EFFIS is authoritative but not operational; Technosylva is priced/positioned for utilities. Greek tourism industry and expat community add English-speaking buyer segments. AI leverage: multilingual event summarization + spread outlook + cross-agency COP.
2. **Portugal + trans-Iberian (PT/ES).** Fogos.pt proves unmet demand for a citizen-grade tool (volunteer-built, known bugs). USE4FOREST EU-funded consortium shows that cross-border coordination is recognized as the gap. Smaller, more concentrated market than Greece but friendlier to a solo operator because the community around fogos.pt is tech-literate and English-comfortable.
3. **Australia (dark horse).** English-speaking, high willingness-to-pay, mature ecosystem means less greenfield *but* the Fires Near Me complaints (10h lag, 5km zones, no polygons) are concrete UX wins an AI-native tool could land. Multi-state fragmentation is a genuine lever for a cross-state aggregator + AI assistant. Risk: locals may view a non-AU startup as unwelcome.

Skipping: Amazon/SSA (low purchasing power, political-enforcement gap dominates), SEA (same; ASEAN coordination is a political problem, not a tooling one).

## Coverage notes

- **Strongest evidence:** Greece (multiple AARs, OECD 2024, IAWF 2023, 2025 EENA incident); Portugal (AGIF 2024, fogos.pt Play Store reviews); Australia (Play Store user reviews); pan-EU (JRC/Copernicus 2025 season stats).
- **Weaker evidence / gaps:** no direct user quotes from Indonesian residents about BMKG/SiPongi usability; no systematic AAR on Chile Feb 2024 info-tooling failure; Sub-Saharan Africa intentionally skimmed per brief.
- Did not contact vendors or validate Technosylva's Mediterranean contract footprint — public signals are thin.

## Surprises (PM attention)

1. **Greece 2025 Patras / 112 mayor incident** is a live, public confidence crisis around the official alert channel. This is a narrow but real window for an independent second source.
2. **Google has already claimed the "shallow global fire layer" slot** (2024 rollout, 14+ countries). A WFN that ships a shallower thing competes with Google and loses. The room above is operational depth.
3. **Fogos.pt exists as a volunteer citizen-grade PT tool** — proof of latent demand and proof that a solo operator CAN land in this space.
