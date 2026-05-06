# Candidate Problem Backlog

Status: `hypothesis` → `validating` → `rejected` | `chosen`.
Evidence: links to `research-log/` + `signals/` files.

---

## A' — Fire Stewardship Agent (revised from A per ADR 0004)

**Status:** proposed — awaiting Vanyo sign-off per ADR 0005

**Thesis:** a free, open, AI-native fire intelligence agent for stewardship-motivated users (conservation trusts, Natura 2000 site managers, Firewise communities, Indigenous fire crews, LTER field scientists, journalists) that watches their specific polygons and explains what is happening to their place, in context. Value is multi-source situation briefs (L2) — not threshold alerts (L1, which Watch Duty owns).

**Why this survives Phase 2 critique:** agent 07's WD-is-in-every-lane attack applies to *B2B utility / consumer-alert* framing. WD's DNA does not reach stewardship users in conservation / academic / sovereignty networks (agent 10 evidence). Agent 09 proves the free-tier architecture works at ~$0/mo target scale.

**Companion elements:**
- **D** folded as a 5-hour byproduct library (not a standalone product)
- **E** as thin MCP surface over A''s backend, ships 2–3 weeks after A' v1
- **C** (Mediterranean / Greek) folded as distribution accelerator, not separate candidate

**User (revised, non-profit framing):** Conservation NGOs, Indigenous fire stewards, protected-area managers, small municipalities, researchers monitoring specific plots, WUI homeowners, diaspora / expats with family in fire country, journalists covering named fire seasons.

**Problem:** They need to know when a fire is about to become *their place's* problem — personalized to their polygons, with a reasoned brief instead of a raw pixel — without buying enterprise GIS or running a research workflow. Watch Duty is US/CA consumer alerts only; EFFIS is institutional; Technosylva is $100k/yr enterprise. Nothing serves the "I care about this specific place, globally, for free, with an AI that explains" use case.

**Cost footprint target:** $0–10/month. Architecture must be on-demand per-AOI (not continuous global ingest); Neon autoscale-to-zero Postgres; GitHub Actions cron for AOI polling; Vercel serverless for API + UI; LLM via AI Gateway with per-user rate limits or BYO-key option.

**Evidence:**
- `research-log/2026-04-21-ai-leverage.md` L1 (ranked #2 by agent 05)
- `research-log/2026-04-21-non-na-geography.md` — global deployability matters; Watch Duty US-only
- `research-log/2026-04-21-repo.md` — AOI watchlist + pause-notifications primitive is unusually clean, ready-for-agent
- `research-log/2026-04-21-github.md` — no OSS equivalent

**AI leverage:** Durable background agent (Workflow DevKit) + tool-use over ingest/query API + structured output + reasoning layer gating "should I wake the human." Moat is the accreting policy graph (AOIs + rules + user history) + curated false-positive evidence graph — not the LLM.

**Survives "remove the AI layer" test:** yes. Without the LLM, this becomes a threshold-based alert system that would flood users with false positives. The reasoning layer is what makes it usable.

**Adversarial critique stub:**
- Technosylva could productize downward ("Pocket Edition" rumored)
- Watch Duty could add a B2B / international tier
- Custom-AOI is an integration-heavy sales model
- Willingness-to-pay at mid-market is unvalidated

**Solo-footprint estimate (rough):** Reuse FIRMS ingest + AOI watchlist + notifications. Cut: spread v1/v2/v3, perimeter authorities, archive scrubber, industrial coverage, current chat assistant, fuels/LFMC/LULC/lightning ingests. Net: likely cuts more code than it adds.

---

## B — Wildfire risk intelligence for insurance / reinsurance

**Status:** REJECTED 2026-04-21 (ADR 0003) — incompatible with non-profit constraint. Regulated, licensed, inherently commercial vertical.

**User:** Insurance underwriters, reinsurance cat modelers, large homeowners seeking FAIR-plan alternatives, mortgage servicers.

**Problem:** CalFire FHSZ maps systematically under-call real risk (Altadena, Palisades burned outside "high hazard" zones). Homeowners pushed into FAIR. No tool updates personalized fire-risk with live denoised detection + historical analogue + recovery data + regulator-auditable rationale.

**Evidence:**
- `research-log/2026-04-21-reddit.md` — strong r/California insurance discourse, explicit distrust of FHSZ
- `signals/2026-04-21-reddit-raw.md` — https://www.reddit.com/r/California/comments/1rukltr/ , https://www.reddit.com/r/California/comments/1q9o4l9/
- `research-log/2026-04-21-ai-leverage.md` L5 (NL historical query) adjacent
- Competitive brief does NOT cover this

**AI leverage:** RAG over historical analogues, structured extraction from news / authority PDFs, explainable risk scores (required for regulator submission).

**Adversarial critique stub:**
- ZestyAI Z-FIRE, CoreLogic, Verisk Firecast exist (proprietary, not live-event-focused)
- Insurance is a regulated + licensed business; shipping a "risk score" has legal exposure
- Requires calibrated ground-truth outcome data (losses by parcel) we don't have
- Sales cycle to insurers is long

---

## C — Greece-first Mediterranean operational second source

**Status:** folded into A' distribution per ADR 0004. Mediterranean Natura 2000 site managers are an archetype-3 slice of A'; 2025 Greek season is a launch-timing accelerator, not a separate candidate.

**User:** Residents, small-town mayors, civil protection units in GR/PT/ES/IT where 112 / authority alerts lag.

**Problem:** 2025 Greek Patras incident shows 112 trust fracturing. EFFIS is authoritative but institutional. Mayors need something between "ignore" and "believe." No trusted operational second source works across Mediterranean in local languages.

**Evidence:**
- `research-log/2026-04-21-non-na-geography.md` — highest confidence gap outside NA; AARs from Mati/Evia/Evros all name same failure mode
- `research-log/2026-04-21-twitter.md` — under-sampled non-English operator voice
- `research-log/2026-04-21-repo.md` — MeteoAlarm + Balkans hindcast configs already exist

**AI leverage:** Multilingual event summaries, cross-source reconciliation, structured extraction from national authority feeds in real time.

**Adversarial critique stub:**
- b2g sales cycle in civil protection kills solo operators
- Trust moat in emergency management is enormous and political
- Credibility as a non-Greek / non-Portuguese / non-Spanish entity is unclear
- Fogos.pt volunteer model suggests price ceiling is ~zero

---

## D — "FIRMS done right" as normalized substrate / library

**Status:** REJECTED AS STANDALONE 2026-04-21 (ADR 0004). NASA's March 2025 Static Thermal Anomalies release closed the industrial-masking moat; FIRMS API v4.0.66 has built-in freshness endpoints; audience <200 devs globally; pyronear's own wrapper sits 20-month-stale. Surviving as a ~5-hour byproduct PyPI publish extracted from A''s internals after A' v1 ships. No roadmap, no SLA, no donations story.

**User:** Every open-source wildfire developer + downstream applications (A, E). LA Times data desk, academic researchers, pyronear, climate-risk modelers.

**Problem:** Every FIRMS consumer reinvents: endpoint drift, SSL/auth, dedup, lineage, freshness SLA, industrial FP masking. No OSS library provides this. WFN already does it internally.

**Evidence:**
- `research-log/2026-04-21-github.md` — datadesk/nasa-wildfires issues #7, #15, #24, #39; no OSS library
- `research-log/2026-04-21-repo.md` — WFN's FIRMS ingest + watermarks + industrial masking is genuinely unusual
- `signals/2026-04-21-github-raw.md` — multiple perimeter-interpolation repeat-work repos

**AI leverage:** AI is not the product here — but a clean substrate enables L1/L3/L5 above.

**Adversarial critique stub:**
- "Data infrastructure" is notoriously hard to monetize
- Maintaining an OSS library has perpetual support cost
- NASA could upgrade FIRMS and obsolete the library
- Audience is small and technical

---

## E — Wildfire MCP / agent-consumable API

**Status:** retained as thin side artifact of A' per ADR 0004. Ships 2–3 weeks after A' v1 as an MCP wrapper over A''s backend — not an independent product. Free tier + BYO key.

**User:** AI application developers building climate-risk SaaS; reinsurance modelers using Claude/ChatGPT agents; journalists.

**Problem:** No global, normalized wildfire data surface consumable by agents. Axion (GEE MCP) exists for generic geospatial but not fire-event-normalized.

**Evidence:**
- `research-log/2026-04-21-ai-leverage.md` L4
- `research-log/2026-04-21-github.md` — no OSS equivalent

**AI leverage:** The product IS AI infrastructure.

**Adversarial critique stub:**
- Chicken-and-egg (need agent ecosystem to be consuming it)
- MCP protocol could be ephemeral; REST adapter mitigates
- Requires D finished first

---

## F — AI-disinformation / provenance triage for fire imagery

**Status:** hypothesis — niche

**User:** News organizations, agencies (NWT Fire, BC Wildfire Service) debunking circulating fake fire imagery.

**Problem:** Generative-AI fake fire images are now a routine live-fire pain point. Agencies manually debunk; no tool cross-checks image claims against FIRMS / EFFIS / authority perimeter agreement.

**Evidence:**
- `research-log/2026-04-21-twitter.md` — NWT 2023 "sensationalized slop" quote; CA 2025 examples
- `signals/2026-04-21-twitter-raw.md` — https://www.cbc.ca/news/canada/north/ai-generated-media-and-misinformation-of-n-w-t-wildfires-circulating-fire-officials-warn-1.7623579

**AI leverage:** Image analysis + geolocation + cross-reference to authority layers + provenance check.

**Adversarial critique stub:**
- Very narrow niche; episodic demand
- Unclear buyer (news orgs don't pay; agencies slow to procure)
- Depends on sustained fake-image problem
- Adjacent to much larger deepfake-detection markets

---

---

## Stage status (post-pivot, A' implementation)

- Stage 0–7: merged (see `CLAUDE.md` snapshot).
- **Stage 8 — authority-perimeter LLM tool-call + data-freshness honesty + outreach plan v1:** brief landed (`pm/briefs/22-stage8-authority-perimeter-and-freshness.md`), `hypothesis → in-progress`. Bundles product-review 2026-05-07 §5 #5, #7, #11.

---

## Rejected / not-brought-forward

- **B (2026-04-21)** — Wildfire risk for insurance. Rejected under ADR 0003 (non-profit constraint). Insurance is commercial, regulated, licensed.
- **D as standalone (2026-04-21)** — Rejected under ADR 0004. NASA STA release March 2025 closed the moat; surviving as byproduct only.
- **C as standalone (2026-04-21)** — Folded into A' per ADR 0004.
- **F — AI-disinformation triage (2026-04-21)** — Parked. Real emerging pain (NWT 2023, CA 2025) but niche, episodic, unclear buyer. Revisit if a named partner (agency or journalism org) approaches.
