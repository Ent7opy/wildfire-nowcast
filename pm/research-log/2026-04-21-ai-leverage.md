# AI-Native Leverage Scout — 2026-04-21

**Agent:** 05 (ai-native-leverage)
**Inputs:** brief `briefs/05-ai-native-leverage.md`, web research (see `signals/2026-04-21-ai-leverage-raw.md`).
**Test applied to every card:** *strip the AI layer — is the tool still roughly as useful?* If yes → decorative, not native. Cards below survive that test.

## Candidate leverage points

### L1. Agentic AOI Watchkeeper
- **User:** Utility vegetation-management lead, protected-area manager, insurance cat team, or infrastructure operator with 10–10,000 fixed assets.
- **Moment:** Asset-owner draws/uploads polygons; subscribes in natural language ("wake me if a detection with confidence > 0.8 appears within 25 km and wind is pushing toward the polygon").
- **Inputs:** FIRMS NRT, HRRR/GFS wind, perimeters (NIFC/EFFIS/CWFIS), user's AOI polygons, historical false-positive mask.
- **Output:** An actionable alert *with a reasoned brief* — not "new FIRMS pixel." Brief says why it matters for *their* asset, what changed in last N hours, confidence caveats, link to map state.
- **AI capability required:** Durable background agent (Workflow DevKit-style) + tool-use over our tile/query API + structured output + a small reasoning model to gate whether to wake the human.
- **Unlocks:** Capability (NL subscription rules compile to queries), UX (non-GIS users get personalized alerts), Economic (one agent replaces a consulting spatial-analyst engagement).
- **Defensibility:** Moat is the *policy graph* (AOI + rule + user history) + curated false-positive mask, not the LLM. Durable.
- **Objection:** Watch Duty owns this surface for the US public. Answer: Watch Duty is free, human-curated, consumer-facing, US-only; "B2B operator with idiosyncratic AOIs + API + webhook" is a different product.

### L2. Fire Situation Brief Agent (long-context synthesis)
- **User:** Duty officer / ops researcher / journalist needing a 5-minute briefing on a named event.
- **Moment:** User clicks an event ("2026-04 Evia fire complex") — agent produces a one-page brief.
- **Inputs:** Our event timeline (detections + clustering), weather history/forecast, perimeter evolution, authority bulletins (NIFC/EFFIS/CWFIS), news articles, historical analogue events via RAG.
- **Output:** Structured brief — what/where/when, growth rate, fuel + weather driver, nearest analogue event (with citation), recommended watch items, uncertainty notes.
- **AI capability required:** Long-context synthesis (200k+), RAG over historical fire archive, tool-use to pull authoritative layers, structured output schema.
- **Unlocks:** Economic (replaces ~1h analyst work per event), UX (same brief for a PhD and an insurance adjuster), Capability (cross-source synthesis previously manual).
- **Defensibility:** Medium. Synthesis skill will commoditize; the *indexed historical analogue corpus* (FIRMS + perimeters + weather + outcomes, normalized) is durable.
- **Objection:** WildfireGPT (Argonne) did a version. Theirs is research-grade and US-centric; a global, event-anchored, API-first version is not shipped. Still needs differentiation.

### L3. Detection Triage Copilot (FIRMS noise killer)
- **User:** Any downstream consumer of FIRMS who drowns in industrial false positives (insurance ops, ops centers outside Northern-hemisphere forest-service coverage).
- **Moment:** New detection arrives; system must say "fire / not fire / review" in <60s.
- **Inputs:** FIRMS pixel, NASA static thermal mask, local industrial sources, OSM landuse, persistent-heat history, weather, diurnal/landcover priors.
- **Output:** Confidence score + **natural-language rationale** + auto-linked evidence (e.g., "matches flare polygon stamped 2024-07 — 43 prior hits at this pixel").
- **AI capability required:** Small classifier (already have: denoiser v2) + LLM *reasoner* producing rationale and citing evidence. Tool-use for evidence lookup.
- **Unlocks:** UX (non-scientists can trust + audit), Capability (auditable model decisions — key for insurance/regulator use), Economic (review queue shrinks by a large factor).
- **Defensibility:** Strong. The *evidence graph* (industrial registry, historical persistence, named false-positive patterns per region) compounds with use. LLM layer is swap-in.
- **Objection:** Remove the LLM and XGBoost still works — is this decorative? No: the audit/rationale is the product. A score without a rationale is not insurable.

### L4. Cross-Authority MCP / "Fire Data API for Agents"
- **User:** AI application developer or power user whose own agent needs wildfire context (climate-risk SaaS, reinsurance modelers, journalists using Claude/ChatGPT).
- **Moment:** Their agent issues a tool call: `fires.active_near(lat, lon, radius_km)` or `fires.historical_analogues(weather_profile, biome)`.
- **Inputs:** Our normalized index across FIRMS + NIFC + EFFIS + CWFIS + WFIGS + Copernicus EMS + our denoised layer.
- **Output:** Structured JSON, citation per field, uncertainty metadata.
- **AI capability required:** MCP server, strict schemas, rate-limited. AI capability here is about *being consumed by* AI — we're infra for other people's agents.
- **Unlocks:** Capability (no single global fire-data MCP today — GEE MCP exists but not fire-event-normalized), Economic (API monetization), Defensibility (data normalization across 6 authorities is the moat, not the protocol).
- **Objection:** MCP may not be the winning protocol; could be ephemeral. Ship boring REST first, MCP as a thin adapter.

### L5. Natural-Language Historical Fire Query (RAG corpus)
- **User:** Researcher, climate-risk analyst, insurance underwriter, catastrophe modeler.
- **Moment:** "Show fires in Mediterranean biome with FWI > 40 and >500 ha growth in first 24h over the last 20 years; cluster by region and plot outcomes."
- **Inputs:** Historical FIRMS, perimeter archives, reanalysis weather (ERA5), landcover, NL query.
- **Output:** Map + table + 1-paragraph summary; "save as agent" to re-run continuously.
- **AI capability required:** NL→structured query (text-to-SQL over our normalized schema), RAG over scientific + event corpus, chart generation.
- **Unlocks:** UX (the #1 "need a grad student for a week" request in climate research), Economic (10× cheaper than ArcGIS + analyst), Capability (questions only askable if the data is normalized — which nobody has done globally).
- **Defensibility:** High if the normalized global fire-history dataset is actually built. Lower if ESRI/CARTO ships it first — monitor.
- **Objection:** Adjacent to Earth Copilot / Bunting Labs / GIS Copilot. Those are generic GIS. Wildfire-specialized vertical is defensible if narrow.

### L6. Structured Extraction from Unofficial Signals (news + social + authority PDFs)
- **User:** Same as L2, but as an input feed.
- **Moment:** Authority posts PDF bulletin / news reports evacuation order / regional feed has a citizen report — we extract structured event data within minutes.
- **Inputs:** RSS + scraped authority bulletins + news APIs + (where ToS allows) social.
- **Output:** Structured events (location, status change, evacuation zone polygon) augmenting satellite detections, especially in the 2–6h FIRMS revisit gap.
- **AI capability required:** Schema-driven LLM extraction, geocoding, confidence routing.
- **Unlocks:** Capability (no satellite gap), Economic (scales across languages), UX (feeds L1/L2).
- **Defensibility:** Medium — published technique (Springer 2025 structured-disaster-extraction) is generalizable. Edge is the *curated source list per region* + trust calibration.
- **Objection:** Hallucination + ToS risk on social. Must be gated by confidence + source trust; never auto-promoted to "confirmed."

## What's already shipped elsewhere

- **Earth Copilot (Microsoft/NASA)** — multi-agent geospatial NL query over Planetary Computer STAC; generic, not fire-vertical.
- **WildfireGPT (Argonne)** — RAG + conversational wildfire analysis; research prototype, US-focused, not API/agent-consumable.
- **Bunting Labs / Mundi + Kue** — NL agent over PostGIS/QGIS; generic GIS, not fire-specialized.
- **ZestyAI Z-FIRE** — wildfire risk for insurance; proprietary, not NL/agent-native, not live-event-focused.
- **PagerDuty SRE Agent (2025)** — the "agent watches, wakes the human" pattern is now production-grade in SRE/ops. Transfers directly to L1.
- **Axion (GEE MCP)** — MCP-over-geospatial pattern is live; no fire-event-normalized equivalent.

## What would require Vercel-ish infra specifically

- **Workflow DevKit (durable execution)** — L1 (AOI Watchkeeper) and L6 (continuous extraction) are exactly the "durable, resumable, crash-safe agent" use case WDK was built for. Self-hosting this with RQ/Redis is possible but WDK collapses weeks of plumbing. Fits solo-maintainer constraint.
- **AI Gateway (multi-model)** — L3 and L5 benefit from model-mixing: cheap small model for triage rationale, big model for synthesis briefs. One key, cost tracking, failover.
- **Runtime Cache + Edge** — L4 (MCP/API) wants low-latency, tag-invalidated caching of normalized layers. Vercel Runtime Cache with tag invalidation on ingest is a natural fit.
- **AI SDK + tool use** — L1/L2 compose multiple tool calls over our API; AI SDK's structured tool-use is a good substrate.

## Ranked take (author opinion)

1. **L3 (Detection Triage Copilot)** is strongest. We already ship the classifier; adding LLM rationale + evidence graph is a thin integration that *changes the buyer* from "science team" to "insurance/reg ops." The AI layer is not decorative — a score without a rationale is not usable by a regulated buyer. Defensibility comes from the curated evidence graph.
2. **L1 (Agentic AOI Watchkeeper)** is the most durable product surface. Pattern is proven (PagerDuty SRE Agent), underlying infra (Workflow DevKit) is GA, Watch Duty does not serve B2B operators with custom AOIs internationally, and the subscription policy graph accretes value. High solo-maintenance leverage because the agent does the human-notification judgment we'd otherwise tune manually.

L2 is attractive but commoditizing fastest. L4 is the right *shape* but depends on finishing data normalization — an outcome, not a product. L5 is the most impressive demo but it's also what Earth Copilot + CARTO are racing toward. L6 is an input, not a standalone product.

## Anti-patterns to avoid

- **Chatbot-on-map** — a sidebar that answers "what's the biggest fire?" when the map already shows it. Decorative. (Our current `AIChatAssistant` risks this.)
- **"AI-powered insights" badges** — labeling existing statistics as AI outputs.
- **LLM-in-the-hot-path for detection** — using an LLM to classify FIRMS pixels in real time is slow and expensive; small classifier + LLM *explainer* is the right split (L3).
- **Autonomous evacuation/suppression decisions** — never. The agent informs; the human decides. Autonomy beyond "should I wake you" is liability, not product.
- **Generic "ask the map anything"** — without a schema + curated corpus, quality caps at hallucination risk. L5 must commit to a normalized dataset; otherwise it's a demo.
- **Building on one model family** — defensibility that evaporates when the next model ships. Route via AI Gateway; treat model choice as config.
