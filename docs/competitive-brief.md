# Wildfire Nowcast — Competitive Brief
**Date:** March 28, 2026
**Purpose:** Product strategy and feature prioritization
**Scope:** Wildfire detection, situational awareness, and weather integration tools used by incident commanders and fire researchers

> **⚠ 2026-04-21 — PARTIALLY SUPERSEDED.** PM Phase 1 research ([`pm/decisions/0002-phase-1-synthesis.md`](../pm/decisions/0002-phase-1-synthesis.md)) contradicts two load-bearing claims in this brief:
>
> 1. **"Detection you can trust" positioning (§Positioning Analysis, line 143)** — contradicted. The denoiser regressed; latest gate report fails (precision 0.124, F1 0.22), prod runs unregistered with a 95% drop rate. Cannot be the wedge without re-validation.
> 2. **Broad audience framing (ICs + dispatchers + researchers + emergency managers)** — too wide for solo-maintenance and implicitly competes with Watch Duty / Technosylva / ArcGIS on their own terms.
>
> The competitive landscape analysis below remains useful as reference. The recommended positioning ("ground truth for active fires") is not binding on the Q2 2026 pivot. See `pm/backlog.md` for current candidate directions.

---

## Executive Summary

The wildfire detection and situational awareness market is **fragmented by design** — no single tool combines real-time denoised detection, spread forecasting, and weather integration in an accessible, globally deployable package. Incident commanders typically run 3–5 separate systems simultaneously. Wildfire Nowcast sits in a genuine white space: the only tool that integrates ML-denoised satellite detections with spread forecasting and weather at hourly refresh, without an enterprise price tag or geographic restriction.

The main risk is not that a competitor does this better — none currently do. The risk is that users don't know Wildfire Nowcast exists, and that the product doesn't yet speak the operational language of its primary users.

---

## The Competitive Set

### Tier 1 — Direct Competitors (solve the same problem for the same users)

#### NASA FIRMS
**What it is:** The data source. MODIS + VIIRS satellite hotspot detections, globally, free, 3-hour latency (real-time for US/Canada).

**Strengths:** Authoritative, free, open formats, trusted globally. Every other tool in this list ingests FIRMS data.

**Weaknesses:** Raw detections with no denoising, no spread prediction, no weather, no UI worth using operationally. Noise from gas flares, industrial heat, and persistent hotspots is a known problem — and FIRMS doesn't solve it.

**Implication for Wildfire Nowcast:** FIRMS is your upstream data source *and* your most common baseline comparison. Users who currently use FIRMS directly will evaluate you on: "is this more trustworthy than what I already get for free?" Your denoiser is your answer to that question. Make the confidence signal visible.

---

#### Watch Duty
**What it is:** Community-powered wildfire alerts combining satellite hotspots, volunteer radio scanner monitoring, and verified first-responder updates. 8M+ users, $25/year premium tier. Non-profit.

**Strengths:** Most trusted consumer fire app in the US. Volunteer-verified reduces false alarms. Wind speed/direction displayed. Air tanker tracking. Used in Emergency Operations Centers as an intelligence feed during major events (LA 2025 fires). Enormous word-of-mouth growth.

**Weaknesses:** US/Canada only. No spread prediction. No fire behavior modeling. No API. Weather is limited to wind — no humidity, no temperature. Accuracy depends on volunteer availability. Not designed for tactical IC use.

**Implication for Wildfire Nowcast:** Watch Duty owns the public/community tier. Don't compete with it there — you'll lose on brand and community. But Watch Duty has no spread prediction, no denoising rigor, and no weather depth. Users who outgrow Watch Duty (agency dispatchers, researchers, ICs who need more than alerts) are your audience.

---

#### ArcGIS Wildfire Aware (ESRI)
**What it is:** Enterprise GIS platform with 22 integrated data layers — VIIRS detections, NIFC perimeters, NWS weather overlays (wind, temperature, humidity, weather warnings), historical perimeters back to 1878. Updated every 15 minutes.

**Strengths:** Most comprehensive weather integration of any fire platform. Customizable for agencies. Enterprise-grade. Long historical record. Integrates everything: satellite, perimeters, weather, terrain, air quality.

**Weaknesses:** Requires ESRI licensing ($2K–$10K+/year). Requires GIS expertise to use well. No fire behavior modeling or spread prediction. Dependent on third-party data freshness. Complex for non-GIS users.

**Implication for Wildfire Nowcast:** ArcGIS is what well-funded agencies already have. You can't out-feature them on raw data layers. What you have that they don't: denoised detections (ESRI passes through raw FIRMS), spread forecasting, and hourly refresh without requiring a GIS team to operate. Your target is agencies and researchers who want operational insight without the ESRI licensing cost and complexity.

---

### Tier 2 — Specialist Tools (solve one part of the problem very well)

#### Technosylva Wildfire Analyst
**What it is:** Cloud-based wildfire spread simulation for utilities and state fire agencies. On-demand spread prediction in seconds, what-if scenarios, asset risk prioritization. 20 utilities, 13 state agencies across 31 US states and Canadian provinces.

**Strengths:** The most operationally deployed spread simulator. Fast. Scenario analysis. Asset-level risk modeling. Integrated weather forecasting. Proven with major utilities (PG&E, Xcel Energy).

**Weaknesses:** Opaque pricing, enterprise-only. Requires trained operators. No public version. Limited to existing customer geographic deployments. Detection is an input requirement — it doesn't do detection itself.

**Implication for Wildfire Nowcast:** Technosylva is your most feature-complete competitor for the spread forecasting component, but it's out of reach for most users (cost, complexity, sales cycle). Your spread model doesn't need to beat Technosylva's physics — it needs to be good enough, accessible, and integrated with detection data that Technosylva doesn't provide natively.

---

#### Pano AI
**What it is:** AI-powered camera detection — 360° panoramic cameras with infrared sensors, 24/7 monitoring, human analyst verification. 10-mile radius per station. Deployed with Austin Energy, Xcel Energy, Washington DNR.

**Strengths:** Minutes-to-alert vs. hours for satellite. Works through smoke and at night. Precise GPS location via triangulation. Human-verified = very low false positive rate.

**Weaknesses:** Coverage limited to installed camera stations. High capital cost. Utility/corridor coverage only. Can't provide regional or global situational awareness.

**Implication for Wildfire Nowcast:** Pano AI is a detection-complementary tool, not a direct competitor. Their cameras catch fires before your satellites do. If you ever wanted to integrate non-satellite detection sources, Pano AI is the kind of partner to think about — their output feeds into platforms like yours.

---

#### NOAA Fire Weather Tools
**What it is:** Authoritative US fire weather forecasting — 8-day outlooks, high-resolution temperature/humidity/wind/lightning/precipitation forecasts, smoke modeling, deployable incident meteorologists.

**Strengths:** Best fire weather forecasting available. Multi-timeframe (tactical to 14 days). Smoke dispersion modeling. Lightning detection for fire starts.

**Weaknesses:** Weather only — no detection, no spread prediction, no incident tracking. Requires meteorological training to use well. Not designed as an integrated operational tool.

**Implication for Wildfire Nowcast:** NOAA's data is what you should be serving through your weather panel, formatted for non-meteorologists. They produce the authoritative numbers; you make them operationally legible at the fire location. The gap is exactly what your weather feature is building toward.

---

#### CWFIS, InciWeb/WFIGS, EFFIS
These are the national/regional authority systems — Canada, US, and Europe respectively. They provide official incident tracking, perimeters, and fire danger forecasting for their regions.

**Common pattern:** Authoritative within their geography, operationally important, but siloed (detection ≠ behavior ≠ weather ≠ incident management in any one tool), and none globally deployable.

**Implication for Wildfire Nowcast:** Your perimeter ingestion already pulls from NIFC, CWFIS, WFIGS, and Copernicus EMS. This is correct — you're consuming their authority and adding interpretation on top. Position yourself as the integration layer, not a replacement for any single authority source.

---

### Tier 3 — Emerging Entrants (watch list)

| Company | What They're Building | Why to Watch |
|---|---|---|
| **OroraTech** | Nanosatellite constellation — thermal detection with 3-minute alert time, global, day/night | First dedicated fire detection satellite constellation. If they scale, they could undercut FIRMS on latency |
| **Dryad Networks** | IoT gas sensors in forests — chemical detection before visible flames | Ultra-early detection; complements satellite, doesn't replace it |
| **Farmonaut** | Satellite detection + spread prediction + weather overlay, commercial | Closest feature overlap with Wildfire Nowcast. Claims 99.5% false positive reduction. Worth monitoring closely |
| **WIFIRE Edge (UC San Diego)** | Research platform: real-time fire simulation with coupled weather | Sophisticated physics model. Could become a commercial platform or be acquired |

---

## Feature Comparison

| Capability | Wildfire Nowcast | NASA FIRMS | Watch Duty | ArcGIS Wildfire | Technosylva | Farmonaut |
|---|---|---|---|---|---|---|
| **Real-time satellite detection** | ✅ Hourly | ✅ 3-hr | ✅ (via FIRMS) | ✅ 15-min | ❌ (input only) | ✅ 2.5-min scan |
| **ML denoising / fire vs. noise** | ✅ XGBoost v2 | ❌ Raw | Partial (human) | ❌ Raw | ❌ | Claimed (ML) |
| **Confidence scoring** | ✅ | Partial (FRP) | ❌ | ❌ | ❌ | ❌ |
| **Spread forecasting** | ✅ v2 | ❌ | ❌ | ❌ | ✅ Best-in-class | ✅ Basic |
| **Weather at fire location** | 🔧 In progress | ❌ | Wind only | ✅ Layers | ✅ Integrated | ✅ Overlay |
| **Relative humidity display** | 🔧 In progress | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Official perimeter ingestion** | ✅ Multi-source | ❌ | Partial | ✅ NIFC | ❌ | ❌ |
| **Historical replay / archive** | ✅ | Partial | ❌ | ✅ (back to 1878) | ❌ | ❌ |
| **API access** | ✅ | ✅ | ❌ | ✅ (paid) | ❌ | Unknown |
| **Globally deployable** | ✅ | ✅ | ❌ (US/CA) | ✅ (paid) | Partial | ✅ |
| **Free / accessible** | ✅ (alpha) | ✅ | ✅ basic | ❌ $2K–$10K+/yr | ❌ Enterprise | ❌ |
| **Personal safety mode** | ✅ | ❌ | ✅ (alerts) | ❌ | ❌ | ❌ |

---

## Positioning Analysis

### How competitors describe themselves

| Tool | Category Claim | Target | Key Differentiator |
|---|---|---|---|
| NASA FIRMS | "Satellite fire data" | Global technical users | Authoritative source |
| Watch Duty | "Wildfire safety" | Public/community | Volunteer-verified, trusted |
| ArcGIS Wildfire | "Fire situational awareness" | Enterprise agencies | 22-layer GIS integration |
| Technosylva | "Wildfire risk intelligence" | Utilities, state agencies | On-demand spread simulation |
| Pano AI | "Early wildfire detection" | Utilities, infrastructure | Minutes-to-alert, camera-based |

### Where Wildfire Nowcast should position

**The gap nobody owns:** *Trusted, denoised, globally deployable fire intelligence — detection confidence + spread outlook + weather context — accessible without enterprise contracts or GIS teams.*

Candidate positioning: **"Ground truth for active fires — detection you can trust, spread you can act on."**

This works because:
- "Ground truth" directly addresses the noise problem that FIRMS has and competitors ignore
- "Detection you can trust" positions the denoiser as a first-class feature, not a technical footnote
- "Spread you can act on" separates from Watch Duty (alerts) and FIRMS (raw dots)
- The whole statement is jargon-free enough for emergency managers while credible to researchers

---

## What Users Are Currently Missing

Based on the competitive landscape, these are the gaps that no current tool addresses well:

**1. Denoised detections with visible confidence.** FIRMS noise is a known operational pain point. Every tool either ignores it (passes raw detections) or uses manual human review (Watch Duty). An ML confidence score shown at the detection level — not hidden in a metrics dashboard — would be novel.

**2. Weather at the fire, not weather as a separate layer.** Every tool that shows weather shows it as a map layer you toggle on separately. Nobody shows wind speed, humidity, and temperature *in context of a specific fire detection* in the detail panel. This is what your weather feature builds. It's a meaningful differentiator.

**3. Hourly-refreshed spread outlook without enterprise cost.** Technosylva has the best spread simulation but it costs tens of thousands of dollars and requires trained operators. Farmonaut has basic spread prediction but is a commercial platform with opaque pricing. There is nothing in between "free but no spread" and "enterprise spread."

**4. Global coverage with regional authority data.** Most tools are either global-but-shallow (FIRMS) or regional-but-authoritative (CWFIS, InciWeb, EFFIS). Your ingestion of multiple authoritative perimeter sources at global scale is structurally unusual.

**5. Integrated archive for researchers.** No free tool has historical replay. ArcGIS has historical data but not a scrubber-style temporal replay. Researchers currently piece together their own archives from FIRMS downloads. Your archive scrubber is a genuine research workflow improvement.

---

## Threats and Risks

**Watch Duty's momentum is the clearest short-term risk.** Their volunteer model and community trust is extremely difficult to replicate. If they add spread prediction or deeper weather integration, they'll crowd out the public tier further. Monitor their roadmap closely.

**Farmonaut is the closest feature overlap.** They have detection + spread + weather in a single commercial platform. They're not well-known in the US fire management community yet, but that could change. Research their data quality and false positive rate claims — "99.5% reduction" with no methodology detail is a marketing number, not a scientific one.

**ESRI can add denoising.** ArcGIS already ingests FIRMS. If ESRI adds ML-based fire/noise classification to their Living Atlas layers, one of your key differentiators moves to a commodity. This is a medium-term risk, not imminent.

**Technosylva could productize downward.** They have a "Pocket Edition" in development — a lighter tactical version. If they price it accessibly and add detection, they become a much more direct competitor.

---

## Strategic Implications

**1. Make the denoiser visible, not invisible.** The single biggest differentiator you have over FIRMS (which everyone else uses as a baseline) is your fire/noise classifier. Right now it's infrastructure. It should be a UI-level concept — users should see the confidence score and understand what it means. This is your "ground truth" claim made concrete.

**2. The weather feature is the right next move.** ArcGIS has weather layers but they're generic map overlays. Nobody shows weather in the context of a specific detected fire. Weather-at-the-fire-location in the detail panel is a genuine product differentiation, especially if you surface the bias-corrected values and show humidity with fire risk context (the <25% / <15% RH thresholds that fire weather people know).

**3. Researchers are your fastest path to credibility.** They already know FIRMS, they already do their own denoising, they already struggle with data quality. Your denoiser + archive replay is exactly what they would pay attention to. Getting one researcher to publish results using your data would do more for your credibility than any amount of marketing.

**4. Don't build a Watch Duty competitor.** Alert-based community wildfire apps is a crowded, trust-driven market. Watch Duty has 8 million users and is a non-profit. That is not a fight worth picking. Stay in the "more information, more analysis" tier and let Watch Duty own the "fast alert" tier.

**5. Your global-deploy architecture is strategically important.** Most tools are US-centric or region-locked. You're built to be globally deployable from day one. Don't let this become just a technical footnote — it's a meaningful differentiation for international researchers and non-US agencies who are currently underserved by everything except FIRMS.

---

## What to Monitor

| Signal | What it Would Mean |
|---|---|
| Watch Duty adds spread prediction | Accelerate your public-facing simplicity; they'll commoditize alerts |
| Farmonaut launches US marketing push | Validate your data quality story; they're the closest feature match |
| ESRI adds ML denoising to Living Atlas | Your detection layer differentiator weakens for enterprise users |
| Technosylva Pocket Edition pricing announced | They're moving downmarket; pressure on your spread feature |
| OroraTech achieves sub-15-min global coverage | Latency ceases to be a differentiation; quality/interpretation matters more |

---

*Research conducted March 2026. Competitive data has a short shelf life in this market — re-evaluate quarterly.*
