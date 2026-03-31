# Wildfire Nowcast — User Research Plan
**Version:** 1.0
**Date:** March 28, 2026
**Research owner:** TBD

---

## Research Question

> How does Wildfire Nowcast fit into the real-world workflows of incident commanders and fire researchers — and where does it break down?

---

## Objectives

1. Map the end-to-end decision workflow for each segment during an active fire event.
2. Identify what data sources and tools they currently use, and what job Wildfire Nowcast would displace or augment.
3. Understand the conditions under which they trust or distrust automated fire detection data.
4. Pinpoint workflow gaps where the product either doesn't reach them or doesn't give them what they need at the moment they need it.
5. Surface unspoken constraints: organizational approval gates, device restrictions, connectivity limitations, and chain-of-command dynamics that affect adoption.

---

## Method: Semi-structured User Interviews

Interviews are the right choice here. Workflow questions require depth and follow-up; surveys would only tell you what people think they do, not what they actually do.

**Format:** 60-minute remote or in-person sessions (in-person preferred for incident commanders during field season)
**Recording:** With consent — transcript + highlights for synthesis
**Session structure:**

| Segment | Duration | Purpose |
|---------|----------|---------|
| Warm-up | 5 min | Establish rapport, confirm role and context |
| Current workflow | 15 min | Walk me through your last active fire event, start to finish |
| Tool inventory | 10 min | What are you looking at? When? On what device? |
| Concept reaction | 15 min | Show the live product or a recorded demo |
| Fit & gaps | 10 min | Where would this live in your workflow? What's missing? |
| Wrap-up | 5 min | Open floor, anything we should have asked |

---

## Participants

### Segment A — Incident Commanders (6–8 participants)
People who make real-time suppression decisions: where to deploy resources, when to call evacuations, when to hold or pull back crews.

**Screener criteria:**
- Active or recently retired Type 1 or Type 2 Incident Commander, or Operations Section Chief
- Has commanded at least one major fire event (100+ acres) in the past 3 years
- Uses digital tools in the field (not exclusively paper/radio)
- Mix of agency types: USFS, CAL FIRE, BLM, state/provincial, international

**Where to find them:** Agency liaison contacts, NWCG training alumni, interagency dispatch centers, wildfire conference networks (e.g., IAWF, FIRESTORM)

### Segment B — Researchers / Analysts (5–7 participants)
People who use fire detection data for scientific or planning purposes: fire behavior modeling, post-fire assessment, risk mapping, policy analysis.

**Screener criteria:**
- Uses FIRMS, VIIRS, MODIS, GOES, or similar satellite fire data regularly
- Works in a research institution, university, government agency, or NGO
- Has a workflow that requires near-real-time or hourly fire data (not just annual summaries)
- Mix of domains: fire behavior modeling, ecology, emergency management research, climate

**Where to find them:** USFS Research Stations, university fire science labs, NIFC, CWFIS, NASA FIRMS user community

---

## Interview Guide

### Segment A — Incident Commanders

**Warm-up**
- Tell me about your role. What kind of fires do you typically work?
- What was the most recent major fire you commanded or worked?

**Current workflow**
- Walk me through what a typical operational period looks like from when you first get a report to when you're making resource deployment decisions.
- When you're trying to understand where the fire is right now — at this minute — what do you look at?
- What does your decision cycle look like? How often are you re-evaluating your picture of the fire?
- What do you do when your sources disagree with each other?

**Tool inventory**
- What's on your screen during an active incident? Walk me through the layers.
- What tools do you use that you'd consider non-negotiable? What would you drop first if you had to cut one?
- Are you working from a fixed command post, a vehicle, or in the field? What device?
- Do you have connectivity issues in the field? How do you plan around them?

**Concept reaction** *(show live product or demo)*
- What's your first read on this?
- What would you use this for? What would you ignore?
- When during an operational period would you open this — and when wouldn't you?
- What would make you distrust what you're seeing?

**Fit & gaps**
- If this existed two years ago, would it have changed any decision you made? Which one?
- What would you need to see before you'd recommend this to your ops team?
- What's the one thing that would make this unusable in your context?

---

### Segment B — Researchers / Analysts

**Warm-up**
- What's your research focus, and how does fire detection data feed into it?
- How often are you pulling fire location data, and what's your typical lag tolerance?

**Current workflow**
- Walk me through how you currently acquire, process, and use fire detection data for a typical project.
- What pre-processing do you do before you trust a raw detection?
- What's your biggest time sink in getting from raw data to something you can analyze?
- What do you do when the data has gaps — cloud cover, sensor gaps, latency?

**Tool inventory**
- What data sources are you pulling from today? (FIRMS, GOES, VIIRS, MODIS, GOES-R, NIFC perimeters, other?)
- Do you build your own denoising or filtering? If so, what does it look like?
- Are you consuming data programmatically, through a UI, or both?
- What downstream tools consume your fire detection output?

**Concept reaction** *(show live product or demo)*
- From a data quality standpoint, what questions would you need answered before you'd use this in a paper or model?
- How does the denoising compare to what you'd do yourself? Where do you trust it, where are you skeptical?
- What's missing that you'd need for your use case?

**Fit & gaps**
- Is there a specific workflow step where this would save you the most time?
- What format would you need the data in to consume it programmatically?
- What metadata do you need alongside the detection (uncertainty, sensor, timestamp precision, version)?
- If this had an API, what would your first call look like?

---

## Timeline

| Week | Activities |
|------|-----------|
| 1 | Finalize screener, recruit via agency and research contacts, schedule sessions |
| 2 | Conduct 4–6 sessions (mix of segments) |
| 3 | Conduct remaining 5–7 sessions, begin affinity mapping |
| 4 | Complete synthesis, draft report, validate top findings with 1–2 participants |

**Total duration:** ~4 weeks
**Estimated sessions:** 11–15

---

## Analysis Approach

**During fieldwork:**
- Take structured notes using the interview guide as a skeleton
- Flag high-signal quotes and moments in real time
- After each session: 15-minute debrief to capture immediate impressions before they fade

**Synthesis (week 3–4):**
- Affinity mapping: group observations across participants into themes
- Jobs-to-be-done framing: what is each segment hiring Wildfire Nowcast to do?
- Workflow journey map: where does the product fit (or not) in the operational or research timeline?
- Gap analysis: unmet needs that the current product doesn't address

---

## Deliverables

1. **Research report** — objectives, method, key findings per segment, workflow maps, prioritized gaps, recommendations
2. **Interview transcripts** — anonymized, stored securely
3. **Highlight reel** — 5–8 key quotes or moments that illustrate the most important findings
4. **Actionable backlog items** — specific product gaps translated into potential features or fixes, ranked by frequency and severity

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Incident commanders are hard to schedule during fire season | Recruit off-season or reach retired ICs; target March–May window |
| Researchers may have niche domain knowledge that skews feedback | Screen for direct fire detection data use, not just fire adjacency |
| Participants may describe ideal behavior, not actual behavior | Use "walk me through your last event" to anchor in real incidents |
| Confidentiality concerns from agency personnel | Make clear research is informational only, no operational data shared |
