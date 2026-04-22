# Brief 13 — LTA Wildfire Resilience Network launch post drafter

## Why this exists

Agent 10's sharpest finding: users in this space discover tools through peer networks, not ad funnels. One excellent post in the Land Trust Alliance Wildfire Resilience Network newsletter outperforms any consumer funnel. Drafting the post *now* — before the product is built — forces the product to match its audience's voice.

This post will NOT be published until v1 ships. It is a design tool.

**Read in order:**
1. `pm/PM_CLAUDE.md`
2. `pm/north-star.md`
3. `pm/decisions/0005-problem-chosen-a-prime.md`
4. `pm/research-log/2026-04-21-user-archetypes.md` — archetype 1 (conservation NGOs / land trusts) + LTA WRN distribution note
5. https://landtrustalliance.org/resources/connect/field-services/west/wildfire-resilience-network — the actual target venue. Read its tone and the kinds of posts it already publishes.

## Goal

Produce `docs/launch-draft-lta-wrn.md` — a ~900–1200 word draft post pitched at the LTA WRN audience. Tone: colleague writing to peers, not a startup pitching. Non-profit voice. No SaaS-marketing language.

## What the post must do

1. Name the pain a land-trust stewardship lead actually feels ("after the 2017 Tubbs fire we organized fire management across 20,000 acres; monitoring it is still duct-taped") — cite at least one real LTA-member org from agent 10's evidence.
2. Explain why existing tools don't fit (Watch Duty is consumer alerts, EFFIS is institutional, FIRMS is raw, ArcGIS is for GIS teams). Not adversarial — just honest.
3. Introduce Fire Stewardship Agent plainly. What it does; what it doesn't. The "depth over speed" positioning line lives here.
4. Show one concrete example of what a stewardship brief looks like (the paragraph from `north-star.md`, or a parallel one written for an LTA-context AOI).
5. Be specific about the non-profit posture ("free, open-source, donation-funded at most, runs on free-tier infra") and why that matters to a peer audience.
6. Ask for three specific things: (a) try it with one of your own preserve polygons, (b) tell us what's wrong with the brief format — it's designed to match your stewardship workflow, not a generic UI, (c) share with one other land trust if useful.
7. Close with an honest caveat: v1 is narrow; here's what is explicitly not done yet; here's how we'll hear from you.

## What the post must NOT do

- No "revolutionary AI-powered platform" language
- No feature tables competing with Watch Duty / Technosylva / ArcGIS
- No paid-tier hooks or upsell
- No vague claims about "spread prediction" or "early detection"
- No "we've had 8M users" fake scale signals

## Deliverable structure

Write `docs/launch-draft-lta-wrn.md`:

- `## Target venue` — note at top: this is a draft for the LTA WRN newsletter, not to be published until v1 ships
- `## Working title` — one candidate + 2 alternatives
- `## Post body` — ~900–1200 words, publish-ready prose
- `## Design implications fed back` — bullet list of product choices the post's voice implies that the spec / architecture should honour (e.g., "brief must name the authority perimeter source, peer audience expects provenance")
- `## Evidence cited` — every org / stat / quote the post uses, with URLs

## Constraints

- ≤ 1,500 words total including wrapper sections.
- Real, verifiable claims only. No invented stats.
- Voice: colleague, not startup.
- `pm/**` and `docs/` permitted for Write.

## Time budget

~35 min.
