# Brief 05 — AI-Native Leverage Scout

## Why this exists

Vanyo has explicitly framed Wildfire Nowcast as an AI-first project. A chatbot bolted onto a map is not AI-native. We need to find where bleeding-edge AI infrastructure (LLMs, agents, tool-use, long-context, structured output, MCP, workflow orchestration, RAG, small-model inference) genuinely changes what's possible for wildfire tooling — the economics, the UX, or the capability ceiling.

Read `pm/PM_CLAUDE.md` first.

## Goal

Articulate 5–8 candidate AI-native leverage points for wildfire tooling, ranked by plausibility and differentiation. Each must pass: *"if I took the AI layer out, would the tool still be roughly as useful?"* — if yes, it's decorative, not native.

## Method

WebSearch + repo reading + reasoning.

**Starting points:**
- How climate/weather AI startups are using LLMs (Salient Predictions, Atmo, Climate AI, Brightband, Jua.ai, ECMWF AIFS, NVIDIA FourCastNet)
- AI agents shipping in adjacent ops domains (incident response, cybersec triage, on-call) — transferable patterns
- Natural-language interfaces over geospatial data (Earth Copilot, Bunting Labs, Placemark)
- Long-context synthesis — can you point an agent at a fire and have it read perimeters + weather + news + historical analogues and produce a situational brief?
- Agentic monitoring — a user sets an AOI; the agent watches and only wakes the user when something decision-relevant changes
- Anomaly triage — AI grading FIRMS detections against context (known industrial sources, historical false-positive zones, diurnal patterns)
- RAG over historical fire data — "show me fires under similar weather conditions in the last 20 years and what happened" for researchers
- Small-model inference at the edge — running classifiers cheaply at global scale
- Tool-use / MCP — letting users (or their agents) compose queries across authorities (NIFC + CWFIS + EFFIS + FIRMS) in one call

**Categories to evaluate each idea against:**
1. *Capability unlock* — was this impossible pre-LLM?
2. *Economic unlock* — does it make something 10× cheaper or 10× faster?
3. *UX unlock* — does it change who can use the tool?
4. *Defensibility* — is the advantage durable, or does it evaporate when OpenAI/Anthropic ship the next model?

## Constraints

- Ideas must be specific enough that a spec agent could brief a build on them. "AI-powered insights" does not qualify.
- Each idea must name: the user, the moment of use, the inputs, the output, and the AI capability it requires.
- Be honest about what's chatbot-on-map cosplay.

## Output (exact paths)

**1. `pm/research-log/2026-04-21-ai-leverage.md`** — ≤900 words:
- `## Candidate leverage points` — 5–8 cards, each structured as: name, user, moment, inputs, output, AI capability required, category unlocks (capability / economic / UX / defensibility), one-line objection.
- `## What's already shipped elsewhere` — who's doing this for adjacent domains, URLs.
- `## What would require Vercel-ish infra specifically` — Workflow DevKit for durable long-running monitoring, AI Gateway for multi-model, etc. Worth calling out since Vanyo mentioned AI-first infra.
- `## Ranked take` — author's opinion: which 2 ideas are most promising and why.
- `## Anti-patterns to avoid` — things that look AI-native but are decorative.

**2. `pm/signals/2026-04-21-ai-leverage-raw.md`** — URLs + one-liners for all referenced examples.

## Time budget

~35 min. This one is thinkier than the ethnography agents.
