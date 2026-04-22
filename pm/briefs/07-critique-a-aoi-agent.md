# Brief 07 — Adversarial critic: candidate A (non-profit AOI agent)

## Why this exists

Phase 1 recommended candidate A (fire-aware AOI agent for anyone with a place to protect). Before we commit, we need an agent whose explicit job is to **kill this idea**. Confirmation bias is the enemy of solo-operator scoping.

**Read first:**
1. `pm/PM_CLAUDE.md` — operating doctrine, now with non-profit + free-tier constraints
2. `pm/decisions/0002-phase-1-synthesis.md` — Phase 1 findings
3. `pm/decisions/0003-nonprofit-and-free-infra-constraints.md` — binding constraints
4. `pm/backlog.md` § "A — Fire-aware AOI agent" — current candidate definition
5. `pm/research-log/2026-04-21-ai-leverage.md` — especially L1 (author's pro-argument)

## Goal

Produce the strongest possible case that candidate A is a bad choice. Steel-man the objection. Force PM_CLAUDE to either drop it or to respond substantively.

## Attack vectors to genuinely press on

1. **User reality.** Do the named non-profit archetypes (conservation NGOs, Indigenous fire stewards, protected-area managers, small municipalities, WUI homeowners, diaspora, journalists) actually have this pain, or are they hypothetical? Find evidence one way or the other.
2. **Watch Duty drift.** Watch Duty is US/CA today. Could they expand internationally before we ship? Does John Mills' "no sale, no expansion" stance actually hold under a major board change or major funding round?
3. **AOI-agent commoditization.** What if OpenAI / Anthropic / Google ship a generic "geofenced alert agent" as part of their assistant product? How defensible is the wildfire-specialized version?
4. **FIRMS dependency risk.** If NASA changes FIRMS access terms (auth, quotas, paid tier), the whole product dies. How real is this risk? Any historical precedent?
5. **Engagement cliff.** Fire seasons are seasonal. 10 months/year of low use = users churn out of the notification list. Does the product survive its own off-season?
6. **Hallucinated value.** The "reasoned brief" AI layer adds latency and LLM cost. Is a threshold-triggered SMS just as good for 90% of users? If yes, AI is decorative.
7. **Free-tier infra reality.** Can we honestly deliver the agent + AOI cron + LLM reasoning at $0–10/month including LLM calls? Sketch the math.
8. **Trust gap.** Non-profit doesn't equal trusted. Why would a Greek mayor or a Colorado homeowner trust a solo developer's side project over EFFIS / Watch Duty / 112?

## Constraints

- You are **NOT** a devil's-advocate cosplay. Every attack must cite evidence (URL, file path, public signal) or an explicit reasoning chain showing how a named incumbent would crush this.
- Hypothetical "could happen" objections are fine **only if** you name a plausible trigger (e.g., "Watch Duty raised $X Series A in 2024 per CrunchBase — international expansion risk is real").
- If the attack requires an assumption Vanyo could reasonably dispute, call that out.

## Output

**`pm/research-log/2026-04-21-critique-a.md`** — ≤1000 words, structured:
- `## Thesis being attacked` — one-sentence restatement of candidate A
- `## Strongest attacks` — top 5, each: attack name, evidence / reasoning, severity (fatal / serious / manageable), what would need to be true for candidate A to survive
- `## Free-tier infra reality check` — honest math on whether the $0–10/month target is achievable for a reasonable number of concurrent AOIs (assume 50 users, 100 AOIs, notification frequency ~1/week/AOI)
- `## Objections I could not substantiate` — honesty section; attacks that didn't hold up under evidence
- `## Net verdict` — kill / rescope / proceed-with-conditions

**`pm/signals/2026-04-21-critique-a-raw.md`** — supporting evidence bullets with URLs.

## Time budget

~35 min. This is a thinky brief. Take the time to press.
