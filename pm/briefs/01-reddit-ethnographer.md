# Brief 01 — Reddit Ethnographer

## Why this exists

Wildfire Nowcast was built for a jury, not for users. PM_CLAUDE is pivoting the project toward a narrower, AI-native tool. Reddit is the fastest way to hear unfiltered wildfire-adjacent user voice without running live interviews.

Read `pm/PM_CLAUDE.md` first for doctrine (no fabrication, cite or retract, condensed log + raw signals).

## Goal

Find recurring pains, workarounds, and tool preferences of wildfire-adjacent Reddit users. Surface what's already being said so PM_CLAUDE doesn't have to invent user problems.

## Method

Load Playwright MCP tools via ToolSearch: `query: "playwright browser_", max_results: 20`. Use `old.reddit.com` (unauthenticated, fewer restrictions) for browsing.

**Target subs:**
- r/wildfire
- r/Firefighting
- r/WildlandFire
- r/CAwildfire, r/CAfire
- r/forestry
- r/California (fire-tagged threads only)
- r/australia, r/sydney, r/melbourne (during past fire seasons)
- r/greece, r/portugal (during past fire seasons)
- any sub you discover during browsing that's wildfire-adjacent

**Lookback:** 12 months (2025-04 through 2026-04). Use Reddit's top-of-year and top-of-month sorts to find high-engagement threads fast.

**Signals to capture (in order of priority):**
1. Tool mentions — Watch Duty, FIRMS, InciWeb, ArcGIS, CAL FIRE app, Windy, AirNow, Fires Near Me, EFFIS, local agency apps. Note which, in what context, praise vs. complaint.
2. "I wish there was" / "why can't I see" / "the problem with X is" — these are latent feature requests.
3. Evacuation and citizen-safety questions — what info do people scramble for?
4. Smoke / AQI questions — strong crossover with wildfire.
5. Researcher / academic posts — rare but high signal.
6. Non-US posts — geography gap evidence.

## Constraints

- Do NOT fabricate quotes. Every claim in your log links to a specific thread URL.
- Anonymise usernames.
- Skip memes, trolling, personal-news posts.
- If a sub is dead (<1 post/month), note and skip.
- If Reddit blocks Playwright, try once with different navigation (old.reddit.com, direct URLs), then stop and report — don't waste time on anti-bot workarounds.

## Output (exact paths)

**1. `pm/research-log/2026-04-21-reddit.md`** — ≤800 words, structured:
- `## Top pain patterns` — 5–10 bullets, each: pattern name, rough frequency, one-line summary, 1 representative URL.
- `## Tool mentions (ranked)` — table: tool, sentiment, count, example URL.
- `## Notable quotes` — 3–5, each: quote, context, URL.
- `## Geographic distribution` — rough breakdown: US / CA / AU / EU / other.
- `## Surprises` — anything that contradicts the pre-pivot competitive brief (`docs/competitive-brief.md`) or suggests a problem nobody's flagged.
- `## Coverage notes` — subs checked, subs skipped, anti-bot issues encountered.

**2. `pm/signals/2026-04-21-reddit-raw.md`** — raw evidence list, one bullet per signal: `[quote or paraphrase] — [URL] — [sub] — [date]`. No synthesis. This is what PM_CLAUDE reads when following up on a specific claim.

## Time budget

~25 min of browsing. If you're at 35 min, stop and write up what you have.
