# Brief 02 — X / Twitter Ethnographer

## Why this exists

Fire Twitter is where ICs, PIOs, fire meteorologists, researchers, and serious amateur weather watchers talk in real time during events. Signal-to-noise is higher than Reddit for *professional* wildfire voice, but access is harder.

Read `pm/PM_CLAUDE.md` first for doctrine.

## Goal

Surface what fire professionals and serious researchers *quote-tweet as useful* vs. *quote-tweet as broken*. Identify tools and data sources they rely on, and workflows they complain about.

## Method

Two-track:

**Track A — Playwright browsing.** Load Playwright via ToolSearch. Try `x.com` / `twitter.com` unauthenticated. Nitter instances (e.g., nitter.net, nitter.privacydev.net) are good fallbacks; check one is live first.

**Track B — Google site search as fallback.** `site:twitter.com wildfire "watch duty"`, `site:twitter.com FIRMS frustrating`, `site:x.com "fire weather" tools`, etc. WebSearch tool is fine for this.

**Target clusters:**
- Fire meteorologists and fire-weather accounts (IMETs, NWS fire-weather offices, private fire-wx forecasters)
- Incident PIOs and interagency dispatch accounts
- Active-fire researchers (UC, CSU, UCSD WIFIRE, USFS PSW / RMRS)
- Citizen weather / storm-chaser accounts that cover fires
- International: Australian RFS / CFA / NSWRFS voices; European fire researchers

**What to look for (priority order):**
1. Tool praise / criticism, with screenshots ideally
2. "The thing I actually need during a fire is…" style threads
3. Real-time frustrations during named events (LA 2025, Australian 2025 season, Greece/Rhodes 2025)
4. Data-access complaints (FIRMS delays, GOES quirks, perimeter lag)
5. AI-in-wildfire takes — positive and skeptical

## Constraints

- No fabrication. Link every claim to a tweet URL (or archive URL if original deleted).
- Anonymise handles unless the account is an official org (agency, university).
- If both Playwright and nitter are blocked, pivot entirely to Track B and say so.

## Output (exact paths)

**1. `pm/research-log/2026-04-21-twitter.md`** — ≤800 words:
- `## Professional voice patterns` — what fire pros agree / disagree on re. tools and data
- `## Tool mentions (ranked)` — table (tool, sentiment, count, example URL)
- `## Named-event case studies` — 2–3 fire events where the online discourse was heavy; what tools were visible, what broke
- `## AI-in-wildfire takes` — what serious practitioners say about AI tooling here
- `## Notable quotes` — 3–5, linked
- `## Coverage notes` — which track worked, which accounts were richest, what was blocked

**2. `pm/signals/2026-04-21-twitter-raw.md`** — raw bullets: `[quote or paraphrase] — [URL] — [handle/org] — [date]`.

## Time budget

~25 min. Stop at 35.
