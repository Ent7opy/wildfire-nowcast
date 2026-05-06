# Brief 03 — GitHub Adjacency Scout

## Why this exists

People who *build* open-source wildfire tools reveal what's missing in commercial ones. Their users (via stars, issues, discussions) reveal unmet needs. This is the fastest way to triangulate on real demand with a paper trail.

Read `pm/PM_CLAUDE.md` first.

## Goal

Find the active open-source wildfire tooling landscape, what's well-used vs. abandoned, and what users are asking for that nobody's built.

## Method

Use GitHub search + `gh` CLI + WebSearch. Playwright optional if rate-limited.

**Searches (at minimum):**
- `wildfire fire detection topic` search on GitHub
- `FIRMS` as a keyword (many FIRMS clients exist — who uses them?)
- `VIIRS fire` / `MODIS fire` client libraries
- `wildfire dashboard`, `wildfire map`, `fire perimeter`
- `fire spread model`, `ROS rate of spread`, `FARSITE python`
- `wildfire LLM`, `wildfire AI`, `fire agent`
- `evacuation map`, `smoke forecast`
- `earth-tools`, `watchduty` (competitors' public surfaces)

**For each relevant repo, capture:**
- Name, URL, stars, last commit, language
- One-line description
- Open issues count, and the top 3 upvoted issues (these are the loudest feature requests)
- Notable discussions
- Who's using it (look at dependents, README case studies)

**Also worth checking:**
- NASA FIRMS GitHub org if any
- Copernicus EMS / EFFIS public repos
- CAL FIRE / NIFC data repos
- OpenWildfire, fire-atlas, Pyro efforts

## Constraints

- Skip repos <10 stars unless they have unusually active recent issues.
- Skip forks of major projects.
- If a repo is clearly abandoned (>18 mo no commits, no issue activity), note as abandoned, don't dig deep.
- Cite URLs for every claim.

## Output (exact paths)

**1. `pm/research-log/2026-04-21-github.md`** — ≤800 words:
- `## Active projects worth watching` — table: name, URL, stars, focus, status (active / slowing / abandoned)
- `## Top unmet feature requests` — pattern-matched across repos, 5–10 bullets, each with 1–2 issue URLs
- `## Tech stack patterns` — what languages, what data sources, what frameworks dominate
- `## White space` — what you'd expect to exist open-source but doesn't
- `## AI/LLM angle` — who's shipping AI for wildfire open-source, and how (real vs. chatbot-on-top)
- `## Coverage notes`

**2. `pm/signals/2026-04-21-github-raw.md`** — raw bullets: `[issue title / quote] — [URL] — [repo] — [upvotes / status]`.

## Time budget

~30 min.
