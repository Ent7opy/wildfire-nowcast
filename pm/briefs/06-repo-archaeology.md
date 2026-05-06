# Brief 06 — Internal Repo Archaeology

## Why this exists

Before we pivot, we need an honest, read-only map of what's actually in the Wildfire Nowcast repo. What's load-bearing, what's over-built, what's half-finished, what's quietly rotting. The pivot will involve cutting scope aggressively — this brief is the input that tells PM_CLAUDE what can go, what stays, and what decisions are reversible.

Read `pm/PM_CLAUDE.md` first.

## Goal

Produce a ground-truth inventory of the repo's moving parts, their maturity, their maintenance cost, and their relationship to the pre-pivot scope vs. any plausible narrower scope.

## Method

Pure codebase exploration. No web calls. Use Glob / Grep / Read liberally. `gh` for issues / PRs if needed.

**Read at minimum:**
- `CLAUDE.md`, `AGENTS.md`, `SCIENCE_DEBT.md`, `docs/` (everything)
- `README.md` and `Makefile`
- `docker-compose.yml` (service shape)
- `api/` — list routers, note which are actually wired
- `ml/` — denoiser v2 artefacts, spread v2 gate reports, look at `models/` dir for registered models
- `ingest/` — sources and their orchestrator entries
- `ui/src/` — component tree, note any dead routes
- Recent commit history (`git log --oneline -n 200`) — what's been touched lately, what's been quiet
- Open GitHub issues via `gh issue list --state open`
- Any `TODO` / `FIXME` / `XXX` density by folder

**What to assess per subsystem:**
- What it does
- Maturity (mvp_operational / science_grade / experimental / abandoned)
- Solo-maintenance cost (low / med / high) — rough
- Whether it would survive a narrow-scope pivot or be cut
- Hidden assumptions that would break under pivot

**Subsystems to tag explicitly:**
- Denoiser v2 pipeline (train / eval / register / promote)
- Spread forecasting v2
- Archive replay (scrubber + RQ jobs)
- Review queue (human-in-the-loop)
- Notifications (webhook + SMTP)
- AI chat assistant (Gemini)
- Weather integration (in progress per competitive brief)
- Industrial coverage warning
- Model registry and gate report machinery
- UI map stack (Deck.GL + MapLibre)
- Ingest orchestrator

## Constraints

- **Read-only.** No edits, no commits.
- Cite files and line ranges for every claim. Generic claims without a path get cut.
- Be honest about what's impressive-but-over-built. Part of the value of this brief is permission to cut.

## Output (exact paths)

**1. `pm/research-log/2026-04-21-repo.md`** — ≤1000 words (this one can run longer):
- `## Subsystem inventory` — table: subsystem, purpose (1 line), maturity, maintenance cost, survives pivot?, notes
- `## Over-built / low-leverage` — honest list of things that cost more than they return
- `## Load-bearing assumptions` — things later choices depend on (e.g., Postgres+PostGIS, global bbox queries, ONNX runtime, FIRMS MAP_KEY availability)
- `## Half-finished work` — things started but not landed, with links to branches / PRs / TODO comments
- `## Open issues signal` — patterns in open issues that tell us where real user friction is
- `## Recommendation to PM` — 3–5 bullets on what seems genuinely precious vs. what seems like IDC-demo surface area

**2. `pm/signals/2026-04-21-repo-raw.md`** — raw paths + 1-line notes per file / subsystem touched, so PM_CLAUDE can drill in when needed.

## Time budget

~35 min.
