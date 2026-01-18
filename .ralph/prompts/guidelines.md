## Purpose
This file defines **repo-specific operational rules** (how to run/verify/work safely).
It does NOT define task scope or planning logic.
When conflicts exist, these rules override generic agent behavior.

---

# Ralph Loop Integration (applies to ALL roles)

## Source of truth
- The orchestrator inbox JSON (`.ralph/inbox/<role>.json`) is the **authoritative input** for that run.
- The only machine-readable deliverable is the outbox JSON (`.ralph/out/<role>.json`).

## Role boundaries (hard rules)
- **Only the Committer role may run `git commit`.**
- Other roles must **not** commit, amend, or rewrite history.
- If you would normally “ask/confirm”, you must instead:
  - stop work,
  - mark `status="blocked"` (or `verdict="changes_requested"`),
  - include the minimal next step in the outbox.

## Execution honesty
- Never claim you ran anything unless you did.
- Use **Ran:** / **Would run:** in verification lists.

---

# Repo Guidelines

This repository expects agents to use the **Make-based workflow** as the default for setup, development, testing, and ingestion. Keep changes minimal, safe, and aligned with existing patterns.

## Command conventions
- Prefer `make` targets over ad hoc commands:
  - `make install`, `make db-up`, `make migrate`, `make dev-api`, `make dev-ui`
  - `make ingest-firms`, `make ingest-weather` (pass args via `ARGS="..."`)
  - `make test`, `make lint`, `make clean`
- Run `make help` to discover targets.

## Shell & Python rules
- Avoid fragile shell quoting and one-liners.
- Do **not** use `python -c` for multi-line code or complex quoting (only trivial one-liners).
- For anything non-trivial: write a small script (e.g., `scripts/debug_weather_runs.py`) and run:
  - `uv run --project ingest python scripts/debug_weather_runs.py`
- Avoid heredocs/shell tricks; prefer “write a file → run it” for portability across Windows/WSL/shells.
- If a terminal command fails twice due to syntax/quoting, stop and switch to the script approach instead of iterating.

## Safety and git hygiene
- Do **not** use destructive git commands (`git reset --hard`, `git checkout -- .`).
- Avoid broad refactors unless required for correctness or explicitly requested.
- Do not revert/overwrite user changes.
- Do not amend commits unless explicitly instructed.

## Editing standards
- Default to ASCII; add Unicode only if already present and justified.
- Prefer small, targeted diffs.
- Add only concise, helpful comments; avoid noisy or obvious remarks.
- Avoid duplicating docs; link to canonical sources instead.

## Testing and linting
- Run relevant checks when touching code:
  - `make lint` and/or `make test` as appropriate.
- Report what was run and the results (Ran/Would run).

## Migrations and database
- Apply migrations via `make migrate`.
- Create migrations via `make revision msg="..."` under `api/migrations/versions/`.
- Ensure migrations are reversible (upgrade/downgrade).

## Configuration and environment
- Keep `.env` in the repo root for shared settings (`FIRMS_MAP_KEY`, `POSTGRES_*`, etc.).
- Weather ingest requires ecCodes for cfgrib; note when relevant.
- Never hardcode secrets; use env/config.

## Ingest conventions
- Follow existing patterns in `ingest/` (e.g., `weather_ingest.py`, `weather_repository.py`, `config.py`).
- Track ingest runs via the appropriate tables (`ingest_batches`, `weather_runs`) and update statuses on failure.
- Weather outputs live under `data/weather/...` as NetCDF following current naming.

## Docs
- `README.md` is the single-source quickstart for commands.
- `docs/SETUP.md` stays minimal (prereqs, .env, Docker/WSL, troubleshooting).
- Detailed ingest specifics belong in focused docs (e.g., `docs/ingest_weather.md`).
- Update docs when changing CLIs, make targets, or visible workflows.

---

# Non-interactive STOP conditions (Ralph-safe)

If any of these are true, **do not proceed**. Instead, report it as a blocker in your outbox JSON with the minimal next step.

- Unsure which make target to use (or none exists)
- Operation is long-running / expensive / produces large data outputs
- Repo conventions are unclear for a high-impact change (schema, public API, storage paths, jobs/infra)
- A command fails twice due to quoting/shell issues (switch to script approach or mark blocked)
