# Ralph Role-Specific Guidelines

## [COMMON]
### Purpose
This file defines **repo-specific operational rules** (how to run/verify/work safely).
It does NOT define task scope or planning logic.
When conflicts exist, these rules override generic agent behavior.

### Ralph Loop Integration
- The orchestrator inbox JSON (`.ralph/inbox/<role>.json`) is the **authoritative input** for that run.
- The only machine-readable deliverable is the outbox JSON (`.ralph/out/<role>.json`).

### Execution honesty
- Never claim you ran anything unless you did.
- Use **Ran:** / **Would run:** in verification lists.

### Safety and git hygiene
- Do **not** use destructive git commands (`git reset --hard`, `git checkout -- .`).
- Avoid broad refactors unless required for correctness or explicitly requested.
- Do not revert/overwrite user changes.
- Do not amend commits unless explicitly instructed.

### Non-interactive STOP conditions
If any of these are true, **do not proceed**. Instead, report it as a blocker in your outbox JSON with the minimal next step.
- Unsure which make target to use (or none exists)
- Operation is long-running / expensive / produces large data outputs
- Repo conventions are unclear for a high-impact change (schema, public API, storage paths, jobs/infra)
- A command fails twice due to quoting/shell issues (switch to script approach or mark blocked)

## [CODING]
(Applies to Worker and Bug Fixer)

### Command conventions
- Prefer `make` targets over ad hoc commands:
  - `make install`, `make db-up`, `make migrate`, `make dev-api`, `make dev-ui`
  - `make ingest-firms`, `make ingest-weather` (pass args via `ARGS="..."`)
  - `make test`, `make lint`, `make clean`
- Run `make help` to discover targets.

### Shell & Python rules
- Avoid fragile shell quoting and one-liners.
- Do **not** use `python -c` for multi-line code or complex quoting (only trivial one-liners).
- For anything non-trivial: write a small script (e.g., `scripts/debug_weather_runs.py`) and run:
  - `uv run --project ingest python scripts/debug_weather_runs.py`
- Avoid heredocs/shell tricks; prefer “write a file → run it” for portability across Windows/WSL/shells.
- If a terminal command fails twice due to syntax/quoting, stop and switch to the script approach instead of iterating.

### Editing standards
- Default to ASCII; add Unicode only if already present and justified.
- Prefer small, targeted diffs.
- Add only concise, helpful comments; avoid noisy or obvious remarks.
- Avoid duplicating docs; link to canonical sources instead.

### Migrations and database
- Apply migrations via `make migrate`.
- Create migrations via `make revision msg="..."` under `api/migrations/versions/`.
- Ensure migrations are reversible (upgrade/downgrade).

### Ingest conventions
- Follow existing patterns in `ingest/` (e.g., `weather_ingest.py`, `weather_repository.py`, `config.py`).
- Track ingest runs via the appropriate tables (`ingest_batches`, `weather_runs`) and update statuses on failure.
- Weather outputs live under `data/weather/...` as NetCDF following current naming.

## [REVIEW]
(Applies to Reviewer)

### Testing and linting
- Run relevant checks when touching code:
  - `make lint` and/or `make test` as appropriate.
- Report what was run and the results (Ran/Would run).

### Configuration and environment
- Keep `.env` in the repo root for shared settings (`FIRMS_MAP_KEY`, `POSTGRES_*`, etc.).
- Weather ingest requires ecCodes for cfgrib; note when relevant.
- Never hardcode secrets; use env/config.

### Docs
- `README.md` is the single-source quickstart for commands.
- `docs/SETUP.md` stays minimal (prereqs, .env, Docker/WSL, troubleshooting).
- Detailed ingest specifics belong in focused docs (e.g., `docs/ingest_weather.md`).
- Update docs when changing CLIs, make targets, or visible workflows.

## [END]
