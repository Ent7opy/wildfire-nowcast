## Purpose
This file defines **repo-specific operational rules**.
It does NOT define task scope or planning logic.
When conflicts exist, these rules override generic agent behavior.

# Agent Guidelines

This repository expects agents to use the Make-based workflow as the default for setup, development, testing, and ingestion. Keep changes minimal, safe, and aligned with existing patterns.

## Command conventions
- Prefer `make` targets over ad hoc commands:
  - `make install`, `make db-up`, `make migrate`, `make dev-api`, `make dev-ui`
  - `make ingest-firms`, `make ingest-weather` (pass args via `ARGS="..."`)
  - `make test`, `make lint`, `make clean`
- Run `make help` to discover targets.

## Shell & Python rules
- Do **not** use `python -c` for multi-line code or complex quoting; only trivial one-liners are acceptable.
- For anything more than trivial, write a small script (e.g., `scripts/debug_weather_runs.py`) and run it via `uv run --project ingest python scripts/debug_weather_runs.py`.
- Avoid heredocs or shell tricks; prefer “write a file → run it” to keep commands portable across Windows/WSL/shells.
- Pattern for ad-hoc inspection (e.g., latest completed `weather_runs` entry):
  ```
  from pathlib import Path
  from pprint import pprint
  import sqlalchemy as sa
  from ingest.repository import get_engine

  engine = get_engine()
  with engine.connect() as conn:
      row = (
          conn.execute(
              sa.text(
                  "SELECT * FROM weather_runs "
                  "WHERE status = 'completed' "
                  "ORDER BY created_at DESC LIMIT 1"
              )
          )
          .mappings()
          .first()
      )

  print("Latest completed weather_run:")
  pprint(dict(row) if row else None)
  if row:
      p = Path(row["storage_path"])
      print("storage_path exists?", p.exists())
      if not p.is_absolute():
          p_abs = (Path.cwd() / p).resolve()
          print("resolved path exists?", p_abs.exists())
          print("resolved path:", p_abs)
  ```
- If a terminal command fails twice due to syntax/quoting, stop and switch to the script approach instead of iterating on the one-liner.

## Safety and git hygiene
- Do **not** use destructive git commands (`git reset --hard`, `git checkout -- .`).
- Do not revert or overwrite user changes. Avoid broad refactors unless requested.
- Do not amend commits unless explicitly asked.

## Editing standards
- Default to ASCII; add Unicode only if already present and justified.
- Use `apply_patch` for single-file edits when practical.
- Add only concise, helpful comments; avoid noisy or obvious remarks.
- Avoid duplicating docs; link to canonical sources instead.

## Testing and linting
- Run relevant checks when touching code:
  - `make lint` (Ruff), `make test` (API + UI) as appropriate.
- Report what was run and the results.

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

## Scope and communication
- Ask for clarification when requirements are ambiguous.
- If an operation is long/expensive, call it out before running.

## STOP conditions
- If unsure which make target to use → stop and ask
- If data outputs would be large or long-running → confirm before running
- If a command fails twice due to quoting/shell issues → switch to script approach
