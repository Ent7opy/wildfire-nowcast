## Branch & Commit Conventions

To keep history clean and tied to issues, we follow these rules.

### Branch names

Branches must start with the **issue key**, with an optional short name:

- `wn-1`
- `wn-1-initial-structure`
- `wn-7-firms-ingest`
- `wn-13-llm-summaries`

Rules:
- Prefix: `wn-<issue-number>`
- Optional: short, kebab-cased description
- One issue per branch (as much as possible)

### Commit messages

Use conventional prefixes whenever possible:

- `feat:` – new user-facing feature or endpoint  
- `fix:` – bug fix  
- `refactor:` – code changes that don’t change behavior  
- `chore:` – maintenance, tooling, docs, CI, etc.

Examples:
- `feat: add FastAPI health and version endpoints`
- `fix: handle missing FIRMS confidence field`
- `refactor: extract spread model config to settings`
- `chore: add docker compose for api and ui`

On feature branches, small WIP commits are fine as long as they use these prefixes.

### Merging to main

We use **squash merges** into `main`:

- Final squash commit title should reference the issue:
  - `wn-1: project setup & initial structure`
  - `wn-7: implement FIRMS ingestion pipeline`

This keeps `main` history linear, with one commit per issue, and makes it easy to track what changed for each ticket.

## Development environment

Use Python 3.11.x (aligned with `.python-version`) and `uv` for dependency management. See [docs/dev-python-env.md](docs/dev-python-env.md) for the exact commands to create `.venv`, install deps, and run the hello-world API and UI. For the single-command Docker stack (API + UI + Postgres/PostGIS + Redis), follow the instructions in [infra/README.md](infra/README.md).