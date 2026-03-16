#!/bin/sh
set -e
# Run Alembic migrations from repo root (e.g. Railway preDeployCommand).
# Use explicit -c so Alembic finds script_location regardless of uv's cwd.
cd "$(dirname "$0")/.."
uv run --project api alembic -c api/alembic.ini upgrade head
