#!/bin/sh
set -e
# Run Alembic migrations. Use from repo root (e.g. Railway preDeployCommand).
# If your service root is api/, use: uv run alembic upgrade head
cd "$(dirname "$0")/.."
cd api && uv run alembic upgrade head
