#!/usr/bin/env bash
# Stop or restart Railway services via the GraphQL API.
# Usage: ./scripts/railway_scale.sh up|down [service ...]
#
# "down" removes the latest deployment (stops the service, no compute cost).
# "up"   redeploys the latest deployment (restarts the service).
# Database volumes are preserved in both cases.
#
# If no services are specified, only app services are targeted.
# Pass 'all' to include databases (PostGIS, Redis).
#
# Prerequisites:
#   brew install railway
#   railway login --browserless   (or set RAILWAY_API_TOKEN env var)
#   railway link                  # link to the wildfire-nowcast project

set -euo pipefail

ACTION="${1:-}"
shift || true

if [[ "$ACTION" != "up" && "$ACTION" != "down" ]]; then
  echo "Usage: $0 up|down [service ...]"
  echo ""
  echo "  up    — redeploy services (start)"
  echo "  down  — remove latest deployment (stop, saves memory cost)"
  echo ""
  echo "If no services are listed, app services are toggled."
  echo "Pass 'all' to include databases (PostGIS, Redis)."
  exit 1
fi

# Resolve Railway CLI (may not be on PATH in non-interactive shells)
RAILWAY="$(command -v railway 2>/dev/null || echo /opt/homebrew/bin/railway)"
if [[ ! -x "$RAILWAY" ]]; then
  echo "Error: Railway CLI not found."
  echo "Install it:  brew install railway"
  echo "Then run:    railway login --browserless && railway link"
  exit 1
fi

# Actual service names from Railway dashboard.
DEFAULT_SERVICES=(
  "wildfire-nowcast"
  "ingest"
)

DB_SERVICES=(
  "PostGIS"
  "Redis"
)

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "all" ]]; then
    SERVICES=("${DEFAULT_SERVICES[@]}" "${DB_SERVICES[@]}")
  else
    SERVICES=("$@")
  fi
else
  SERVICES=("${DEFAULT_SERVICES[@]}")
fi

VERB=$( [[ "$ACTION" == "up" ]] && echo "Starting" || echo "Stopping" )
echo "$VERB ${#SERVICES[@]} service(s)..."
echo ""

FAILED=0
for svc in "${SERVICES[@]}"; do
  printf "  %-24s" "$svc"
  if [[ "$ACTION" == "down" ]]; then
    if $RAILWAY down -s "$svc" -y 2>/dev/null; then
      echo "✓  (deployment removed)"
    else
      echo "✗  (may already be stopped, or check service name)"
      FAILED=$((FAILED + 1))
    fi
  else
    if $RAILWAY redeploy -s "$svc" -y 2>/dev/null; then
      echo "✓  (redeploying)"
    else
      echo "✗  (check service name)"
      FAILED=$((FAILED + 1))
    fi
  fi
done

echo ""
if [[ $FAILED -gt 0 ]]; then
  echo "Done with $FAILED error(s)."
  echo "Check service names with: $RAILWAY service status --all"
else
  if [[ "$ACTION" == "down" ]]; then
    echo "Done. Services stopped — no memory charges while down."
  else
    echo "Done. Services redeploying — check status with: $RAILWAY service status --all"
  fi
fi
