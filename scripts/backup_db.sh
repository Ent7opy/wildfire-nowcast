#!/bin/bash
# Backup PostgreSQL database for Wildfire Nowcast.
# Requires Docker Compose and the 'db' service running.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# Load environment variables from .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

POSTGRES_USER="${POSTGRES_USER:-wildfire}"
POSTGRES_DB="${POSTGRES_DB:-wildfire}"
BACKUP_DIR="${BACKUP_DIR:-data/backups}"
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/${POSTGRES_DB}_${TIMESTAMP}.sql.gz"

# Ensure backup directory exists
mkdir -p "${BACKUP_DIR}"

echo "Backing up database '${POSTGRES_DB}' as user '${POSTGRES_USER}'..."

# Run pg_dump inside the db container, compress on the fly
docker compose exec -T db \
    pg_dump \
        --username="${POSTGRES_USER}" \
        --dbname="${POSTGRES_DB}" \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges \
        --no-comments \
        --quote-all-identifiers \
        --exclude-table-data='public.alembic_version' \
        --exclude-table-data='public.ingest_watermarks' \
        --exclude-table-data='public.rq_jobs' \
        --exclude-table-data='public.rq_job_dependents' \
        | gzip > "${BACKUP_FILE}"

# Verify backup was created
if [ -s "${BACKUP_FILE}" ]; then
    BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
    echo "✅ Backup created: ${BACKUP_FILE} (${BACKUP_SIZE})"
else
    echo "❌ Backup failed: output file is empty or missing"
    exit 1
fi

# Optional: prune old backups (keep last 7 days)
if command -v find &> /dev/null; then
    find "${BACKUP_DIR}" -name "${POSTGRES_DB}_*.sql.gz" -type f -mtime +7 -delete
    echo "🗑️  Pruned backups older than 7 days"
fi