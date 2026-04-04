#!/bin/bash
# Restore PostgreSQL database from a backup file.
# WARNING: This will replace the current database content.

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

if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    echo "Available backups:"
    ls -1 "${BACKUP_DIR}/${POSTGRES_DB}_"*.sql.gz 2>/dev/null | head -10
    exit 1
fi

BACKUP_FILE="$1"
if [ ! -f "${BACKUP_FILE}" ]; then
    echo "Error: backup file '${BACKUP_FILE}' not found"
    exit 1
fi

echo "Restoring database '${POSTGRES_DB}' from ${BACKUP_FILE}..."
echo "WARNING: This will overwrite existing data in the database."
read -p "Are you sure? (type 'yes' to continue): " confirm
if [ "${confirm}" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Drop existing connections (optional, may fail if other services are running)
echo "Terminating existing connections..."
docker compose exec -T db \
    psql --username="${POSTGRES_USER}" --dbname="${POSTGRES_DB}" \
        -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid();" \
        2>/dev/null || true

# Restore the backup
gunzip -c "${BACKUP_FILE}" | \
    docker compose exec -T db \
        psql --username="${POSTGRES_USER}" --dbname="${POSTGRES_DB}" \
            --quiet --single-transaction

echo "✅ Database restored successfully."