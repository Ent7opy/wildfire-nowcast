# Database Backup & Recovery

This document outlines the backup and recovery procedures for the Wildfire Nowcast PostgreSQL database. It addresses the audit requirement (issue #301) for a documented recovery procedure.

## Overview

- **Backup format:** Plain SQL, compressed with gzip.
- **Backup location:** `data/backups/` (relative to repository root).
- **Retention:** Automatic pruning of backups older than 7 days (configurable).
- **Tools:** `make backup`, `make restore`, manual scripts.

## Quick Start

### Create a backup

```bash
make backup
```

This runs `scripts/backup_db.sh`, which:
1. Creates a timestamped backup file: `data/backups/wildfire_20260404_120000.sql.gz`
2. Excludes high‑volatility tables (alembic_version, ingest_watermarks, RQ jobs) to keep backups small.
3. Prunes backups older than 7 days.

### List available backups

```bash
make backup-list
```

### Restore from a backup

```bash
make restore BACKUP=data/backups/wildfire_20260404_120000.sql.gz
```

The restore script will:
1. Ask for confirmation (`yes`).
2. Terminate existing database connections (if possible).
3. Restore the SQL dump in a single transaction.

## Manual Scripts

You can also run the scripts directly:

```bash
scripts/backup_db.sh
scripts/restore_db.sh path/to/backup.sql.gz
```

## Automated Backups (Production)

For production deployments, schedule regular backups using cron (or your platform’s scheduler).

### Cron Example

Add to your crontab (runs daily at 2 AM):

```cron
0 2 * * * cd /path/to/wildfire-nowcast && make backup
```

### Environment Variables

The scripts respect the following environment variables (set in `.env` or exported):

- `POSTGRES_USER` (default: `wildfire`)
- `POSTGRES_DB` (default: `wildfire`)
- `BACKUP_DIR` (default: `data/backups`)

## What Gets Backed Up

The backup includes:
- All tables (except excluded ones).
- Schema (tables, indexes, constraints, sequences).
- Data (excluding ephemeral tables).

**Excluded tables (data only):**
- `public.alembic_version` (Alembic migration tracking)
- `public.ingest_watermarks` (ingestion progress; safe to reconstruct)
- `public.rq_jobs`, `public.rq_job_dependents` (Redis‑backed job queue; transient)

These exclusions keep backup size manageable and avoid restoring transient state.

## Recovery Procedure

### Scenario 1: Accidental data loss

1. **Stop the ingest scheduler and API** to prevent new writes:
   ```bash
   docker compose stop ingest_scheduler api worker
   ```
2. **Restore the most recent backup**:
   ```bash
   make restore BACKUP=data/backups/wildfire_$(date -u +"%Y%m%d")_*.sql.gz
   ```
3. **Restart services**:
   ```bash
   docker compose start api worker ingest_scheduler
   ```

### Scenario 2: Database corruption or server failure

If the PostgreSQL container is unusable:

1. **Replace the database volume** (if using Docker volumes):
   ```bash
   docker compose down -v  # WARNING: deletes all data
   docker compose up -d db
   ```
2. **Restore from backup** after the new DB is running:
   ```bash
   make restore BACKUP=path/to/latest-backup.sql.gz
   ```
3. **Run migrations** (if needed):
   ```bash
   make migrate
   ```
4. **Restart the stack**.

### Scenario 3: Point‑in‑time recovery

PostgreSQL’s WAL‑based point‑in‑time recovery is not configured by default. For production deployments, consider enabling continuous archiving and WAL shipping (outside the scope of this document).

## Production Considerations

- **Off‑site storage:** The `data/backups/` directory is local to the server. For disaster recovery, copy backups to cloud storage (S3, GCS, etc.) or a remote server.
- **Encryption:** If backups contain sensitive data, encrypt them before off‑site transfer (e.g., with `gpg`).
- **Monitoring:** Monitor backup success/failure (e.g., log exit codes, send alerts).
- **Testing:** Periodically test restoration on a staging environment.

## Integration with CI/CD

Backup scripts can be integrated into deployment pipelines to create a snapshot before applying migrations. Example GitHub Actions step:

```yaml
- name: Backup before migration
  run: make backup
```

## Troubleshooting

### “pg_dump: error: connection to database failed”

Ensure the `db` service is running:
```bash
docker compose ps db
```

### “Permission denied” when writing backups

Ensure the `data/backups/` directory is writable by the user running the script.

### Restore fails due to active connections

The restore script attempts to terminate existing connections, but if other services (API, worker) are still running, they may hold locks. Stop those services before restoring.

## Related Documentation

- [Database Schema](../api/migrations/)
- [Docker Compose Configuration](../docker-compose.yml)
- [Issue #301: No database backup or documented recovery procedure](https://github.com/Ent7opy/wildfire-nowcast/issues/301)