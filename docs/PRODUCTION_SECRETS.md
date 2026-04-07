# Production Secrets Guide

How to manage secrets for Wildfire Nowcast deployments.

## Secret Inventory

| Variable | Required | Rotation frequency | Notes |
|----------|----------|--------------------|-------|
| `FIRMS_MAP_KEY` | Yes | Yearly / on compromise | NASA FIRMS API key |
| `POSTGRES_PASSWORD` | Yes | Quarterly | Database password |
| `INTERNAL_API_KEY` | Yes (prod) | Quarterly | Protects `/internal/*` endpoints |
| `GEMINI_API_KEY` | No | On compromise | AI assistant; proxied through API |
| `WEBHOOK_SECRET` | No | Quarterly | HMAC signing for outgoing webhooks |
| `NOTIFICATION_SMTP_PASSWORD` | No | Quarterly | Email alert delivery |
| `NOTIFICATION_WEBHOOK_URL` | No | On compromise | Slack/Discord incoming webhook |
| `CDSAPI_KEY` | No | Yearly | Copernicus CDS drought data |
| `LFMC_ECLAND_API_TOKEN` | No | On compromise | LFMC ecLand bearer token |
| `SPREAD_MODEL_CATALOG_SIGNING_KEY` | No | Quarterly | Model catalog HMAC key |
| `DATABASE_URL` | Alt | Quarterly | Full connection string (alternative to `POSTGRES_*`) |

## Injecting Secrets on Railway

Railway is the primary deployment platform. Secrets are set as service variables.

### Per-service variables

1. Open the Railway project dashboard.
2. Select the service (e.g., `api`, `worker`, `ingest_scheduler`).
3. Go to **Variables** tab.
4. Add each secret as a key-value pair.

### Shared variables (recommended)

For secrets used by multiple services (database credentials, `FIRMS_MAP_KEY`):

1. Create a **Shared Variable Group** in Railway project settings.
2. Add the shared secrets there.
3. Link the group to each service that needs it.

### Railway PostGIS reference variables

Railway auto-provisions `DATABASE_URL`, `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE` when you attach a PostGIS plugin. Reference them with `${{ PostGIS.DATABASE_URL }}` syntax in service variables. Do NOT hardcode these.

Set `DB_SSL_REQUIRE=true` on Railway -- asyncpg requires explicit SSL for remote connections.

## Injecting Secrets in Docker Compose (production)

Do NOT use `.env` files in production Docker Compose deployments. Use one of these approaches:

### Option A: Docker secrets (preferred)

```yaml
# docker-compose.prod.yml
services:
  api:
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
      FIRMS_MAP_KEY_FILE: /run/secrets/firms_map_key
    secrets:
      - postgres_password
      - firms_map_key

secrets:
  postgres_password:
    external: true
  firms_map_key:
    external: true
```

Create secrets with:
```bash
echo "your-strong-password" | docker secret create postgres_password -
echo "your-firms-key" | docker secret create firms_map_key -
```

Note: The application currently reads from environment variables, not `*_FILE` vars. If using Docker secrets, add a small entrypoint script that exports file contents to env vars:

```bash
#!/bin/sh
for f in /run/secrets/*; do
  var_name=$(basename "$f" | tr '[:lower:]' '[:upper:]')
  export "$var_name"="$(cat "$f")"
done
exec "$@"
```

### Option B: Environment variables from host

```bash
export POSTGRES_PASSWORD="strong-password-here"
export FIRMS_MAP_KEY="your-key-here"
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Ensure the shell session or systemd unit that runs `docker compose` has the variables set (e.g., via `/etc/environment` or a systemd `EnvironmentFile` that is root-readable only).

## Secret Rotation Procedure

### Database password (`POSTGRES_PASSWORD`)

1. Generate a new password: `openssl rand -base64 32`
2. Update the password in PostgreSQL: `ALTER USER wildfire WITH PASSWORD 'new-password';`
3. Update the variable in Railway (or your deployment target) for **every service** that connects to the database: `api`, `worker`, `ingest_scheduler`, `migrate`, `tiles`.
4. Restart all affected services.
5. Verify with `make health-check` or `curl http://localhost:8000/health`.

### API keys (`FIRMS_MAP_KEY`, `GEMINI_API_KEY`, `CDSAPI_KEY`)

1. Generate or request a new key from the provider.
2. Update the variable in Railway / your deployment target.
3. Restart the affected service(s).
4. Verify the ingest or feature still works (e.g., trigger a test ingest run).
5. Revoke the old key at the provider.

### `INTERNAL_API_KEY`

1. Generate a new key: `openssl rand -hex 32`
2. Update the variable on all services that call `/internal/*` endpoints (typically `ingest_scheduler` and any external automation).
3. Update the variable on the `api` service.
4. Restart affected services.

### `WEBHOOK_SECRET`

1. Generate a new secret: `openssl rand -hex 32`
2. Update on the `api` service (sender) and on the receiving side (Slack app / webhook consumer verification config).
3. Restart the `api` service.

### `NOTIFICATION_SMTP_PASSWORD`

1. Update the password in your email provider.
2. Update the variable on the `api` service.
3. Restart and verify by triggering a test notification.

### `SPREAD_MODEL_CATALOG_SIGNING_KEY`

**Important:** When rotating this key you must update both the key **and** re-sign every catalog entry in a single deploy. The key and its signatures must always match — deploying a new key without re-signing the catalog (or vice-versa) will cause signature verification failures and block model promotion.

1. Generate a new key: `openssl rand -hex 32`
2. Re-sign the model catalog with the new key.
3. Update the `SPREAD_MODEL_CATALOG_SIGNING_KEY` variable on all services (`api`, `worker`).
4. Deploy the re-signed catalog and the new key together.
5. Verify with a test model promotion.

## What NOT To Do

- **Never commit `.env` to git.** It is in `.gitignore` but accidents happen. The repo has a gitleaks pre-commit hook to catch this -- install it with `pre-commit install`.
- **Never log secrets.** Do not print or log environment variables that contain keys or passwords. The codebase uses `${VAR:-}` defaults; make sure log statements do not interpolate secret values.
- **Never embed secrets in Docker images.** Use runtime environment variables or Docker secrets, not build args or `COPY .env`.
- **Never share secrets in Slack, email, or issue trackers.** Use a secrets manager or share via Railway's variable UI.
- **Never use the same secret across environments.** Dev, staging, and production must have independent credentials.
- **Never disable the pre-commit hook.** If gitleaks flags a false positive, add the path or pattern to `.gitleaks.toml` allowlist rather than skipping the hook.

## Pre-commit Hooks

All developers must enable the gitleaks pre-commit hook after cloning the repository. This prevents secrets from being committed to version control.

```bash
pip install pre-commit   # or: brew install pre-commit
pre-commit install
```

This enables the gitleaks hook defined in `.pre-commit-config.yaml`. Every `git commit` will scan staged files for secrets before allowing the commit.

To run manually against all files:

```bash
pre-commit run gitleaks --all-files
```
