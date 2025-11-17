# Infrastructure & Local Stack

This `infra` folder documents the local Docker Compose stack (defined in `docker-compose.yml` at the repo root) that runs the FastAPI backend, Streamlit UI, Postgres+PostGIS database, and Redis cache with a single command.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) or the equivalent Docker Engine for your OS.
- [Docker Compose](https://docs.docker.com/compose/) v2+ (bundled with Docker Desktop).

## Running the local stack

1. (Optional) Create a `.env` file in the repo root to override the defaults below. Any undefined variables fall back to the values shown in the table.
2. From the repo root, build and start all services:
   ```bash
   docker compose up --build
   ```
3. Wait until each service reports that it is healthy (you should see `api`, `ui`, `db`, and `redis` listed without `starting` status).

To stop and remove the containers (data volumes are preserved unless you pass `-v`):

```bash
docker compose down
```

To reset the Postgres data volume (use with caution):

```bash
docker compose down -v
```

## Service endpoints

| Service | URL (host) | Notes |
| --- | --- | --- |
| FastAPI backend | `http://localhost:8000/health` | Exposes `/health` and any future API endpoints. |
| Streamlit UI | `http://localhost:8501/` | Powered by the Streamlit app in `ui/`. |
| Postgres+PostGIS | `localhost:5432` | Connection info matches the default env vars below. |
| Redis | `localhost:6379` | Ready for future caching/queue needs. |

## Environment variables

These values can be overridden by defining them in a `.env` file (or your shell) before running Compose.

| Variable | Purpose | Default |
| --- | --- | --- |
| `POSTGRES_USER` | Database user | `wildfire` |
| `POSTGRES_PASSWORD` | Database password | `wildfire` |
| `POSTGRES_DB` | Database name | `wildfire` |
| `POSTGRES_PORT` | Host port mapped to Postgres | `5432` |
| `REDIS_PORT` | Host port mapped to Redis | `6379` |
| `API_BASE_URL` | Service name used by the UI | `http://api:8000` |
| `APP_ENV` | Shared indicator for dev vs prod behaviors | `dev` |

## Notes

- Both the `api` and `ui` images install dependencies via `uv` and share the same `app/.venv` created during build.
- The `api` service respects `UVICORN_RELOAD_DIRS=/app/api`, so code changes in `./api` reflected via the bind mount trigger FastAPI’s `--reload`.
- The `ui` service enables `STREAMLIT_SERVER_RUN_ON_SAVE` for rapid Streamlit feedback.
- Postgres uses an official `postgis/postgis` image plus a named volume (`db_data`). Redis stores data in `redis_data`, keeping your cache state between restarts.
- Use `docker compose logs -f api` (or any service name) to tail logs during development.

