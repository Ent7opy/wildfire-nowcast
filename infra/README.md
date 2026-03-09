# Infrastructure & Local Stack

This `infra` folder documents the local Docker Compose stack (defined in `docker-compose.yml` at the repo root) that runs the FastAPI backend, React UI, Postgres+PostGIS database, and Redis cache with a single command.

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
| React UI | `http://localhost:8501/` | Powered by the React SPA in `ui/`. |
| Postgres+PostGIS | `localhost:5432` | Connection info matches the default env vars below. |
| Redis | `localhost:6379` | Ready for future caching/queue needs. |
| TiTiler | `localhost:8080` | Raster tile server (COG). |
| Vector Tiles | `localhost:7800` | Vector tile server (pg_tileserv). |

## Environment variables

These values can be overridden by defining them in a `.env` file (or your shell) before running Compose.

| Variable | Purpose | Default |
| --- | --- | --- |
| `POSTGRES_USER` | Database user | `wildfire` |
| `POSTGRES_PASSWORD` | Database password | `wildfire` |
| `POSTGRES_DB` | Database name | `wildfire` |
| `POSTGRES_PORT` | Host port mapped to Postgres | `5432` |
| `REDIS_PORT` | Host port mapped to Redis | `6379` |
| `VITE_API_BASE_URL` | API URL resolved by the UI container runtime | `http://api:8000` |
| `VITE_API_PUBLIC_BASE_URL` | API URL used by browser requests | `http://localhost:8000` |
| `VITE_VECTOR_TILES_PUBLIC_BASE_URL` | Public vector tile server URL | `http://localhost:7800` |
| `VITE_FORECAST_REGION_NAME` | Default forecast region fallback | `smoke_grid` |
| `APP_ENV` | Shared indicator for dev vs prod behaviors | `dev` |

## Database & Migrations

The application uses PostgreSQL with PostGIS for spatial data storage. Database schema changes are managed through Alembic migrations.

### Starting the Database

To start just the database service:

```bash
make db-up
# or: docker compose up db -d
```

To stop the database:

```bash
make db-down
# or: docker compose down db
```

### Running Migrations

After starting the database, run migrations to set up the initial schema:

```bash
make migrate
# or: cd api && uv run alembic upgrade head
```

This will:
- Enable the PostGIS extension
- Create a `schema_meta` table for tracking schema metadata
- Apply any pending migrations

### Creating New Migrations

When you need to make schema changes:

1. Modify your database models or write raw SQL in migration files
2. Generate a new migration:

```bash
make revision msg="add user table"
# or: cd api && uv run alembic revision -m "add user table"
```

3. Edit the generated migration file in `api/migrations/versions/`
4. Run the migration: `make migrate`

### Database Connection

The API connects to PostgreSQL using these default environment variables:
- `POSTGRES_HOST=db` (service name in Docker)
- `POSTGRES_PORT=5432`
- `POSTGRES_USER=wildfire`
- `POSTGRES_PASSWORD=wildfire`
- `POSTGRES_DB=wildfire`

Override these in a `.env` file for custom configurations.

### Local Development

For local development outside Docker, set `POSTGRES_HOST=localhost` and ensure PostgreSQL is running on your host machine.

## Notes

- The `api` image installs Python dependencies via `uv`; the `ui` image installs Node dependencies via `npm`.
- The `api` service respects `UVICORN_RELOAD_DIRS=/app/api`, so code changes in `./api` reflected via the bind mount trigger FastAPI's `--reload`.
- The `ui` service runs Vite in dev mode on port `8501` for rapid React feedback.
- Postgres uses an official `postgis/postgis` image plus a named volume (`db_data`). Redis stores data in `redis_data`, keeping your cache state between restarts.
- Use `docker compose logs -f api` (or any service name) to tail logs during development.
