# Infrastructure

Local Docker Compose stack defined in `docker-compose.yml` at the repo root.

## Start / stop

```bash
docker compose up --build      # build and start all services
docker compose down            # stop (volumes preserved)
docker compose down -v         # stop + delete volumes (destructive)
```

## Service endpoints

| Service | Host URL | Notes |
|---------|----------|-------|
| FastAPI | `http://localhost:8000` | REST API + health endpoints |
| React UI | `http://localhost:8501` | Vite dev server |
| PostgreSQL+PostGIS | `localhost:5433` | Default DB credentials below |
| Redis | `localhost:6379` | Cache + job queue |
| TiTiler | `http://localhost:8080` | COG raster tile server |
| pg_tileserv | `http://localhost:7800` | Vector tile server |

## Environment variables

Override in `.env` at repo root before running Compose.

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_USER` | `wildfire` | DB user |
| `POSTGRES_PASSWORD` | `wildfire` | DB password |
| `POSTGRES_DB` | `wildfire` | DB name |
| `POSTGRES_PORT` | `5433` | Host port mapped to Postgres container |
| `REDIS_PORT` | `6379` | Host port mapped to Redis |
| `VITE_API_BASE_URL` | `http://api:8000` | API URL for container-to-container |
| `VITE_API_PUBLIC_BASE_URL` | `http://localhost:8000` | API URL for browser requests |
| `VITE_VECTOR_TILES_PUBLIC_BASE_URL` | `http://localhost:7800` | Public vector tile URL |
| `VITE_FORECAST_REGION_NAME` | `smoke_grid` | Default forecast region |
| `APP_ENV` | `dev` | dev vs prod behavior flag |

## Database migrations

```bash
make db-up       # start DB only
make migrate     # apply Alembic migrations
make revision msg="description"   # create new migration
```

Migrations live in `api/migrations/versions/`. Run via `cd api && uv run alembic upgrade head`.

## Notes

- `api` hot-reloads on changes to `./api` via bind mount + `--reload`.
- `ui` runs Vite dev server on port 8501.
- Postgres uses a named volume (`db_data`); Redis uses `redis_data`.
- Tail logs: `docker compose logs -f api`
