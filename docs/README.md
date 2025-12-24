# Docs Hub & Map

Purpose: give newcomers and contributors a single navigation point, reduce duplication between pages, and clarify which doc to read for which task. Keep links relative so they work in editors and GitHub.

## Reading order (new contributor)
1) `WILDFIRE_NOWCAST_101.md` – product and scope overview.  
2) `architecture.md` – how the system fits together today vs planned.  
3) `SETUP.md` → `dev-python-env.md` – environment and tooling details.  
4) Ingestion docs below as needed.

## Categories & links
- Orientation: `WILDFIRE_NOWCAST_101.md`
- System architecture: `architecture.md`
- Setup & tooling: `SETUP.md`, `dev-python-env.md`
- Ingestion (pipelines): `ingest/ingest_firms.md`, `ingest/ingest_weather.md`, `ingest/ingest_dem.md`, `ingest/data_quality.md`
- ML: `denoiser.md`, `ml/denoiser.md`
- Terrain + grid contract: `terrain_grid.md`
- Data & schema: `data/data_formats.md`, `data/data_schema_fires.md`, `data/db-migrations.md`
- Scripts/helpers: `scripts/` are lightly documented inline; ingestion make targets in `Makefile`

## Ingestion doc map
- Fire detections: `ingest/ingest_firms.md` (config, run, validation)  
- Weather: `ingest/ingest_weather.md` (GFS 0.25°)  
- Terrain: `ingest/ingest_dem.md` (Copernicus GLO-30)  
- Validation: `ingest/data_quality.md` (what gets logged/checked)  
- Canonical shapes: `data/data_formats.md` (fire/weather/DEM outputs)

## Data & persistence references
- Database migrations: `data/db-migrations.md` (Alembic workflow)
- Fire tables: `data/data_schema_fires.md` (schema + indexes)
- Dataset shapes: `data/data_formats.md`

## Contributing to docs (light guidance)
- Start with a one-line “audience” and “when to use this doc” if adding new pages.
- Prefer linking to an existing doc instead of restating the same bullets.
- Keep commands copy/pastable; note platform-specific flags when relevant.
- Update this hub when adding a new doc so navigation stays accurate.

## Folder layout
- `ingest/` – ingestion pipelines, validation
- `data/` – dataset shapes, schemas, migrations
- Add short READMEs in new folders if you add more docs to keep navigation clear.


