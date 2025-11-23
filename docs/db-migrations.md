## Database migrations (short guide)

We use [Alembic](https://alembic.sqlalchemy.org/) for database migration management with PostgreSQL + PostGIS. Migrations are versioned and reversible.

---

## Directory structure

```text
api/
├── alembic.ini
├── db.py                  # SQLAlchemy engine setup
├── config.py              # Database connection settings
└── migrations/
    ├── env.py             # Migration environment configuration
    ├── script.py.mako     # Migration template
    └── versions/          # Migration files
```

---

## Migration workflow

### 1. Make Schema Changes

When you need to modify the database schema:

- **For new tables/models**: Define them using SQLAlchemy models (future) or raw SQL in migrations
- **For existing tables**: Write ALTER statements in migration files
- **For spatial features**: Use PostGIS functions and types

### 2. Generate Migration

Create a new migration file with a descriptive message:

```bash
make revision msg="add wildfire perimeter table"
# or: cd api && uv run alembic revision -m "add wildfire perimeter table"
```

This creates a new file in `api/migrations/versions/` with `upgrade()` and `downgrade()` functions.

### 3. Edit the Migration

Fill in the `upgrade()` function with your schema changes:

```python
def upgrade() -> None:
    op.create_table(
        'wildfire_perimeters',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('fire_name', sa.String(255)),
        sa.Column('geometry', geoalchemy2.Geometry('POLYGON', srid=4326)),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('now()')),
    )

def downgrade() -> None:
    op.drop_table('wildfire_perimeters')
```

### 4. Test locally

```bash
# Start database
make db-up

# Run migrations
make migrate

# Verify changes
docker compose exec db psql -U wildfire -d wildfire -c "\dt"
```

### 5. Commit migration

Commit both the migration file and any related code changes.

---

## Tips & troubleshooting

- **Migration naming**
  - Use descriptive, action‑oriented messages: `"add wildfire perimeter table"` not `"changes"`.

- **Spatial data**
  - Always specify SRID for geometry columns (4326 for WGS84 lat/lng).
  - Use appropriate PostGIS types: `POINT`, `LINESTRING`, `POLYGON`, `MULTI*`.
  - Enable PostGIS extension in the initial migration (already done).

- **Rollbacks**
  - Always implement `downgrade()` functions.
  - Test rollbacks: `alembic downgrade -1`.

- **Troubleshooting**
  - Migration fails: check DB logs (`docker compose logs db`) and run `docker compose exec db psql -U wildfire -d wildfire`.
  - Data conflicts: use conditional checks or one‑off data scripts; test on a copy of real data if possible.

- **Environment variables**
  - DB connection is driven by `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`. Defaults are documented in `infra/README.md`.

As SQLAlchemy models mature, we can enable autogeneration via `target_metadata` in `env.py` and extend this doc with more patterns.
