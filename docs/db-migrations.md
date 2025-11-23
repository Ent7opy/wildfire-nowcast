# Database Migrations Guide

This guide explains how to work with database schema changes in the wildfire-nowcast project.

## Overview

We use [Alembic](https://alembic.sqlalchemy.org/) for database migration management with PostgreSQL + PostGIS. Migrations are versioned and allow for repeatable, reversible schema changes.

## Directory Structure

```
api/
├── migrations/
│   ├── env.py              # Migration environment configuration
│   ├── script.py.mako      # Migration template
│   ├── versions/           # Migration files
│   └── README              # Alembic-generated docs
├── alembic.ini            # Alembic configuration
├── db.py                  # SQLAlchemy engine setup
└── config.py              # Database connection settings
```

## Migration Workflow

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

### 4. Test Locally

```bash
# Start database
make db-up

# Run migrations
make migrate

# Verify changes
docker compose exec db psql -U wildfire -d wildfire -c "\dt"
```

### 5. Commit Migration

Commit both the migration file and any related code changes.

## Best Practices

### Migration Naming
- Use descriptive, action-oriented messages: `"add user authentication table"` not `"changes"`
- Keep messages under 50 characters for readability

### Idempotent Operations
- Use `IF NOT EXISTS` for CREATE statements
- Check for existence before dropping objects
- Use conditional logic for complex changes

### Spatial Data
- Always specify SRID for geometry columns (4326 for WGS84 lat/lng)
- Use appropriate PostGIS types: `POINT`, `LINESTRING`, `POLYGON`, `MULTI*`
- Enable PostGIS extension in the initial migration (already done)

### Rollbacks
- Always implement `downgrade()` functions
- Test rollbacks: `alembic downgrade -1`
- Avoid destructive operations that can't be safely reversed

## Common Patterns

### Adding a Spatial Column
```python
op.add_column(
    'fires',
    sa.Column('location', geoalchemy2.Geometry('POINT', srid=4326))
)
```

### Creating an Index
```python
op.create_index(
    'idx_fires_location',
    'fires',
    ['location'],
    postgresql_using='gist'  # For spatial indexes
)
```

### Adding PostGIS Functions
```python
# Enable additional extensions if needed
op.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology;")
```

## Troubleshooting

### Migration Fails
- Check database connectivity: `docker compose logs db`
- Verify migration syntax: `python -m py_compile api/migrations/versions/xxxx.py`
- Test manually: `docker compose exec db psql -U wildfire -d wildfire`

### Conflicts with Existing Data
- Use conditional checks in migrations
- Consider data migration scripts for complex transformations
- Test on a copy of production data

### PostGIS Issues
- Verify extension is enabled: `SELECT postgis_full_version();`
- Check SRID consistency across related tables
- Use `ST_IsValid()` for geometry validation

## Environment Variables

Database connection is configured via environment variables (see `infra/README.md` for defaults):

- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`

## Future Considerations

- **ORM Integration**: When SQLAlchemy models are added, enable `target_metadata` in `env.py` for auto-generation
- **Testing**: Add migration tests to ensure upgrade/downgrade cycles work
- **CI/CD**: Automate migration runs in deployment pipelines
