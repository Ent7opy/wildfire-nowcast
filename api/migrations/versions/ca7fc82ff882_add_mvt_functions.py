"""add mvt functions

Revision ID: ca7fc82ff882
Revises: 56502d0fdf18
Create Date: 2026-01-11 18:58:58.306640

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ca7fc82ff882'
down_revision: Union[str, Sequence[str], None] = '56502d0fdf18'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # mvt_fires
    op.execute("""
    CREATE OR REPLACE FUNCTION mvt_fires(z integer, x integer, y integer, start_time timestamptz DEFAULT now() - interval '24 hours', end_time timestamptz DEFAULT now(), min_confidence float DEFAULT 0)
    RETURNS bytea AS $$
    DECLARE
        mvt bytea;
    BEGIN
        WITH tile AS (
            SELECT ST_TileEnvelope(z, x, y) AS bbox
        ),
        grid AS (
            SELECT
                id,
                acq_time,
                confidence,
                frp,
                sensor,
                source,
                ST_AsMVTGeom(ST_Transform(geom, 3857), tile.bbox) AS geom
            FROM fire_detections, tile
            WHERE
                acq_time BETWEEN start_time AND end_time
                AND (confidence IS NULL OR confidence >= min_confidence)
                AND ST_Intersects(ST_Transform(geom, 3857), tile.bbox)
        )
        SELECT ST_AsMVT(grid, 'fires', 4096, 'geom')
        INTO mvt
        FROM grid;

        RETURN mvt;
    END;
    $$ LANGUAGE plpgsql STABLE;
    """)

    # mvt_forecast_contours
    op.execute("""
    CREATE OR REPLACE FUNCTION mvt_forecast_contours(z integer, x integer, y integer, run_id bigint DEFAULT NULL)
    RETURNS bytea AS $$
    DECLARE
        mvt bytea;
    BEGIN
        WITH tile AS (
            SELECT ST_TileEnvelope(z, x, y) AS bbox
        ),
        latest_run AS (
            SELECT id FROM spread_forecast_runs WHERE status = 'completed' ORDER BY forecast_reference_time DESC LIMIT 1
        ),
        target_run AS (
            SELECT COALESCE(run_id, (SELECT id FROM latest_run)) AS id
        ),
        grid AS (
            SELECT
                c.horizon_hours,
                c.threshold,
                ST_AsMVTGeom(ST_Transform(c.geom, 3857), tile.bbox) AS geom
            FROM spread_forecast_contours c, tile, target_run
            WHERE
                c.run_id = target_run.id
                AND ST_Intersects(ST_Transform(c.geom, 3857), tile.bbox)
        )
        SELECT ST_AsMVT(grid, 'forecast_contours', 4096, 'geom')
        INTO mvt
        FROM grid;

        RETURN mvt;
    END;
    $$ LANGUAGE plpgsql STABLE;
    """)

    # mvt_aois
    op.execute("""
    CREATE OR REPLACE FUNCTION mvt_aois(z integer, x integer, y integer)
    RETURNS bytea AS $$
    DECLARE
        mvt bytea;
    BEGIN
        WITH tile AS (
            SELECT ST_TileEnvelope(z, x, y) AS bbox
        ),
        grid AS (
            SELECT
                id,
                name,
                area_km2,
                ST_AsMVTGeom(ST_Transform(geom, 3857), tile.bbox) AS geom
            FROM aois, tile
            WHERE
                ST_Intersects(ST_Transform(geom, 3857), tile.bbox)
        )
        SELECT ST_AsMVT(grid, 'aois', 4096, 'geom')
        INTO mvt
        FROM grid;

        RETURN mvt;
    END;
    $$ LANGUAGE plpgsql STABLE;
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP FUNCTION IF EXISTS mvt_fires;")
    op.execute("DROP FUNCTION IF EXISTS mvt_forecast_contours;")
    op.execute("DROP FUNCTION IF EXISTS mvt_aois;")
