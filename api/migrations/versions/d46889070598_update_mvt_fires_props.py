"""update mvt_fires props

Revision ID: d46889070598
Revises: 29a201e56491
Create Date: 2026-01-13 23:50:26.003936

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'd46889070598'
down_revision: Union[str, Sequence[str], None] = '29a201e56491'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
    CREATE OR REPLACE FUNCTION mvt_fires(
        z integer, 
        x integer, 
        y integer, 
        start_time timestamptz DEFAULT now() - interval '24 hours', 
        end_time timestamptz DEFAULT now(), 
        min_confidence float DEFAULT 0,
        include_noise boolean DEFAULT false
    )
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
                lon,
                lat,
                is_noise,
                denoised_score,
                ST_AsMVTGeom(ST_Transform(geom, 3857), tile.bbox) AS geom
            FROM fire_detections, tile
            WHERE
                acq_time BETWEEN start_time AND end_time
                AND (confidence IS NULL OR confidence >= min_confidence)
                AND (include_noise IS TRUE OR is_noise IS NOT TRUE)
                AND ST_Intersects(ST_Transform(geom, 3857), tile.bbox)
        )
        SELECT ST_AsMVT(grid, 'fires', 4096, 'geom')
        INTO mvt
        FROM grid;

        RETURN mvt;
    END;
    $$ LANGUAGE plpgsql STABLE;
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Revert to version from ca7fc82ff882
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
                AND (is_noise IS NOT TRUE)
                AND ST_Intersects(ST_Transform(geom, 3857), tile.bbox)
        )
        SELECT ST_AsMVT(grid, 'fires', 4096, 'geom')
        INTO mvt
        FROM grid;

        RETURN mvt;
    END;
    $$ LANGUAGE plpgsql STABLE;
    """)
