"""update mvt_fires to include likelihood scores

Revision ID: 20260119_update_mvt_fires_likelihood
Revises: 20260119_add_fire_likelihood
Create Date: 2026-01-19 20:10:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '20260119_update_mvt_fires_likelihood'
down_revision: Union[str, Sequence[str], None] = '20260119_add_fire_likelihood'
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
                confidence_score,
                persistence_score,
                landcover_score,
                weather_score,
                false_source_masked,
                fire_likelihood,
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
    # Revert to previous version (d46889070598)
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
