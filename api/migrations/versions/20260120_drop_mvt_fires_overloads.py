"""drop legacy mvt_fires overloads

Revision ID: 20260120_drop_mvt_fires_overloads
Revises: 20260119_add_mvt_fires_likelihood_filter
Create Date: 2026-01-20 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260120_drop_mvt_fires_overloads"
down_revision: Union[str, Sequence[str], None] = "20260119_add_mvt_fires_likelihood_filter"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Multiple overloaded `mvt_fires(...)` functions cause pg_tileserv to fail with:
    #   ERROR: function public.mvt_fires(...) is not unique (SQLSTATE 42725)
    # when parameters are passed via querystring (types are "unknown").
    #
    # Keep only the latest signature (with min_fire_likelihood) and rely on defaults
    # for callers that omit trailing params.
    op.execute(
        """
        DROP FUNCTION IF EXISTS public.mvt_fires(
            integer, integer, integer,
            timestamp with time zone, timestamp with time zone,
            double precision
        );
        """
    )
    op.execute(
        """
        DROP FUNCTION IF EXISTS public.mvt_fires(
            integer, integer, integer,
            timestamp with time zone, timestamp with time zone,
            double precision, boolean
        );
        """
    )

    # Re-assert the canonical function definition (idempotent if it already exists).
    op.execute(
        """
        CREATE OR REPLACE FUNCTION mvt_fires(
            z integer,
            x integer,
            y integer,
            start_time timestamptz DEFAULT now() - interval '24 hours',
            end_time timestamptz DEFAULT now(),
            min_confidence float DEFAULT 0,
            include_noise boolean DEFAULT false,
            min_fire_likelihood float DEFAULT NULL
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
                    AND (min_fire_likelihood IS NULL OR fire_likelihood IS NULL OR fire_likelihood >= min_fire_likelihood)
                    AND ST_Intersects(ST_Transform(geom, 3857), tile.bbox)
            )
            SELECT ST_AsMVT(grid, 'fires', 4096, 'geom')
            INTO mvt
            FROM grid;

            RETURN mvt;
        END;
        $$ LANGUAGE plpgsql STABLE;
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Restore a prior signature without min_fire_likelihood (not recommended for current UI).
    op.execute(
        """
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
        """
    )
