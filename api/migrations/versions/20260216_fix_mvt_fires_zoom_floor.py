"""remove unintended low-zoom floor from mvt_fires

Revision ID: 20260216_fix_mvt_fires_zoom_floor
Revises: 20260215_runtime_watermarks_and_model_registry
Create Date: 2026-02-16 16:30:00.000000
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260216_fix_mvt_fires_zoom_floor"
down_revision: Union[str, Sequence[str], None] = "20260215_runtime_watermarks_and_model_registry"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_MVT_FIRES_SQL = """
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
            AND (
                min_fire_likelihood IS NULL
                OR fire_likelihood IS NULL
                OR fire_likelihood >= min_fire_likelihood
            )
            AND ST_Intersects(ST_Transform(geom, 3857), tile.bbox)
    )
    SELECT ST_AsMVT(grid, 'fires', 4096, 'geom')
    INTO mvt
    FROM grid;

    RETURN mvt;
END;
$$ LANGUAGE plpgsql STABLE;
"""


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(_MVT_FIRES_SQL)


def downgrade() -> None:
    """Downgrade schema."""
    # Keep canonical no-floor behavior on downgrade to avoid reintroducing
    # drifted behavior that hides fires at low zoom levels.
    op.execute(_MVT_FIRES_SQL)

