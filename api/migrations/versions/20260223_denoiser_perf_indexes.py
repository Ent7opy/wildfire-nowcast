"""add denoiser query performance indexes

Revision ID: 20260223_denoiser_perf_indexes
Revises: 20260222_add_fuel_moisture_runs
Create Date: 2026-02-23 11:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260223_denoiser_perf_indexes"
down_revision: Union[str, Sequence[str], None] = "20260222_add_fuel_moisture_runs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_fire_detections_ingest_batch_id
        ON fire_detections (ingest_batch_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_fire_detections_ingest_batch_event_id
        ON fire_detections (ingest_batch_id, event_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_fire_detections_geog
        ON fire_detections USING gist ((geom::geography))
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_fire_perimeters_geog
        ON fire_perimeters USING gist ((geom::geography))
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_industrial_sources_geog
        ON industrial_sources USING gist ((geom::geography))
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_denoiser_labels_v2_rule_detection
        ON denoiser_labels_v2 (rule_version, fire_detection_id)
        """
    )

    op.execute("ANALYZE fire_detections")
    op.execute("ANALYZE fire_perimeters")
    op.execute("ANALYZE industrial_sources")
    op.execute("ANALYZE denoiser_labels_v2")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_denoiser_labels_v2_rule_detection")
    op.execute("DROP INDEX IF EXISTS ix_industrial_sources_geog")
    op.execute("DROP INDEX IF EXISTS ix_fire_perimeters_geog")
    op.execute("DROP INDEX IF EXISTS ix_fire_detections_geog")
    op.execute("DROP INDEX IF EXISTS ix_fire_detections_ingest_batch_event_id")
    op.execute("DROP INDEX IF EXISTS ix_fire_detections_ingest_batch_id")
