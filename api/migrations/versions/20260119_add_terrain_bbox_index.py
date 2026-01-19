"""add spatial index on terrain_features_metadata.bbox

Revision ID: 20260119_terrain_bbox_index
Revises: 20260119_jit_jobs
Create Date: 2026-01-19 13:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260119_terrain_bbox_index"
down_revision: Union[str, Sequence[str], None] = "20260119_jit_jobs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add GIST spatial index on terrain_features_metadata.bbox."""
    op.execute(
        "CREATE INDEX ix_terrain_features_bbox "
        "ON terrain_features_metadata USING GIST (bbox)"
    )


def downgrade() -> None:
    """Drop GIST spatial index on terrain_features_metadata.bbox."""
    op.drop_index("ix_terrain_features_bbox", table_name="terrain_features_metadata")
