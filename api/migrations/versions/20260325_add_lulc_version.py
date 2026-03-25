"""Add lulc_version column to fire_detections

Records which ESA WorldCover release was used to classify each detection.
Populated by the LULC ingest pipeline alongside landcover_class/label/score.
Allows stale-row detection when a new WorldCover release is ingested.

Revision ID: 20260325_add_lulc_version
Revises: 20260325_add_query_indexes
Create Date: 2026-03-25 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260325_add_lulc_version"
down_revision: Union[str, Sequence[str], None] = "20260325_add_query_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add lulc_version (ESA WorldCover release tag, e.g. 'v200_2021') column."""
    op.add_column(
        "fire_detections",
        sa.Column("lulc_version", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Drop lulc_version column."""
    op.drop_column("fire_detections", "lulc_version")
