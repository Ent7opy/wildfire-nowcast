"""add landcover_class and landcover_label columns to fire_detections

These columns store ESA WorldCover LULC class code (integer) and human-readable
label (text) for each detection. They are populated by the LULC/fuels ingest
pipeline and used by denoiser v2 inference for agriculture-detection filtering.
Previously existed only in the pre-wipe DB without a formal migration.

Revision ID: 20260323_add_landcover_class_label
Revises: 20260315_cleanup_legacy_tables_and_extensions
Create Date: 2026-03-23 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260323_add_landcover_class_label"
down_revision: Union[str, Sequence[str], None] = "20260315_cleanup_legacy_tables_and_extensions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add landcover_class (ESA WorldCover class code) and landcover_label columns."""
    op.add_column(
        "fire_detections",
        sa.Column("landcover_class", sa.Integer(), nullable=True),
    )
    op.add_column(
        "fire_detections",
        sa.Column("landcover_label", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Drop landcover_class and landcover_label columns."""
    op.drop_column("fire_detections", "landcover_label")
    op.drop_column("fire_detections", "landcover_class")
