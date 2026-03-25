"""Add is_archive boolean to fire_detections

Archive-ingested rows get a shorter retention window (ARCHIVE_RETENTION_DAYS,
default 3) than NRT data (RETENTION_DAYS, default 14).  The column defaults to
false so all existing rows are treated as NRT.

Revision ID: 20260325_add_is_archive
Revises: 20260325_terrain_fallback_flag
Create Date: 2026-03-25 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260325_add_is_archive"
down_revision: Union[str, Sequence[str], None] = "20260325_terrain_fallback_flag"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "fire_detections",
        sa.Column(
            "is_archive",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )
    # Partial index: cleanup only needs to scan archive rows separately.
    op.create_index(
        "ix_fire_detections_archive_acq_time",
        "fire_detections",
        ["acq_time"],
        postgresql_where=sa.text("is_archive = true"),
    )


def downgrade() -> None:
    op.drop_index("ix_fire_detections_archive_acq_time", table_name="fire_detections")
    op.drop_column("fire_detections", "is_archive")
