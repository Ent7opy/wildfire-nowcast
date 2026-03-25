"""Add terrain_fallback_used flag to terrain_features_metadata

Tracks whether a terrain_features_metadata row was produced from a
flat-terrain stub (when no real DEM file was available) or from a
lower-resolution fallback DEM.  Downstream consumers must check this
flag and treat the data as approximate.

Revision ID: 20260325_terrain_fallback_flag
Revises: 20260323_add_landcover_class_label
Create Date: 2026-03-25 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260325_terrain_fallback_flag"
down_revision: Union[str, Sequence[str], None] = "20260323_add_landcover_class_label"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "terrain_features_metadata",
        sa.Column(
            "terrain_fallback_used",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )


def downgrade() -> None:
    op.drop_column("terrain_features_metadata", "terrain_fallback_used")
