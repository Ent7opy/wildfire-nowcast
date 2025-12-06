"""add grid fields to terrain_metadata

Revision ID: 20251208_grid_fields
Revises: 20251207_add_terrain_metadata
Create Date: 2025-12-08 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20251208_grid_fields"
down_revision: Union[str, Sequence[str], None] = "20251207_add_terrain_metadata"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add grid definition columns to terrain_metadata."""
    op.add_column("terrain_metadata", sa.Column("cell_size_deg", sa.Float(), nullable=True))
    op.add_column("terrain_metadata", sa.Column("origin_lat", sa.Float(), nullable=True))
    op.add_column("terrain_metadata", sa.Column("origin_lon", sa.Float(), nullable=True))
    op.add_column("terrain_metadata", sa.Column("grid_n_lat", sa.Integer(), nullable=True))
    op.add_column("terrain_metadata", sa.Column("grid_n_lon", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Remove grid definition columns from terrain_metadata."""
    op.drop_column("terrain_metadata", "grid_n_lon")
    op.drop_column("terrain_metadata", "grid_n_lat")
    op.drop_column("terrain_metadata", "origin_lon")
    op.drop_column("terrain_metadata", "origin_lat")
    op.drop_column("terrain_metadata", "cell_size_deg")

