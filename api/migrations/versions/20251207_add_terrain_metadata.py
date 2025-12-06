"""add terrain_metadata table

Revision ID: 20251207_add_terrain_metadata
Revises: 20251206
Create Date: 2025-12-06 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


class Geometry(sa.types.UserDefinedType):
    """Minimal PostGIS geometry type helper for migrations."""

    def __init__(self, geometry_type: str, srid: int) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw: object) -> str:  # pragma: no cover - string format helper
        return f"geometry({self.geometry_type}, {self.srid})"


# revision identifiers, used by Alembic.
revision: str = "20251207_add_terrain_metadata"
down_revision: Union[str, Sequence[str], None] = "20251206"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create terrain_metadata table."""
    op.create_table(
        "terrain_metadata",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("region_name", sa.Text(), nullable=False),
        sa.Column("dem_source", sa.Text(), nullable=False),
        sa.Column("crs_epsg", sa.Integer(), nullable=False),
        sa.Column("resolution_m", sa.Float(), nullable=False),
        sa.Column("bbox", Geometry("POLYGON", 4326), nullable=False),
        sa.Column("raster_path", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_terrain_metadata_region_name_created_at",
        "terrain_metadata",
        ["region_name", "created_at"],
    )


def downgrade() -> None:
    """Drop terrain_metadata table."""
    op.drop_index("ix_terrain_metadata_region_name_created_at", table_name="terrain_metadata")
    op.drop_table("terrain_metadata")

