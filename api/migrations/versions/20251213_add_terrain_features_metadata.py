"""add terrain_features_metadata table

Revision ID: 20251213_terrain_features
Revises: 20251208_grid_fields
Create Date: 2025-12-13 00:00:00.000000
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
revision: str = "20251213_terrain_features"
down_revision: Union[str, Sequence[str], None] = "20251208_grid_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create terrain_features_metadata table."""
    op.create_table(
        "terrain_features_metadata",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("region_name", sa.Text(), nullable=False),
        sa.Column("source_dem_metadata_id", sa.BigInteger(), nullable=False),
        sa.Column("slope_path", sa.Text(), nullable=False),
        sa.Column("aspect_path", sa.Text(), nullable=False),
        sa.Column("crs_epsg", sa.Integer(), nullable=False),
        sa.Column("cell_size_deg", sa.Float(), nullable=False),
        sa.Column("origin_lat", sa.Float(), nullable=False),
        sa.Column("origin_lon", sa.Float(), nullable=False),
        sa.Column("grid_n_lat", sa.Integer(), nullable=False),
        sa.Column("grid_n_lon", sa.Integer(), nullable=False),
        sa.Column("bbox", Geometry("POLYGON", 4326), nullable=False),
        sa.Column("slope_units", sa.Text(), nullable=False, server_default="degrees"),
        sa.Column("aspect_units", sa.Text(), nullable=False, server_default="degrees"),
        sa.Column(
            "aspect_convention",
            sa.Text(),
            nullable=False,
            server_default="clockwise_from_north_downslope",
        ),
        sa.Column("nodata_value", sa.Float(), nullable=False, server_default="-9999"),
        sa.Column("slope_min", sa.Float(), nullable=True),
        sa.Column("slope_max", sa.Float(), nullable=True),
        sa.Column("aspect_min", sa.Float(), nullable=True),
        sa.Column("aspect_max", sa.Float(), nullable=True),
        sa.Column("coverage_fraction", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["source_dem_metadata_id"],
            ["terrain_metadata.id"],
            name="fk_terrain_features_metadata_source_dem_metadata_id",
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        "ix_terrain_features_region_created_at",
        "terrain_features_metadata",
        ["region_name", "created_at"],
    )
    op.create_index(
        "ix_terrain_features_source_dem_metadata_id",
        "terrain_features_metadata",
        ["source_dem_metadata_id"],
    )


def downgrade() -> None:
    """Drop terrain_features_metadata table."""
    op.drop_index(
        "ix_terrain_features_source_dem_metadata_id",
        table_name="terrain_features_metadata",
    )
    op.drop_index(
        "ix_terrain_features_region_created_at",
        table_name="terrain_features_metadata",
    )
    op.drop_table("terrain_features_metadata")

