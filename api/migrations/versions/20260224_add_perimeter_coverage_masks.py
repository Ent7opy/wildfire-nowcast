"""add perimeter coverage masks for covered-first denoiser gating

Revision ID: 20260224_add_perimeter_coverage_masks
Revises: 20260223_denoiser_perf_indexes
Create Date: 2026-02-24 15:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


class Geometry(sa.types.UserDefinedType):
    """Minimal PostGIS geometry type helper for migrations."""

    def __init__(self, geometry_type: str, srid: int) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw: object) -> str:
        return f"geometry({self.geometry_type}, {self.srid})"


# revision identifiers, used by Alembic.
revision: str = "20260224_add_perimeter_coverage_masks"
down_revision: Union[str, Sequence[str], None] = "20260223_denoiser_perf_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "perimeter_coverage_masks",
        sa.Column("mask_id", sa.String(length=128), primary_key=True),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("reliability_tier", sa.String(length=32), nullable=False, server_default="gold"),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("geom", Geometry("MULTIPOLYGON", 4326), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_perimeter_coverage_masks_geom",
        "perimeter_coverage_masks",
        ["geom"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_perimeter_coverage_masks_active_window",
        "perimeter_coverage_masks",
        ["is_active", "valid_from", "valid_to"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_perimeter_coverage_masks_active_window", table_name="perimeter_coverage_masks")
    op.drop_index("ix_perimeter_coverage_masks_geom", table_name="perimeter_coverage_masks")
    op.drop_table("perimeter_coverage_masks")
