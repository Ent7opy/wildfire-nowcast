"""add fire_perimeters table for ground truth labeling

Revision ID: 20260130_add_fire_perimeters
Revises: 20260120_drop_mvt_fires_overloads
Create Date: 2026-01-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


class Geometry(sa.types.UserDefinedType):
    """Minimal PostGIS geometry type helper for migrations."""

    def __init__(self, geometry_type: str, srid: int) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw: object) -> str:
        return f"geometry({self.geometry_type}, {self.srid})"


# revision identifiers, used by Alembic.
revision: str = "20260130_add_fire_perimeters"
down_revision: Union[str, Sequence[str], None] = "20260120_drop_mvt_fires_overloads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "fire_perimeters",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("geom", Geometry("MULTIPOLYGON", 4326), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=True),
        sa.Column("fire_name", sa.Text(), nullable=True),
        sa.Column("fire_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fire_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acres", sa.Float(), nullable=True),
        sa.Column("cause", sa.Text(), nullable=True),
        sa.Column("state", sa.Text(), nullable=True),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_fire_perimeters_geom",
        "fire_perimeters",
        ["geom"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_fire_perimeters_fire_start",
        "fire_perimeters",
        ["fire_start"],
    )
    op.create_index(
        "ix_fire_perimeters_fire_end",
        "fire_perimeters",
        ["fire_end"],
    )
    op.create_index(
        "ix_fire_perimeters_source_source_id",
        "fire_perimeters",
        ["source", "source_id"],
        unique=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_fire_perimeters_source_source_id", table_name="fire_perimeters")
    op.drop_index("ix_fire_perimeters_fire_end", table_name="fire_perimeters")
    op.drop_index("ix_fire_perimeters_fire_start", table_name="fire_perimeters")
    op.drop_index("ix_fire_perimeters_geom", table_name="fire_perimeters")
    op.drop_table("fire_perimeters")
