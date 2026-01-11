"""add aois table

Revision ID: 56502d0fdf18
Revises: 20251228_spread_forecasts
Create Date: 2026-01-11 18:53:31.680855

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '56502d0fdf18'
down_revision: Union[str, Sequence[str], None] = '20251228_spread_forecasts'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


class Geometry(sa.types.UserDefinedType):
    """Minimal PostGIS geometry type helper for migrations."""

    def __init__(self, geometry_type: str, srid: int) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw: object) -> str:
        return f"geometry({self.geometry_type}, {self.srid})"


def upgrade() -> None:
    """Create aois table."""
    op.create_table(
        "aois",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("owner_id", sa.Text(), nullable=True),
        sa.Column("geom", Geometry("MULTIPOLYGON", 4326), nullable=False),
        sa.Column("bbox", Geometry("POLYGON", 4326), nullable=False),
        sa.Column("area_km2", sa.Float(), nullable=False),
        sa.Column("vertex_count", sa.Integer(), nullable=False),
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
        "ix_aois_geom",
        "aois",
        ["geom"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_aois_bbox",
        "aois",
        ["bbox"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_aois_created_at",
        "aois",
        ["created_at"],
    )


def downgrade() -> None:
    """Drop aois table."""
    op.drop_index("ix_aois_created_at", table_name="aois")
    op.drop_index("ix_aois_bbox", table_name="aois")
    op.drop_index("ix_aois_geom", table_name="aois")
    op.drop_table("aois")
