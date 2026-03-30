"""add ne_populated_places table for review queue location context

Stores Natural Earth 10m populated places for nearest-place spatial queries
in the review queue and fire detail endpoints. Populated by the seed script
at scripts/seed_ne_populated_places.py — run once after migrating.

Revision ID: 20260330_add_ne_populated_places
Revises: 20260327_add_aoi_watch_columns
Create Date: 2026-03-30 12:00:00.000000
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


revision: str = "20260330_add_ne_populated_places"
down_revision: Union[str, Sequence[str], None] = "20260327_add_aoi_watch_columns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create ne_populated_places table."""
    op.create_table(
        "ne_populated_places",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("geom", Geometry("POINT", 4326), nullable=False),
        sa.Column("pop_max", sa.BigInteger(), nullable=True),
        sa.Column("adm0_a3", sa.String(length=3), nullable=True),
        sa.Column("adm1name", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_ne_populated_places_geom",
        "ne_populated_places",
        ["geom"],
        postgresql_using="gist",
    )


def downgrade() -> None:
    """Drop ne_populated_places table."""
    op.drop_index("ix_ne_populated_places_geom", table_name="ne_populated_places")
    op.drop_table("ne_populated_places")
