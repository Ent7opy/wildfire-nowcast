"""add geometry provenance fields for fire events/fronts

Revision ID: 20260312_add_fire_geom_provenance
Revises: 20260312_add_reverse_geocode_cache
Create Date: 2026-03-12 20:45:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260312_add_fire_geom_provenance"
down_revision: Union[str, Sequence[str], None] = "20260312_add_reverse_geocode_cache"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "fire_fronts",
        sa.Column("geom_source", sa.String(length=24), nullable=False, server_default="estimated"),
    )
    op.add_column(
        "fire_fronts",
        sa.Column("geom_method", sa.String(length=48), nullable=False, server_default="estimated_convex"),
    )
    op.add_column(
        "fire_fronts",
        sa.Column("geom_quality", sa.Float(), nullable=False, server_default=sa.text("0.4")),
    )
    op.add_column("fire_fronts", sa.Column("authoritative_perimeter_id", sa.BigInteger(), nullable=True))
    op.add_column("fire_fronts", sa.Column("authority_profile", sa.String(length=64), nullable=True))

    op.create_foreign_key(
        "fk_fire_fronts_authoritative_perimeter_id",
        "fire_fronts",
        "authoritative_perimeters",
        ["authoritative_perimeter_id"],
        ["perimeter_id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_fire_fronts_geom_source_authority_profile",
        "fire_fronts",
        ["geom_source", "authority_profile"],
    )
    op.create_index(
        "ix_fire_fronts_authoritative_perimeter_id",
        "fire_fronts",
        ["authoritative_perimeter_id"],
    )

    op.add_column(
        "fire_events",
        sa.Column("geom_source", sa.String(length=24), nullable=False, server_default="estimated"),
    )
    op.add_column(
        "fire_events",
        sa.Column("geom_method", sa.String(length=48), nullable=False, server_default="estimated_convex"),
    )
    op.add_column(
        "fire_events",
        sa.Column("geom_quality", sa.Float(), nullable=False, server_default=sa.text("0.4")),
    )
    op.add_column("fire_events", sa.Column("authoritative_perimeter_id", sa.BigInteger(), nullable=True))
    op.add_column("fire_events", sa.Column("authority_profile", sa.String(length=64), nullable=True))

    op.create_foreign_key(
        "fk_fire_events_authoritative_perimeter_id",
        "fire_events",
        "authoritative_perimeters",
        ["authoritative_perimeter_id"],
        ["perimeter_id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_fire_events_geom_source_authority_profile",
        "fire_events",
        ["geom_source", "authority_profile"],
    )
    op.create_index(
        "ix_fire_events_authoritative_perimeter_id",
        "fire_events",
        ["authoritative_perimeter_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_fire_events_authoritative_perimeter_id", table_name="fire_events")
    op.drop_index("ix_fire_events_geom_source_authority_profile", table_name="fire_events")
    op.drop_constraint("fk_fire_events_authoritative_perimeter_id", "fire_events", type_="foreignkey")
    op.drop_column("fire_events", "authority_profile")
    op.drop_column("fire_events", "authoritative_perimeter_id")
    op.drop_column("fire_events", "geom_quality")
    op.drop_column("fire_events", "geom_method")
    op.drop_column("fire_events", "geom_source")

    op.drop_index("ix_fire_fronts_authoritative_perimeter_id", table_name="fire_fronts")
    op.drop_index("ix_fire_fronts_geom_source_authority_profile", table_name="fire_fronts")
    op.drop_constraint("fk_fire_fronts_authoritative_perimeter_id", "fire_fronts", type_="foreignkey")
    op.drop_column("fire_fronts", "authority_profile")
    op.drop_column("fire_fronts", "authoritative_perimeter_id")
    op.drop_column("fire_fronts", "geom_quality")
    op.drop_column("fire_fronts", "geom_method")
    op.drop_column("fire_fronts", "geom_source")
