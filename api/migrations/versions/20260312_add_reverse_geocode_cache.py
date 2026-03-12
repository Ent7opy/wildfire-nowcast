"""add reverse geocode cache table

Revision ID: 20260312_add_reverse_geocode_cache
Revises: 20260303_add_fire_event_moisture_fields
Create Date: 2026-03-12 18:20:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20260312_add_reverse_geocode_cache"
down_revision: Union[str, Sequence[str], None] = "20260303_add_fire_event_moisture_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "reverse_geocode_cache",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("cached_lat", sa.Float(), nullable=False),
        sa.Column("cached_lon", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="unresolved"),
        sa.Column("location_name", sa.Text(), nullable=True),
        sa.Column("country_name", sa.Text(), nullable=True),
        sa.Column("admin1_name", sa.Text(), nullable=True),
        sa.Column("admin2_name", sa.Text(), nullable=True),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("provider", "cached_lat", "cached_lon", name="uq_reverse_geocode_provider_coord"),
    )
    op.create_index(
        "ix_reverse_geocode_cache_expires_at",
        "reverse_geocode_cache",
        ["expires_at"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_reverse_geocode_cache_expires_at", table_name="reverse_geocode_cache")
    op.drop_table("reverse_geocode_cache")
