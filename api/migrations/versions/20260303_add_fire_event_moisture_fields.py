"""add fire_events moisture feature columns

Revision ID: 20260303_add_fire_event_moisture_fields
Revises: 20260301_global_authoritative_industrial_v1
Create Date: 2026-03-03 18:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260303_add_fire_event_moisture_fields"
down_revision: Union[str, Sequence[str], None] = "20260301_global_authoritative_industrial_v1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("fire_events", sa.Column("lfmc_mean", sa.Float(), nullable=True))
    op.add_column("fire_events", sa.Column("dfmc_10hr_mean", sa.Float(), nullable=True))
    op.add_column(
        "fire_events",
        sa.Column("lfmc_is_available", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "fire_events",
        sa.Column("dfmc_is_available", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("fire_events", "dfmc_is_available")
    op.drop_column("fire_events", "lfmc_is_available")
    op.drop_column("fire_events", "dfmc_10hr_mean")
    op.drop_column("fire_events", "lfmc_mean")

