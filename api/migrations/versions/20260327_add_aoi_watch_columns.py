"""add aoi watch columns

Revision ID: 20260327_add_aoi_watch_columns
Revises: 20260327_add_cursor_pagination_indexes
Create Date: 2026-03-27 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260327_add_aoi_watch_columns"
down_revision: Union[str, Sequence[str], None] = "20260327_add_cursor_pagination_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add watchlist columns to aois table."""
    op.add_column("aois", sa.Column("watch_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("aois", sa.Column("watch_interval_minutes", sa.Integer(), nullable=True))
    op.add_column("aois", sa.Column("watch_alert_threshold", sa.Float(), nullable=True))
    op.add_column("aois", sa.Column("watch_last_checked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("aois", sa.Column("watch_last_alerted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("aois", sa.Column("watch_last_spread_prob", sa.Float(), nullable=True))

    op.create_index("ix_aois_watch_enabled", "aois", ["watch_enabled"])


def downgrade() -> None:
    """Remove watchlist columns from aois table."""
    op.drop_index("ix_aois_watch_enabled", table_name="aois")
    op.drop_column("aois", "watch_last_spread_prob")
    op.drop_column("aois", "watch_last_alerted_at")
    op.drop_column("aois", "watch_last_checked_at")
    op.drop_column("aois", "watch_alert_threshold")
    op.drop_column("aois", "watch_interval_minutes")
    op.drop_column("aois", "watch_enabled")
