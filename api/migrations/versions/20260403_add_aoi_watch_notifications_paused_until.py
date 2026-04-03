"""add aoi watch_notifications_paused_until column

Revision ID: 20260403_add_aoi_watch_notifications_paused_until
Revises: d46889070598
Create Date: 2026-04-03 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260403_add_aoi_watch_notifications_paused_until"
down_revision: Union[str, Sequence[str], None] = "d46889070598"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add watch_notifications_paused_until column to aois table.

    NULL means notifications are active.  A future timestamp means notifications
    are paused until that time.  A past timestamp is treated as expired (active).
    """
    op.add_column(
        "aois",
        sa.Column(
            "watch_notifications_paused_until",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=None,
        ),
    )


def downgrade() -> None:
    """Remove watch_notifications_paused_until column from aois table."""
    op.drop_column("aois", "watch_notifications_paused_until")
