"""Merge three unresolved migration heads into a single lineage.

Heads being merged:
  - 20260402_add_ignition_model_family (ignition model family CHECK)
  - 20260403_review_queue_unique_pending_event (review queue partial unique index)
  - 20260403_add_aoi_watch_notifications_paused_until (AOI watch pause column)

Revision ID: 20260404_merge_three_heads
Revises: 20260402_add_ignition_model_family, 20260403_review_queue_unique_pending_event, 20260403_add_aoi_watch_notifications_paused_until
Create Date: 2026-04-04 00:00:00.000000

"""

from typing import Sequence, Union

revision: str = "20260404_merge_three_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20260402_add_ignition_model_family",
    "20260403_review_queue_unique_pending_event",
    "20260403_add_aoi_watch_notifications_paused_until",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Merge migration — no schema changes."""
    pass


def downgrade() -> None:
    """Merge migration — no schema changes."""
    pass
