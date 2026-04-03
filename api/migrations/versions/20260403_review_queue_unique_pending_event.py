"""Add partial unique index on denoiser_review_queue(event_id) WHERE status = 'pending'.

Without this index a persistent fire event accumulates one review queue row per
inference run (up to 24+ per day), and operator resolutions are immediately
discarded on the next run because the plain INSERT adds a fresh 'open' row.

The partial index covers only rows with status = 'pending' (or the legacy value
'open') so that resolved/closed rows are not affected — the same event can
legitimately appear in the queue again after it has been fully resolved.

The application layer (ml/denoiser_inference_v2.py) uses ON CONFLICT DO NOTHING
so that a second INSERT for an already-pending event is silently dropped.

Revision ID: 20260403_review_queue_unique_pending_event
Revises: 20260402_review_queue_feedback_loop
Create Date: 2026-04-03
"""

from alembic import op

revision = "20260403_review_queue_unique_pending_event"
down_revision = "20260402_review_queue_feedback_loop"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Partial unique index: only one pending/open row per event_id at a time.
    # Resolved rows (status IN ('resolved', 'closed', …)) are excluded so the
    # same event can re-enter the queue after it has been fully resolved.
    op.execute(
        """
        CREATE UNIQUE INDEX uix_review_queue_event_pending
        ON denoiser_review_queue (event_id)
        WHERE status IN ('open', 'pending')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uix_review_queue_event_pending")
