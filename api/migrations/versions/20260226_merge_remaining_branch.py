"""Merge remaining orphaned migration branch into the main chain.

Consolidates the final divergent migration head:
  - 20260225_merge_migration_branches (main chain from PR #315)
  - 20260119_add_persistence_score (orphaned branch from b8e90cee90c8)

This is a no-op merge migration; all schema changes are in the parent
revisions. Without this merge, Alembic raises:
  "FAILED: Multiple head revisions are present for given argument 'head'"

Revision ID: 20260226_merge_remaining_branch
Revises: 20260225_merge_migration_branches, 20260119_add_persistence_score
Create Date: 2026-02-26 00:00:00.000000
"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "20260226_merge_remaining_branch"
down_revision: Union[str, Sequence[str], None] = (
    "20260225_merge_migration_branches",
    "20260119_add_persistence_score",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op: this migration only merges two branch heads."""
    pass


def downgrade() -> None:
    """No-op: this migration only merges two branch heads."""
    pass
