"""Merge migration branches into a single linear history.

Consolidates two divergent migration heads:
  - 20260119_jit_jobs (from 20260119_add_jit_forecast_jobs.py)
  - 20260224_add_perimeter_coverage_masks

This is a no-op merge migration; all schema changes are in the parent
revisions. Without this merge, Alembic raises:
  "FAILED: Multiple head revisions are present for given argument 'head'"

Revision ID: 20260225_merge_migration_branches
Revises: 20260119_jit_jobs, 20260224_add_perimeter_coverage_masks
Create Date: 2026-02-25 00:00:00.000000
"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "20260225_merge_migration_branches"
down_revision: Union[str, Sequence[str], None] = (
    "20260119_jit_jobs",
    "20260224_add_perimeter_coverage_masks",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op: this migration only merges two branch heads."""
    pass


def downgrade() -> None:
    """No-op: this migration only merges two branch heads."""
    pass
