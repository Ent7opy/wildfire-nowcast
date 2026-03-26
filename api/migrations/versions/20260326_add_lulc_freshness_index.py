"""Add partial index on fire_detections for LULC freshness queries

The data-freshness endpoint scans fire_detections for:
  - MAX(created_at) WHERE lulc_version IS NOT NULL (global)
  - COUNT(*) WHERE created_at >= NOW() - INTERVAL '7 days' (with/without lulc_version)

Without an index, both require a full sequential scan on a potentially large
table. The partial composite index below covers both aggregates in one pass:
  - Filters to classified rows only (partial: lulc_version IS NOT NULL)
  - Supports created_at range queries via DESC ordering
  - Includes lulc_version to avoid a heap fetch for the version value

Revision ID: 20260326_add_lulc_freshness_index
Revises: 20260326_add_lfmc_coverage_fraction
Create Date: 2026-03-26 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260326_add_lulc_freshness_index"
down_revision: Union[str, Sequence[str], None] = "20260326_add_lfmc_coverage_fraction"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_fire_detections_lulc_created_at",
        "fire_detections",
        ["created_at", "lulc_version"],
        postgresql_where="lulc_version IS NOT NULL",
    )


def downgrade() -> None:
    op.drop_index("ix_fire_detections_lulc_created_at", table_name="fire_detections")
