"""Add coverage_fraction to fuel_moisture_runs

Fraction of grid cells (in the bounding box) with valid (non-NaN) LFMC
values, computed from the downloaded NetCDF at ingest time.  Stored so the
freshness endpoint can surface lfmc_coverage_fraction without reloading
raster files at query time.

Revision ID: 20260326_add_lfmc_coverage_fraction
Revises: 20260325_add_lulc_version
Create Date: 2026-03-26 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260326_add_lfmc_coverage_fraction"
down_revision: Union[str, Sequence[str], None] = "20260325_add_lulc_version"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add coverage_fraction (0.0–1.0) column to fuel_moisture_runs."""
    op.add_column(
        "fuel_moisture_runs",
        sa.Column("coverage_fraction", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("fuel_moisture_runs", "coverage_fraction")
