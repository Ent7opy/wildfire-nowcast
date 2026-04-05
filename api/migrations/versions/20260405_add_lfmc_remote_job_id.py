"""Add remote_job_id to fuel_moisture_runs for ECMWF job tracking

Track the remote ECMWF LFMC job ID so that on timeout, we can
cancel the orphaned job instead of leaving it running.

Revision ID: 20260405_add_lfmc_remote_job_id
Revises: 20260405_add_geometry_srid_constraints
Create Date: 2026-04-05 08:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260405_add_lfmc_remote_job_id"
down_revision: Union[str, Sequence[str], None] = "20260405_add_geometry_srid_constraints"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add remote_job_id column to fuel_moisture_runs."""
    op.add_column(
        "fuel_moisture_runs",
        sa.Column("remote_job_id", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    """Remove remote_job_id column from fuel_moisture_runs."""
    op.drop_column("fuel_moisture_runs", "remote_job_id")
