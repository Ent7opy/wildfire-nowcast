"""add fuel moisture runs metadata table

Revision ID: 20260222_add_fuel_moisture_runs
Revises: 20260221_event_denoiser_v2
Create Date: 2026-02-22 22:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260222_add_fuel_moisture_runs"
down_revision: Union[str, Sequence[str], None] = "20260221_event_denoiser_v2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "fuel_moisture_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bbox_min_lon", sa.Float(), nullable=False),
        sa.Column("bbox_min_lat", sa.Float(), nullable=False),
        sa.Column("bbox_max_lon", sa.Float(), nullable=False),
        sa.Column("bbox_max_lat", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("provider", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "ix_fuel_moisture_runs_run_time",
        "fuel_moisture_runs",
        ["run_time"],
    )
    op.create_index(
        "ix_fuel_moisture_runs_bbox",
        "fuel_moisture_runs",
        ["bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_fuel_moisture_runs_bbox", table_name="fuel_moisture_runs")
    op.drop_index("ix_fuel_moisture_runs_run_time", table_name="fuel_moisture_runs")
    op.drop_table("fuel_moisture_runs")
