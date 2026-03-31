"""Add drought_index_runs and ignition_lightning_proxy tables for ignition probability signals.

drought_index_runs  — tracks Copernicus GDO raster ingest runs (one row per
                      successful weekly fetch).  Mirrors the fuel_moisture_runs
                      pattern used by lfmc_ecland_ingest.

ignition_lightning_proxy — materialised staging table refreshed each ingest
                           cycle with a boolean thunderstorm_active flag per
                           grid cell, derived from MeteoAlarm thunderstorm
                           warnings.  Schema is intentionally stable so future
                           upgrades to the lightning source do not require
                           feature extraction changes.

Revision ID: 20260331_add_ignition_signals
Revises: 20260330_add_ne_populated_places
Create Date: 2026-03-31 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260331_add_ignition_signals"
down_revision: Union[str, Sequence[str], None] = "20260330_add_ne_populated_places"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ------------------------------------------------------------------ #
    # drought_index_runs                                                   #
    # ------------------------------------------------------------------ #
    op.create_table(
        "drought_index_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("valid_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bbox_min_lon", sa.Float(), nullable=False),
        sa.Column("bbox_min_lat", sa.Float(), nullable=False),
        sa.Column("bbox_max_lon", sa.Float(), nullable=False),
        sa.Column("bbox_max_lat", sa.Float(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("provider", sa.String(length=255), nullable=True),
        sa.Column("variable", sa.String(length=255), nullable=True),
        sa.Column("coverage_fraction", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_drought_index_runs_valid_time",
        "drought_index_runs",
        ["valid_time"],
    )
    op.create_index(
        "ix_drought_index_runs_provider_status",
        "drought_index_runs",
        ["provider", "status"],
    )

    # ------------------------------------------------------------------ #
    # ignition_lightning_proxy                                             #
    # ------------------------------------------------------------------ #
    op.create_table(
        "ignition_lightning_proxy",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("grid_lon", sa.Float(), nullable=False),
        sa.Column("grid_lat", sa.Float(), nullable=False),
        sa.Column("thunderstorm_active", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("valid_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_ignition_lightning_proxy_valid_time",
        "ignition_lightning_proxy",
        ["valid_time"],
    )
    op.create_index(
        "ix_ignition_lightning_proxy_grid",
        "ignition_lightning_proxy",
        ["grid_lon", "grid_lat"],
    )
    op.create_index(
        "ix_ignition_lightning_proxy_active",
        "ignition_lightning_proxy",
        ["thunderstorm_active"],
        postgresql_where=sa.text("thunderstorm_active = true"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_ignition_lightning_proxy_active", table_name="ignition_lightning_proxy")
    op.drop_index("ix_ignition_lightning_proxy_grid", table_name="ignition_lightning_proxy")
    op.drop_index("ix_ignition_lightning_proxy_valid_time", table_name="ignition_lightning_proxy")
    op.drop_table("ignition_lightning_proxy")

    op.drop_index("ix_drought_index_runs_provider_status", table_name="drought_index_runs")
    op.drop_index("ix_drought_index_runs_valid_time", table_name="drought_index_runs")
    op.drop_table("drought_index_runs")
