"""Add weather_point_cache table for DB-backed weather point lookups.

Stores GFS weather variable values at native 0.25° grid points near active fire
detections.  Both the fire-detail / scoring point-lookup path and the spread
model's weather-cube builder query this table directly, eliminating the need
for global NetCDF files.

The ``run_id`` FK cascades deletes from ``weather_runs`` so the periodic
``db_cleanup`` retention sweep automatically prunes point-cache rows when
their parent run expires.

Revision ID: 20260401_add_weather_point_cache
Revises: 20260331_add_ignition_signals
Create Date: 2026-04-01
"""

from alembic import op
import sqlalchemy as sa

revision = "20260401_add_weather_point_cache"
down_revision = "20260331_add_ignition_signals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "weather_point_cache",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.BigInteger,
            sa.ForeignKey("weather_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("forecast_hour", sa.SmallInteger, nullable=False),
        sa.Column("lat_grid", sa.Float, nullable=False),
        sa.Column("lon_grid", sa.Float, nullable=False),
        sa.Column("u10", sa.Float, nullable=True),
        sa.Column("v10", sa.Float, nullable=True),
        sa.Column("t2m", sa.Float, nullable=True),
        sa.Column("rh2m", sa.Float, nullable=True),
        sa.Column("tp", sa.Float, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # Primary lookup: point + forecast hour, ordered by recency.
    op.create_index(
        "ix_wpc_lookup",
        "weather_point_cache",
        ["lat_grid", "lon_grid", "forecast_hour", "created_at"],
        postgresql_using="btree",
    )

    # FK index for cascade-delete performance.
    op.create_index(
        "ix_wpc_run_id",
        "weather_point_cache",
        ["run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_wpc_run_id", table_name="weather_point_cache")
    op.drop_index("ix_wpc_lookup", table_name="weather_point_cache")
    op.drop_table("weather_point_cache")
