"""add spread forecast products

Revision ID: 20251228_spread_forecasts
Revises: 4757a7deda9f
Create Date: 2025-12-28 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


class Geometry(sa.types.UserDefinedType):
    """Minimal PostGIS geometry type helper for migrations."""

    def __init__(self, geometry_type: str, srid: int) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw: object) -> str:
        return f"geometry({self.geometry_type}, {self.srid})"


# revision identifiers, used by Alembic.
revision: str = '20251228_spread_forecasts'
down_revision: Union[str, Sequence[str], None] = '4757a7deda9f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create spread forecast products tables."""
    op.create_table(
        "spread_forecast_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("bbox", Geometry("POLYGON", 4326), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("forecast_reference_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("region_name", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_spread_forecast_runs_bbox",
        "spread_forecast_runs",
        ["bbox"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_spread_forecast_runs_forecast_reference_time",
        "spread_forecast_runs",
        ["forecast_reference_time"],
    )

    op.create_table(
        "spread_forecast_rasters",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("horizon_hours", sa.Integer(), nullable=False),
        sa.Column("file_format", sa.Text(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["spread_forecast_runs.id"],
            name="fk_spread_forecast_rasters_run_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_spread_forecast_rasters_run_id",
        "spread_forecast_rasters",
        ["run_id"],
    )

    op.create_table(
        "spread_forecast_contours",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("horizon_hours", sa.Integer(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("geom", Geometry("MULTIPOLYGON", 4326), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["spread_forecast_runs.id"],
            name="fk_spread_forecast_contours_run_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_spread_forecast_contours_run_id",
        "spread_forecast_contours",
        ["run_id"],
    )
    op.create_index(
        "ix_spread_forecast_contours_geom",
        "spread_forecast_contours",
        ["geom"],
        postgresql_using="gist",
    )


def downgrade() -> None:
    """Drop spread forecast products tables."""
    op.drop_index("ix_spread_forecast_contours_geom", table_name="spread_forecast_contours")
    op.drop_index("ix_spread_forecast_contours_run_id", table_name="spread_forecast_contours")
    op.drop_table("spread_forecast_contours")
    op.drop_index("ix_spread_forecast_rasters_run_id", table_name="spread_forecast_rasters")
    op.drop_table("spread_forecast_rasters")
    op.drop_index("ix_spread_forecast_runs_forecast_reference_time", table_name="spread_forecast_runs")
    op.drop_index("ix_spread_forecast_runs_bbox", table_name="spread_forecast_runs")
    op.drop_table("spread_forecast_runs")

