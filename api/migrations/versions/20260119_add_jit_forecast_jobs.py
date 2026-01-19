"""add jit_forecast_jobs table

Revision ID: 20260119_jit_jobs
Revises: d46889070598
Create Date: 2026-01-19 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '20260119_jit_jobs'
down_revision: Union[str, Sequence[str], None] = 'd46889070598'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


class Geometry(sa.types.UserDefinedType):
    """Minimal PostGIS geometry type helper for migrations."""

    def __init__(self, geometry_type: str, srid: int) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw: object) -> str:
        return f"geometry({self.geometry_type}, {self.srid})"


def upgrade() -> None:
    op.create_table(
        "jit_forecast_jobs",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("bbox", Geometry("POLYGON", 4326), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("request", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_jit_forecast_jobs_status_created_at",
        "jit_forecast_jobs",
        ["status", "created_at"],
    )
    op.create_index(
        "ix_jit_forecast_jobs_bbox",
        "jit_forecast_jobs",
        ["bbox"],
        postgresql_using="gist",
    )


def downgrade() -> None:
    op.drop_index("ix_jit_forecast_jobs_bbox", table_name="jit_forecast_jobs")
    op.drop_index("ix_jit_forecast_jobs_status_created_at", table_name="jit_forecast_jobs")
    op.drop_table("jit_forecast_jobs")
