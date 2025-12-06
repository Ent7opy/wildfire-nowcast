"""add weather_runs table

Revision ID: 20251206
Revises: b9c3e5f427af
Create Date: 2025-12-06 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20251206"
down_revision: Union[str, Sequence[str], None] = "b9c3e5f427af"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create weather_runs table."""
    op.create_table(
        "weather_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("run_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_hours", sa.Integer(), nullable=False),
        sa.Column("step_hours", sa.Integer(), nullable=False),
        sa.Column("bbox_min_lon", sa.Float(), nullable=True),
        sa.Column("bbox_min_lat", sa.Float(), nullable=True),
        sa.Column("bbox_max_lon", sa.Float(), nullable=True),
        sa.Column("bbox_max_lat", sa.Float(), nullable=True),
        sa.Column("file_format", sa.Text(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Drop weather_runs table."""
    op.drop_table("weather_runs")

