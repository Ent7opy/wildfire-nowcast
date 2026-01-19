"""add confidence_score column

Revision ID: b8e90cee90c8
Revises: 20260119_weather_bbox_index
Create Date: 2026-01-19 19:59:26.090928

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b8e90cee90c8'
down_revision: Union[str, Sequence[str], None] = '20260119_weather_bbox_index'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "fire_detections",
        sa.Column("confidence_score", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("fire_detections", "confidence_score")
