"""add false_source_masked column

Revision ID: 20260119_add_false_source_masked
Revises: 20260119_add_landcover_score
Create Date: 2026-01-19 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260119_add_false_source_masked'
down_revision: Union[str, Sequence[str], None] = '20260119_add_landcover_score'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "fire_detections",
        sa.Column("false_source_masked", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("fire_detections", "false_source_masked")
