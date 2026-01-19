"""add landcover_score column

Revision ID: 20260119_add_landcover_score
Revises: 20260119_add_persistence_score
Create Date: 2026-01-19 20:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260119_add_landcover_score'
down_revision: Union[str, Sequence[str], None] = '20260119_add_persistence_score'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "fire_detections",
        sa.Column("landcover_score", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("fire_detections", "landcover_score")
