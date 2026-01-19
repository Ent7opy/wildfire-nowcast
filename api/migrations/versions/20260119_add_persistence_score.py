"""add persistence_score column

Revision ID: 20260119_add_persistence_score
Revises: b8e90cee90c8
Create Date: 2026-01-19 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260119_add_persistence_score'
down_revision: Union[str, Sequence[str], None] = 'b8e90cee90c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "fire_detections",
        sa.Column("persistence_score", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("fire_detections", "persistence_score")
