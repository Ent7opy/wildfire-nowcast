"""empty message

Revision ID: 2c8fb90854b0
Revises: 1ef68811334b
Create Date: 2025-11-23 14:16:51.882680

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = '2c8fb90854b0'
down_revision: Union[str, Sequence[str], None] = '1ef68811334b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
