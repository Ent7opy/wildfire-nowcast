"""initial_schema_postgis

Revision ID: 1ef68811334b
Revises: 
Create Date: 2025-11-23 14:05:22.326756

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1ef68811334b'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable PostGIS extension
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

    # Create schema_meta table to track schema versions and metadata
    op.create_table(
        'schema_meta',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('app_version', sa.String(50), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop the schema_meta table
    op.drop_table('schema_meta')

    # Note: We don't drop the PostGIS extension in downgrade
    # as it may be used by other schemas or applications
