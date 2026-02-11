"""add spread forecast cache-key index

Revision ID: 20260211_add_spread_forecast_cache_key_index
Revises: 20260130_add_fire_perimeters
Create Date: 2026-02-11 21:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260211_add_spread_forecast_cache_key_index"
down_revision: Union[str, Sequence[str], None] = "20260130_add_fire_perimeters"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_spread_forecast_runs_cache_key_recent
        ON spread_forecast_runs ((metadata->>'cache_key'), created_at DESC)
        WHERE status = 'completed' AND metadata ? 'cache_key'
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_spread_forecast_runs_cache_key_recent")
