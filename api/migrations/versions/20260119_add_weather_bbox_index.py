"""add spatial index on weather_runs bbox columns

Revision ID: 20260119_weather_bbox_index
Revises: 20260119_terrain_bbox_index
Create Date: 2026-01-19 13:01:00.000000
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260119_weather_bbox_index"
down_revision: Union[str, Sequence[str], None] = "20260119_terrain_bbox_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add functional GIST index on weather_runs bbox envelope."""
    op.execute(
        "CREATE INDEX ix_weather_runs_bbox "
        "ON weather_runs USING GIST ("
        "ST_MakeEnvelope(bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, 4326)"
        ")"
    )


def downgrade() -> None:
    """Drop functional GIST index on weather_runs bbox envelope."""
    op.drop_index("ix_weather_runs_bbox", table_name="weather_runs")
