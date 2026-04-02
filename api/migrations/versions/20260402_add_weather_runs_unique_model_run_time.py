"""Add UNIQUE constraint on weather_runs(model, run_time) to prevent duplicates.

Hourly HRRR ingest re-runs or retries would otherwise insert duplicate
(model, run_time) rows because create_weather_run_record used a plain INSERT
with no conflict handling.  The constraint + ON CONFLICT DO NOTHING guard in
the repository layer make the operation idempotent.

Revision ID: 20260402_add_weather_runs_unique_model_run_time
Revises: 20260401_add_weather_point_cache
Create Date: 2026-04-02
"""

from alembic import op

revision = "20260402_add_weather_runs_unique_model_run_time"
down_revision = "20260401_add_weather_point_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_weather_runs_model_run_time",
        "weather_runs",
        ["model", "run_time"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_weather_runs_model_run_time", "weather_runs")
