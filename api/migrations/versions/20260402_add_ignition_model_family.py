"""Add 'ignition' to model_registry family CHECK constraint.

Extends the model_registry and model_promotions tables to accept
family='ignition' so that the ignition probability model can be registered
and promoted via the existing model registry workflow.

Revision ID: 20260402_add_ignition_model_family
Revises: 20260402_add_weather_runs_unique_model_run_time
Create Date: 2026-04-02 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260402_add_ignition_model_family"
down_revision: Union[str, Sequence[str], None] = "20260402_add_weather_runs_unique_model_run_time"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD_FAMILY_CHECK = "family IN ('denoiser', 'spread')"
_NEW_FAMILY_CHECK = "family IN ('denoiser', 'spread', 'ignition')"


def upgrade() -> None:
    """Extend the family CHECK constraint on model_registry and model_promotions."""
    # model_registry
    op.drop_constraint("ck_model_registry_family", "model_registry", type_="check")
    op.create_check_constraint(
        "ck_model_registry_family",
        "model_registry",
        _NEW_FAMILY_CHECK,
    )

    # model_promotions (if the constraint exists there too).
    with op.get_context().autocommit_block():
        conn = op.get_bind()
        result = conn.execute(
            sa.text(
                """
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_name = 'model_promotions'
                  AND constraint_type = 'CHECK'
                  AND constraint_name LIKE '%family%'
                """
            )
        ).fetchall()
        for row in result:
            conn.execute(
                sa.text(
                    f"ALTER TABLE model_promotions DROP CONSTRAINT IF EXISTS {row[0]}"
                )
            )

    op.create_check_constraint(
        "ck_model_promotions_family",
        "model_promotions",
        _NEW_FAMILY_CHECK,
    )


def downgrade() -> None:
    """Revert family CHECK constraint to exclude 'ignition'."""
    op.drop_constraint("ck_model_promotions_family", "model_promotions", type_="check")
    op.create_check_constraint(
        "ck_model_promotions_family",
        "model_promotions",
        _OLD_FAMILY_CHECK,
    )

    op.drop_constraint("ck_model_registry_family", "model_registry", type_="check")
    op.create_check_constraint(
        "ck_model_registry_family",
        "model_registry",
        _OLD_FAMILY_CHECK,
    )
