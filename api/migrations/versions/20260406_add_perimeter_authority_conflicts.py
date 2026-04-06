"""Add authority_tier to fire_perimeters and perimeter_authority_conflicts audit table

Introduces authority-aware conflict resolution for perimeter ingestion:

1. ``fire_perimeters.authority_tier`` -- classifies each perimeter source
   (gold / silver / bronze / blocked) so upsert logic can reject lower-
   authority overwrites.
2. ``perimeter_authority_conflicts`` -- audit log of every rejected or
   accepted overwrite, enabling traceability of perimeter provenance.

Revision ID: 20260406_add_perimeter_authority_conflicts
Revises: 20260405_add_lfmc_remote_job_id
Create Date: 2026-04-06 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260406_add_perimeter_authority_conflicts"
down_revision: Union[str, Sequence[str], None] = "20260405_add_lfmc_remote_job_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Add authority_tier to legacy fire_perimeters table.
    op.add_column(
        "fire_perimeters",
        sa.Column(
            "authority_tier",
            sa.String(length=16),
            nullable=False,
            server_default="gold",
        ),
    )

    # 2. Audit table for perimeter authority conflicts.
    op.create_table(
        "perimeter_authority_conflicts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("table_name", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=128), nullable=False),
        sa.Column("incoming_tier", sa.String(length=16), nullable=False),
        sa.Column("existing_tier", sa.String(length=16), nullable=False),
        sa.Column(
            "outcome",
            sa.String(length=16),
            nullable=False,
            comment="accepted or rejected",
        ),
        sa.Column("run_id", sa.String(length=128), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_perimeter_authority_conflicts_source",
        "perimeter_authority_conflicts",
        ["table_name", "source", "source_id"],
    )
    op.create_index(
        "ix_perimeter_authority_conflicts_created",
        "perimeter_authority_conflicts",
        ["created_at"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_perimeter_authority_conflicts_created",
        table_name="perimeter_authority_conflicts",
    )
    op.drop_index(
        "ix_perimeter_authority_conflicts_source",
        table_name="perimeter_authority_conflicts",
    )
    op.drop_table("perimeter_authority_conflicts")
    op.drop_column("fire_perimeters", "authority_tier")
