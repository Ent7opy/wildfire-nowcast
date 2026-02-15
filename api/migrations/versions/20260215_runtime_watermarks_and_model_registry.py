"""add ingest watermarks and model registry tables

Revision ID: 20260215_runtime_watermarks_and_model_registry
Revises: 20260211_add_spread_forecast_cache_key_index
Create Date: 2026-02-15 20:05:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20260215_runtime_watermarks_and_model_registry"
down_revision: Union[str, Sequence[str], None] = "20260211_add_spread_forecast_cache_key_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


FAMILY_CHECK = "family IN ('denoiser', 'spread')"


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "ingest_watermarks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("area_key", sa.String(length=255), nullable=False),
        sa.Column("last_acq_time_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_batch_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["last_batch_id"],
            ["ingest_batches.id"],
            name="fk_ingest_watermarks_last_batch_id",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint(
            "source",
            "area_key",
            name="uq_ingest_watermarks_source_area_key",
        ),
    )
    op.create_index(
        "ix_ingest_watermarks_updated_at",
        "ingest_watermarks",
        ["updated_at"],
    )

    op.create_table(
        "model_registry",
        sa.Column("model_id", sa.String(length=128), primary_key=True),
        sa.Column("family", sa.String(length=32), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column(
            "metrics_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="registered"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(FAMILY_CHECK, name="ck_model_registry_family"),
    )
    op.create_index(
        "ix_model_registry_family_created_at",
        "model_registry",
        ["family", "created_at"],
    )

    op.create_table(
        "model_promotions",
        sa.Column("family", sa.String(length=32), primary_key=True),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column(
            "promoted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("promoted_by", sa.Text(), nullable=True),
        sa.Column("rollback_model_id", sa.String(length=128), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["model_registry.model_id"],
            name="fk_model_promotions_model_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["rollback_model_id"],
            ["model_registry.model_id"],
            name="fk_model_promotions_rollback_model_id",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(FAMILY_CHECK, name="ck_model_promotions_family"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("model_promotions")
    op.drop_index("ix_model_registry_family_created_at", table_name="model_registry")
    op.drop_table("model_registry")
    op.drop_index("ix_ingest_watermarks_updated_at", table_name="ingest_watermarks")
    op.drop_table("ingest_watermarks")
