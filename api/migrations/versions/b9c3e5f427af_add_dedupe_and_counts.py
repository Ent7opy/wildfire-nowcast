"""Add dedupe hash and ingest batch metrics.

Revision ID: b9c3e5f427af
Revises: 37962c109cd5
Create Date: 2025-12-04 10:30:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "b9c3e5f427af"
down_revision = "37962c109cd5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ingest_batches",
        sa.Column("records_fetched", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "ingest_batches",
        sa.Column("records_inserted", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "ingest_batches",
        sa.Column(
            "records_skipped_duplicates",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )

    op.add_column(
        "fire_detections",
        sa.Column("dedupe_hash", sa.String(length=64), nullable=False, server_default=""),
    )
    op.execute(
        sa.text(
            """
            UPDATE fire_detections
            SET dedupe_hash = md5(
                coalesce(source, '') || '|' ||
                round(lat::numeric, 4)::text || '|' ||
                round(lon::numeric, 4)::text || '|' ||
                acq_time::text
            )
            """
        )
    )
    op.alter_column("fire_detections", "dedupe_hash", server_default=None)

    op.create_index(
        "uq_fire_detections_source_dedupe_hash",
        "fire_detections",
        ["source", "dedupe_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_fire_detections_source_dedupe_hash", table_name="fire_detections")
    op.drop_column("fire_detections", "dedupe_hash")

    op.drop_column("ingest_batches", "records_skipped_duplicates")
    op.drop_column("ingest_batches", "records_inserted")
    op.drop_column("ingest_batches", "records_fetched")

