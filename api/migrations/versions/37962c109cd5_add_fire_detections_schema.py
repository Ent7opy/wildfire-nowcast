"""add fire detections schema

Revision ID: 37962c109cd5
Revises: 2c8fb90854b0
Create Date: 2025-12-02 12:08:42.598119

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


class Geometry(sa.types.UserDefinedType):
    """Minimal PostGIS geometry type helper for migrations."""

    def __init__(self, geometry_type: str, srid: int) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw: object) -> str:  # pragma: no cover - simple string format
        return f"geometry({self.geometry_type}, {self.srid})"


# revision identifiers, used by Alembic.
revision: str = '37962c109cd5'
down_revision: Union[str, Sequence[str], None] = '2c8fb90854b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "ingest_batches",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("record_count", sa.Integer(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "fire_detections",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("geom", Geometry("POINT", 4326), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("acq_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sensor", sa.String(length=32), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("brightness", sa.Float(), nullable=True),
        sa.Column("bright_t31", sa.Float(), nullable=True),
        sa.Column("frp", sa.Float(), nullable=True),
        sa.Column("scan", sa.Float(), nullable=True),
        sa.Column("track", sa.Float(), nullable=True),
        sa.Column(
            "raw_properties",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("denoised_score", sa.Float(), nullable=True),
        sa.Column("is_noise", sa.Boolean(), nullable=True),
        sa.Column("ingest_batch_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["ingest_batch_id"],
            ["ingest_batches.id"],
            name="fk_fire_detections_ingest_batch_id",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "lat BETWEEN -90 AND 90",
            name="ck_fire_detections_lat_bounds",
        ),
        sa.CheckConstraint(
            "lon BETWEEN -180 AND 180",
            name="ck_fire_detections_lon_bounds",
        ),
    )
    op.create_index(
        "ix_fire_detections_geom",
        "fire_detections",
        ["geom"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_fire_detections_acq_time",
        "fire_detections",
        ["acq_time"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_fire_detections_acq_time", table_name="fire_detections")
    op.drop_index("ix_fire_detections_geom", table_name="fire_detections")
    op.drop_table("fire_detections")
    op.drop_table("ingest_batches")
