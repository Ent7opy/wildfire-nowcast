"""add fire labels and industrial sources

Revision ID: 4757a7deda9f
Revises: 20251213_terrain_features
Create Date: 2025-12-23 21:12:29.134675

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

    def get_col_spec(self, **kw: object) -> str:
        return f"geometry({self.geometry_type}, {self.srid})"


# revision identifiers, used by Alembic.
revision: str = '4757a7deda9f'
down_revision: Union[str, Sequence[str], None] = '20251213_terrain_features'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "fire_labels",
        sa.Column("fire_detection_id", sa.BigInteger(), primary_key=True),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("rule_version", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("rule_params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "labeled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["fire_detection_id"],
            ["fire_detections.id"],
            name="fk_fire_labels_fire_detection_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_fire_labels_rule_version_labeled_at",
        "fire_labels",
        ["rule_version", "labeled_at"],
    )
    op.create_index(
        "ix_fire_labels_label_rule_version",
        "fire_labels",
        ["label", "rule_version"],
    )

    op.create_table(
        "industrial_sources",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("type", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("source_version", sa.Text(), nullable=True),
        sa.Column("geom", Geometry("POINT", 4326), nullable=False),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_industrial_sources_geom",
        "industrial_sources",
        ["geom"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_industrial_sources_source_version",
        "industrial_sources",
        ["source", "source_version"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_industrial_sources_source_version", table_name="industrial_sources")
    op.drop_index("ix_industrial_sources_geom", table_name="industrial_sources")
    op.drop_table("industrial_sources")
    op.drop_index("ix_fire_labels_label_rule_version", table_name="fire_labels")
    op.drop_index("ix_fire_labels_rule_version_labeled_at", table_name="fire_labels")
    op.drop_table("fire_labels")
