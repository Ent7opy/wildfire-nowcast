"""drop legacy fire_labels, schema_meta, and unused geocoder extensions

fire_labels was the v1 labeling table, superseded by denoiser_labels_v2
(which adds event linkage and per-rule-version uniqueness).
schema_meta was created in the initial migration but never used or populated.
postgis_tiger_geocoder and postgis_topology were installed as part of the
PostGIS setup but are not used — geocoding goes through an external API
cached in reverse_geocode_cache.

Revision ID: 20260315_cleanup_legacy_tables_and_extensions
Revises: 20260312_add_fire_geom_provenance
Create Date: 2026-03-15 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260315_cleanup_legacy_tables_and_extensions"
down_revision: Union[str, Sequence[str], None] = "20260312_add_fire_geom_provenance"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop legacy v1 label table (superseded by denoiser_labels_v2)
    op.drop_index("ix_fire_labels_rule_version_labeled_at", table_name="fire_labels")
    op.drop_index("ix_fire_labels_label_rule_version", table_name="fire_labels")
    op.drop_table("fire_labels")

    # Drop schema_meta — never populated, no code references
    op.drop_table("schema_meta")

    # Drop unused PostGIS extensions (Tiger geocoder + topology).
    # CASCADE removes the tiger, tiger_data, and topology schemas and all
    # their tables (34 Tiger tables + 2 topology tables).
    op.execute("DROP EXTENSION IF EXISTS postgis_tiger_geocoder CASCADE")
    op.execute("DROP EXTENSION IF EXISTS postgis_topology CASCADE")
    op.execute("DROP EXTENSION IF EXISTS fuzzystrmatch CASCADE")


def downgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch")
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology")
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder")

    op.create_table(
        "schema_meta",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("app_version", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="schema_meta_pkey"),
    )

    op.create_table(
        "fire_labels",
        sa.Column("fire_detection_id", sa.BigInteger(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("rule_version", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("rule_params", postgresql.JSONB(), nullable=True),
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
        ),
        sa.PrimaryKeyConstraint("fire_detection_id", name="fire_labels_pkey"),
    )
    op.create_index(
        "ix_fire_labels_label_rule_version",
        "fire_labels",
        ["label", "rule_version"],
    )
    op.create_index(
        "ix_fire_labels_rule_version_labeled_at",
        "fire_labels",
        ["rule_version", "labeled_at"],
    )
