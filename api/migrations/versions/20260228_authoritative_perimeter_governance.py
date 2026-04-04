"""add authoritative perimeter governance tables and coverage provenance

Revision ID: 20260228_authoritative_perimeter_governance
Revises: 20260225_merge_migration_branches
Create Date: 2026-02-28 09:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


class Geometry(sa.types.UserDefinedType):
    """Minimal PostGIS geometry type helper for migrations."""

    def __init__(self, geometry_type: str, srid: int) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw: object) -> str:
        return f"geometry({self.geometry_type}, {self.srid})"


# revision identifiers, used by Alembic.
revision: str = "20260228_authoritative_perimeter_governance"
down_revision: Union[str, Sequence[str], None] = "20260225_merge_migration_branches"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "authoritative_perimeter_ingest_runs",
        sa.Column("run_id", sa.String(length=128), primary_key=True),
        sa.Column("source_profile", sa.String(length=64), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("source_layer", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("records_fetched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_upserted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_skipped", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("http_429_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_backoff_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_last_edit", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
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
    )
    op.create_index(
        "ix_authoritative_perimeter_ingest_runs_profile_finished",
        "authoritative_perimeter_ingest_runs",
        ["source_profile", "finished_at"],
    )
    op.create_index(
        "ix_authoritative_perimeter_ingest_runs_status",
        "authoritative_perimeter_ingest_runs",
        ["status"],
    )

    op.create_table(
        "authoritative_perimeters",
        sa.Column("perimeter_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("source_profile", sa.String(length=64), nullable=False),
        sa.Column("source_layer", sa.String(length=128), nullable=False),
        sa.Column("source_object_id", sa.String(length=128), nullable=False),
        sa.Column("poly_irwinid", sa.String(length=128), nullable=True),
        sa.Column("poly_sourceglobalid", sa.String(length=128), nullable=True),
        sa.Column("poly_featurestatus", sa.String(length=64), nullable=True),
        sa.Column("poly_featureaccess", sa.String(length=64), nullable=True),
        sa.Column("poly_isvisible", sa.String(length=32), nullable=True),
        sa.Column("attr_isvalid", sa.Integer(), nullable=True),
        sa.Column("attr_isquarantined", sa.Integer(), nullable=True),
        sa.Column("poly_source", sa.String(length=32), nullable=True),
        sa.Column("poly_mapmethod", sa.String(length=128), nullable=True),
        sa.Column("attr_firediscoverydatetime", sa.DateTime(timezone=True), nullable=True),
        sa.Column("poly_polygondatetime", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attr_containmentdatetime", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attr_controldatetime", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tier", sa.String(length=16), nullable=False, server_default="bronze"),
        sa.Column("is_authoritative", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("geom", Geometry("MULTIPOLYGON", 4326), nullable=False),
        sa.Column("raw_attributes", sa.JSON(), nullable=True),
        sa.Column("run_id", sa.String(length=128), nullable=True),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
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
            ["run_id"],
            ["authoritative_perimeter_ingest_runs.run_id"],
            name="fk_authoritative_perimeters_run_id",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint(
            "source_profile",
            "source_layer",
            "source_object_id",
            name="uq_authoritative_perimeters_source",
        ),
    )
    op.create_index(
        "ix_authoritative_perimeters_geom",
        "authoritative_perimeters",
        ["geom"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_authoritative_perimeters_authority_window",
        "authoritative_perimeters",
        ["is_authoritative", "tier", "poly_polygondatetime"],
    )
    op.create_index(
        "ix_authoritative_perimeters_profile_irwinid",
        "authoritative_perimeters",
        ["source_profile", "poly_irwinid"],
    )

    op.add_column(
        "perimeter_coverage_masks",
        sa.Column(
            "authority_profile",
            sa.String(length=64),
            nullable=False,
            server_default="wfigs_us",
        ),
    )
    op.add_column(
        "perimeter_coverage_masks",
        sa.Column(
            "tier_policy",
            sa.String(length=32),
            nullable=False,
            server_default="silver_gold",
        ),
    )
    op.add_column("perimeter_coverage_masks", sa.Column("run_id", sa.String(length=128), nullable=True))
    op.add_column("perimeter_coverage_masks", sa.Column("source_uri", sa.Text(), nullable=True))
    op.add_column("perimeter_coverage_masks", sa.Column("source_version", sa.String(length=128), nullable=True))
    op.add_column(
        "perimeter_coverage_masks",
        sa.Column("coverage_start", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "perimeter_coverage_masks",
        sa.Column("coverage_end", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "perimeter_coverage_masks",
        sa.Column("provenance_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
    )
    op.create_index(
        "ix_perimeter_coverage_masks_profile_window",
        "perimeter_coverage_masks",
        ["is_active", "authority_profile", "coverage_start", "coverage_end"],
    )
    op.create_foreign_key(
        "fk_perimeter_coverage_masks_run_id",
        source_table="perimeter_coverage_masks",
        referent_table="authoritative_perimeter_ingest_runs",
        local_cols=["run_id"],
        remote_cols=["run_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "fk_perimeter_coverage_masks_run_id",
        table_name="perimeter_coverage_masks",
        type_="foreignkey",
    )
    op.drop_index("ix_perimeter_coverage_masks_profile_window", table_name="perimeter_coverage_masks")
    op.drop_column("perimeter_coverage_masks", "provenance_json")
    op.drop_column("perimeter_coverage_masks", "coverage_end")
    op.drop_column("perimeter_coverage_masks", "coverage_start")
    op.drop_column("perimeter_coverage_masks", "source_version")
    op.drop_column("perimeter_coverage_masks", "source_uri")
    op.drop_column("perimeter_coverage_masks", "run_id")
    op.drop_column("perimeter_coverage_masks", "tier_policy")
    op.drop_column("perimeter_coverage_masks", "authority_profile")

    op.drop_index("ix_authoritative_perimeters_profile_irwinid", table_name="authoritative_perimeters")
    op.drop_index("ix_authoritative_perimeters_authority_window", table_name="authoritative_perimeters")
    op.drop_index("ix_authoritative_perimeters_geom", table_name="authoritative_perimeters")
    op.drop_table("authoritative_perimeters")

    op.drop_index(
        "ix_authoritative_perimeter_ingest_runs_status",
        table_name="authoritative_perimeter_ingest_runs",
    )
    op.drop_index(
        "ix_authoritative_perimeter_ingest_runs_profile_finished",
        table_name="authoritative_perimeter_ingest_runs",
    )
    op.drop_table("authoritative_perimeter_ingest_runs")
