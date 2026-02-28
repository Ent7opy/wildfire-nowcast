"""global authoritative industrial layer v1

Revision ID: 20260301_global_authoritative_industrial_v1
Revises: 20260228_authoritative_perimeter_governance
Create Date: 2026-03-01 00:00:00.000000
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
revision: str = "20260301_global_authoritative_industrial_v1"
down_revision: Union[str, Sequence[str], None] = "20260228_authoritative_perimeter_governance"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


AUTHORITY_TIER_CHECK = "authority_tier IN ('gold','silver','blocked')"
SECTOR_TAXONOMY_CHECK = "sector_taxonomy IN ('NACE','NAICS','ANZSIC','GBT4754','fuel_type','other')"
COORD_PRECISION_TYPE_CHECK = "coordinate_precision_type IN ('reported','derived','curated')"
VERIFICATION_MODE_CHECK = "verification_mode IN ('endpoint','curated','hybrid')"


def upgrade() -> None:
    op.create_table(
        "authoritative_industrial_ingest_runs",
        sa.Column("run_id", sa.String(length=128), primary_key=True),
        sa.Column("source_profile", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("records_fetched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_upserted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_skipped", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("source_version", sa.String(length=128), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "ix_authoritative_industrial_runs_profile_finished",
        "authoritative_industrial_ingest_runs",
        ["source_profile", "finished_at"],
    )
    op.create_index(
        "ix_authoritative_industrial_runs_status",
        "authoritative_industrial_ingest_runs",
        ["status"],
    )

    op.add_column("industrial_sources", sa.Column("source_profile", sa.Text(), nullable=True))
    op.add_column("industrial_sources", sa.Column("authority_name", sa.Text(), nullable=True))
    op.add_column("industrial_sources", sa.Column("authority_tier", sa.Text(), nullable=True))
    op.add_column("industrial_sources", sa.Column("country_iso3", sa.String(length=3), nullable=True))
    op.add_column("industrial_sources", sa.Column("jurisdiction_code", sa.Text(), nullable=True))
    op.add_column("industrial_sources", sa.Column("source_id", sa.Text(), nullable=True))
    op.add_column("industrial_sources", sa.Column("sector_code", sa.Text(), nullable=True))
    op.add_column("industrial_sources", sa.Column("sector_taxonomy", sa.Text(), nullable=True))
    op.add_column("industrial_sources", sa.Column("thermal_potential_class", sa.Numeric(6, 3), nullable=True))
    op.add_column("industrial_sources", sa.Column("coordinate_precision_type", sa.Text(), nullable=True))
    op.add_column("industrial_sources", sa.Column("coordinate_precision_m", sa.Numeric(10, 2), nullable=True))
    op.add_column("industrial_sources", sa.Column("verification_mode", sa.Text(), nullable=True))
    op.add_column("industrial_sources", sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True))
    op.add_column("industrial_sources", sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True))
    op.add_column("industrial_sources", sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "industrial_sources",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column("industrial_sources", sa.Column("run_id", sa.String(length=128), nullable=True))

    op.create_foreign_key(
        "fk_industrial_sources_run_id",
        source_table="industrial_sources",
        referent_table="authoritative_industrial_ingest_runs",
        local_cols=["run_id"],
        remote_cols=["run_id"],
        ondelete="SET NULL",
    )

    op.create_check_constraint(
        "ck_industrial_sources_authority_tier",
        "industrial_sources",
        AUTHORITY_TIER_CHECK,
    )
    op.create_check_constraint(
        "ck_industrial_sources_sector_taxonomy",
        "industrial_sources",
        SECTOR_TAXONOMY_CHECK,
    )
    op.create_check_constraint(
        "ck_industrial_sources_coordinate_precision_type",
        "industrial_sources",
        COORD_PRECISION_TYPE_CHECK,
    )
    op.create_check_constraint(
        "ck_industrial_sources_verification_mode",
        "industrial_sources",
        VERIFICATION_MODE_CHECK,
    )

    op.create_index(
        "ix_industrial_sources_profile_source_id",
        "industrial_sources",
        ["source_profile", "source_id"],
        unique=True,
        postgresql_where=sa.text("source_id IS NOT NULL"),
    )
    op.create_index(
        "ix_industrial_sources_country_tier_active",
        "industrial_sources",
        ["country_iso3", "authority_tier", "is_active"],
    )

    op.create_table(
        "industrial_no_go_zones",
        sa.Column("zone_id", sa.String(length=128), primary_key=True),
        sa.Column("zone_name", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("region_code", sa.Text(), nullable=True),
        sa.Column("geom", Geometry("MULTIPOLYGON", 4326), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("policy_version", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "ix_industrial_no_go_zones_geom",
        "industrial_no_go_zones",
        ["geom"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_industrial_no_go_zones_active_policy",
        "industrial_no_go_zones",
        ["is_active", "policy_version"],
    )

    op.create_table(
        "industrial_mask_policies",
        sa.Column("policy_version", sa.String(length=128), primary_key=True),
        sa.Column("strict_no_go", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("gold_buffer_m", sa.Numeric(10, 2), nullable=False),
        sa.Column("silver_buffer_min_m", sa.Numeric(10, 2), nullable=False),
        sa.Column("silver_buffer_max_m", sa.Numeric(10, 2), nullable=False),
        sa.Column("active_from", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("active_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "industrial_mask_audit",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("fire_detection_id", sa.BigInteger(), nullable=False),
        sa.Column("industrial_source_id", sa.BigInteger(), nullable=True),
        sa.Column("policy_version", sa.String(length=128), nullable=False),
        sa.Column("masked", sa.Boolean(), nullable=False),
        sa.Column("mask_reason", sa.Text(), nullable=False),
        sa.Column("matched_distance_m", sa.Numeric(10, 2), nullable=True),
        sa.Column("applied_buffer_m", sa.Numeric(10, 2), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(
            ["fire_detection_id"],
            ["fire_detections.id"],
            name="fk_industrial_mask_audit_fire_detection_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["industrial_source_id"],
            ["industrial_sources.id"],
            name="fk_industrial_mask_audit_industrial_source_id",
            ondelete="SET NULL",
        ),
    )
    op.create_index(
        "ix_industrial_mask_audit_detection_created",
        "industrial_mask_audit",
        ["fire_detection_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_industrial_mask_audit_detection_created", table_name="industrial_mask_audit")
    op.drop_table("industrial_mask_audit")

    op.drop_table("industrial_mask_policies")

    op.drop_index("ix_industrial_no_go_zones_active_policy", table_name="industrial_no_go_zones")
    op.drop_index("ix_industrial_no_go_zones_geom", table_name="industrial_no_go_zones")
    op.drop_table("industrial_no_go_zones")

    op.drop_index("ix_industrial_sources_country_tier_active", table_name="industrial_sources")
    op.drop_index("ix_industrial_sources_profile_source_id", table_name="industrial_sources")

    op.drop_constraint("ck_industrial_sources_verification_mode", "industrial_sources", type_="check")
    op.drop_constraint("ck_industrial_sources_coordinate_precision_type", "industrial_sources", type_="check")
    op.drop_constraint("ck_industrial_sources_sector_taxonomy", "industrial_sources", type_="check")
    op.drop_constraint("ck_industrial_sources_authority_tier", "industrial_sources", type_="check")

    op.drop_constraint("fk_industrial_sources_run_id", "industrial_sources", type_="foreignkey")

    op.drop_column("industrial_sources", "run_id")
    op.drop_column("industrial_sources", "is_active")
    op.drop_column("industrial_sources", "last_verified_at")
    op.drop_column("industrial_sources", "valid_to")
    op.drop_column("industrial_sources", "valid_from")
    op.drop_column("industrial_sources", "verification_mode")
    op.drop_column("industrial_sources", "coordinate_precision_m")
    op.drop_column("industrial_sources", "coordinate_precision_type")
    op.drop_column("industrial_sources", "thermal_potential_class")
    op.drop_column("industrial_sources", "sector_taxonomy")
    op.drop_column("industrial_sources", "sector_code")
    op.drop_column("industrial_sources", "source_id")
    op.drop_column("industrial_sources", "jurisdiction_code")
    op.drop_column("industrial_sources", "country_iso3")
    op.drop_column("industrial_sources", "authority_tier")
    op.drop_column("industrial_sources", "authority_name")
    op.drop_column("industrial_sources", "source_profile")

    op.drop_index(
        "ix_authoritative_industrial_runs_status",
        table_name="authoritative_industrial_ingest_runs",
    )
    op.drop_index(
        "ix_authoritative_industrial_runs_profile_finished",
        table_name="authoritative_industrial_ingest_runs",
    )
    op.drop_table("authoritative_industrial_ingest_runs")
