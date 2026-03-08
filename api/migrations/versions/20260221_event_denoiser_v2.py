"""add event-level denoiser v2 schema

Revision ID: 20260221_event_denoiser_v2
Revises: 20260216_fix_mvt_fires_zoom_floor
Create Date: 2026-02-21 23:10:00.000000
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
revision: str = "20260221_event_denoiser_v2"
down_revision: Union[str, Sequence[str], None] = "20260216_fix_mvt_fires_zoom_floor"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


DENOISER_DECISIONS_CHECK = "denoiser_decision IS NULL OR denoiser_decision IN ('pass', 'downweight', 'drop', 'review')"
REVIEW_STATUS_CHECK = "status IN ('open', 'resolved', 'dismissed')"


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "fire_fronts",
        sa.Column("front_id", sa.String(length=64), primary_key=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.Column("sensor", sa.String(length=32), nullable=True),
        sa.Column("overpass_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("overpass_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("detection_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("frp_max", sa.Float(), nullable=True),
        sa.Column("frp_mean", sa.Float(), nullable=True),
        sa.Column("confidence_max", sa.Float(), nullable=True),
        sa.Column("geom", Geometry("GEOMETRY", 4326), nullable=True),
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
        "ix_fire_fronts_geom",
        "fire_fronts",
        ["geom"],
        postgresql_using="gist",
    )

    op.create_table(
        "fire_events",
        sa.Column("event_id", sa.String(length=64), primary_key=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.Column("sensor", sa.String(length=32), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("detection_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("front_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("event_score", sa.Float(), nullable=True),
        sa.Column("denoiser_decision", sa.String(length=16), nullable=True),
        sa.Column("review_required", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("geom", Geometry("GEOMETRY", 4326), nullable=True),
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
        sa.CheckConstraint(DENOISER_DECISIONS_CHECK, name="ck_fire_events_decision"),
    )
    op.create_index(
        "ix_fire_events_geom",
        "fire_events",
        ["geom"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_fire_events_time",
        "fire_events",
        ["start_time", "end_time"],
    )

    op.create_table(
        "fire_event_memberships",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("fire_detection_id", sa.BigInteger(), nullable=False),
        sa.Column("front_id", sa.String(length=64), nullable=True),
        sa.Column("event_id", sa.String(length=64), nullable=True),
        sa.Column("member_role", sa.String(length=32), nullable=False, server_default="member"),
        sa.Column(
            "linked_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["fire_detection_id"],
            ["fire_detections.id"],
            name="fk_fire_event_memberships_detection",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["front_id"],
            ["fire_fronts.front_id"],
            name="fk_fire_event_memberships_front",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["fire_events.event_id"],
            name="fk_fire_event_memberships_event",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("fire_detection_id", name="uq_fire_event_memberships_detection"),
    )
    op.create_index(
        "ix_fire_event_memberships_event_front",
        "fire_event_memberships",
        ["event_id", "front_id"],
    )

    op.create_table(
        "denoiser_labels_v2",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("fire_detection_id", sa.BigInteger(), nullable=True),
        sa.Column("event_id", sa.String(length=64), nullable=True),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("rule_version", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("rule_params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("weak_supervision", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "labeled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["fire_detection_id"],
            ["fire_detections.id"],
            name="fk_denoiser_labels_v2_detection",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["fire_events.event_id"],
            name="fk_denoiser_labels_v2_event",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("fire_detection_id", "rule_version", name="uq_denoiser_labels_v2_detection_rule"),
    )
    op.create_index(
        "ix_denoiser_labels_v2_rule_labeled",
        "denoiser_labels_v2",
        ["rule_version", "labeled_at"],
    )
    op.create_index(
        "ix_denoiser_labels_v2_label_rule",
        "denoiser_labels_v2",
        ["label", "rule_version"],
    )

    op.create_table(
        "denoiser_eval_runs",
        sa.Column("run_id", sa.String(length=128), primary_key=True),
        sa.Column("model_id", sa.String(length=128), nullable=True),
        sa.Column("family", sa.String(length=32), nullable=False, server_default="denoiser"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("gate_report_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("slice_metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("artifact_uri", sa.Text(), nullable=True),
        sa.Column(
            "evaluated_at",
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
    )
    op.create_index(
        "ix_denoiser_eval_runs_evaluated_at",
        "denoiser_eval_runs",
        ["evaluated_at"],
    )

    op.create_table(
        "denoiser_drift_metrics",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("model_id", sa.String(length=128), nullable=True),
        sa.Column("metric_name", sa.String(length=64), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("threshold_value", sa.Float(), nullable=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("triggered_rollback", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_denoiser_drift_metrics_created",
        "denoiser_drift_metrics",
        ["created_at"],
    )
    op.create_index(
        "ix_denoiser_drift_metrics_metric",
        "denoiser_drift_metrics",
        ["metric_name", "created_at"],
    )

    op.create_table(
        "denoiser_review_queue",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.String(length=64), nullable=True),
        sa.Column("fire_detection_id", sa.BigInteger(), nullable=True),
        sa.Column("reason", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False, server_default="medium"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="open"),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("resolved_by", sa.Text(), nullable=True),
        sa.Column("resolved_notes", sa.Text(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
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
            ["event_id"],
            ["fire_events.event_id"],
            name="fk_denoiser_review_queue_event",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["fire_detection_id"],
            ["fire_detections.id"],
            name="fk_denoiser_review_queue_detection",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(REVIEW_STATUS_CHECK, name="ck_denoiser_review_queue_status"),
    )
    op.create_index(
        "ix_denoiser_review_queue_status_created",
        "denoiser_review_queue",
        ["status", "created_at"],
    )
    op.create_index(
        "ix_denoiser_review_queue_event_status",
        "denoiser_review_queue",
        ["event_id", "status"],
    )

    op.add_column("fire_detections", sa.Column("front_id", sa.String(length=64), nullable=True))
    op.add_column("fire_detections", sa.Column("event_id", sa.String(length=64), nullable=True))
    op.add_column("fire_detections", sa.Column("event_score", sa.Float(), nullable=True))
    op.add_column("fire_detections", sa.Column("denoiser_decision", sa.String(length=16), nullable=True))
    op.add_column(
        "fire_detections",
        sa.Column("review_required", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column("fire_detections", sa.Column("denoiser_model_id", sa.String(length=128), nullable=True))
    op.add_column("fire_detections", sa.Column("denoiser_scored_at", sa.DateTime(timezone=True), nullable=True))

    op.create_foreign_key(
        "fk_fire_detections_front_id",
        "fire_detections",
        "fire_fronts",
        ["front_id"],
        ["front_id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_fire_detections_event_id",
        "fire_detections",
        "fire_events",
        ["event_id"],
        ["event_id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "ck_fire_detections_denoiser_decision",
        "fire_detections",
        DENOISER_DECISIONS_CHECK,
    )

    op.create_index("ix_fire_detections_event_id", "fire_detections", ["event_id"])
    op.create_index(
        "ix_fire_detections_acq_decision_review",
        "fire_detections",
        ["acq_time", "denoiser_decision", "review_required"],
    )
    op.create_index(
        "ix_fire_detections_event_time",
        "fire_detections",
        ["event_id", "acq_time"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_fire_detections_event_time", table_name="fire_detections")
    op.drop_index("ix_fire_detections_acq_decision_review", table_name="fire_detections")
    op.drop_index("ix_fire_detections_event_id", table_name="fire_detections")
    op.drop_constraint("ck_fire_detections_denoiser_decision", "fire_detections", type_="check")
    op.drop_constraint("fk_fire_detections_event_id", "fire_detections", type_="foreignkey")
    op.drop_constraint("fk_fire_detections_front_id", "fire_detections", type_="foreignkey")

    op.drop_column("fire_detections", "denoiser_scored_at")
    op.drop_column("fire_detections", "denoiser_model_id")
    op.drop_column("fire_detections", "review_required")
    op.drop_column("fire_detections", "denoiser_decision")
    op.drop_column("fire_detections", "event_score")
    op.drop_column("fire_detections", "event_id")
    op.drop_column("fire_detections", "front_id")

    op.drop_index("ix_denoiser_review_queue_event_status", table_name="denoiser_review_queue")
    op.drop_index("ix_denoiser_review_queue_status_created", table_name="denoiser_review_queue")
    op.drop_table("denoiser_review_queue")

    op.drop_index("ix_denoiser_drift_metrics_metric", table_name="denoiser_drift_metrics")
    op.drop_index("ix_denoiser_drift_metrics_created", table_name="denoiser_drift_metrics")
    op.drop_table("denoiser_drift_metrics")

    op.drop_index("ix_denoiser_eval_runs_evaluated_at", table_name="denoiser_eval_runs")
    op.drop_table("denoiser_eval_runs")

    op.drop_index("ix_denoiser_labels_v2_label_rule", table_name="denoiser_labels_v2")
    op.drop_index("ix_denoiser_labels_v2_rule_labeled", table_name="denoiser_labels_v2")
    op.drop_table("denoiser_labels_v2")

    op.drop_index("ix_fire_event_memberships_event_front", table_name="fire_event_memberships")
    op.drop_table("fire_event_memberships")

    op.drop_index("ix_fire_events_time", table_name="fire_events")
    op.drop_index("ix_fire_events_geom", table_name="fire_events")
    op.drop_table("fire_events")

    op.drop_index("ix_fire_fronts_geom", table_name="fire_fronts")
    op.drop_table("fire_fronts")
