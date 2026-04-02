"""Review queue → denoiser training feedback loop.

Adds the schema required to close the learning loop from operator review queue
resolutions back into denoiser v2 training labels:

  1. label_weight (FLOAT, default 1.0) on denoiser_labels_v2 — allows
     per-label confidence weighting so review-queue labels (default 0.8) can
     be down-weighted relative to authoritative perimeter labels (1.0).

  2. label_conflicts — audit table recording cases where an operator label
     disagrees with an existing authoritative-perimeter label.  Perimeter
     always wins; the conflict is logged here for QA inspection.

  3. operator_label_quality — weekly-computed per-operator accuracy against
     subsequent authoritative perimeters.  Used to weight labels in training:
     high-accuracy operators → 1.0, low-accuracy → 0.5.  Initially only one
     row exists because the UI sends resolved_by='operator' for all resolutions.

Revision ID: 20260402_review_queue_feedback_loop
Revises: 20260402_add_weather_runs_unique_model_run_time
Create Date: 2026-04-02
"""

from alembic import op
import sqlalchemy as sa

revision = "20260402_review_queue_feedback_loop"
down_revision = "20260402_add_weather_runs_unique_model_run_time"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add label_weight to denoiser_labels_v2
    op.add_column(
        "denoiser_labels_v2",
        sa.Column(
            "label_weight",
            sa.Float(),
            nullable=False,
            server_default=sa.text("1.0"),
        ),
    )

    # Index on source for efficient review_queue label lookups
    op.create_index(
        "ix_denoiser_labels_v2_source",
        "denoiser_labels_v2",
        ["source"],
    )
    # Composite index to support the NOT EXISTS subquery in label_review_queue.py:
    #   WHERE dl.fire_detection_id = fem.fire_detection_id AND dl.source = 'review_queue'
    op.create_index(
        "ix_denoiser_labels_v2_detection_source",
        "denoiser_labels_v2",
        ["fire_detection_id", "source"],
    )

    # 2. label_conflicts — audit log for operator vs. perimeter disagreements
    op.create_table(
        "label_conflicts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.String(length=64), nullable=True),
        sa.Column("fire_detection_id", sa.BigInteger(), nullable=True),
        sa.Column("perimeter_label", sa.String(length=32), nullable=False),
        sa.Column("operator_label", sa.String(length=32), nullable=False),
        sa.Column("resolved_by", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["fire_events.event_id"],
            name="fk_label_conflicts_event",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["fire_detection_id"],
            ["fire_detections.id"],
            name="fk_label_conflicts_detection",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_label_conflicts_event_id",
        "label_conflicts",
        ["event_id"],
    )
    op.create_index(
        "ix_label_conflicts_created_at",
        "label_conflicts",
        ["created_at"],
    )

    # 3. operator_label_quality — per-operator accuracy computed weekly
    # Unique on resolved_by: currently only one row ('operator') until auth is added.
    op.create_table(
        "operator_label_quality",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("resolved_by", sa.Text(), nullable=False),
        sa.Column("fire_label_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("fire_correct_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("noise_label_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("noise_correct_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("fire_accuracy", sa.Float(), nullable=True),
        sa.Column("noise_accuracy", sa.Float(), nullable=True),
        # Derived training weight: min(1.0, max(0.5, avg(fire_accuracy, noise_accuracy)))
        sa.Column("label_weight", sa.Float(), nullable=False, server_default=sa.text("1.0")),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("resolved_by", name="uq_operator_label_quality_resolved_by"),
    )


def downgrade() -> None:
    op.drop_table("operator_label_quality")

    op.drop_index("ix_label_conflicts_created_at", table_name="label_conflicts")
    op.drop_index("ix_label_conflicts_event_id", table_name="label_conflicts")
    op.drop_table("label_conflicts")

    op.drop_index("ix_denoiser_labels_v2_detection_source", table_name="denoiser_labels_v2")
    op.drop_index("ix_denoiser_labels_v2_source", table_name="denoiser_labels_v2")
    op.drop_column("denoiser_labels_v2", "label_weight")
