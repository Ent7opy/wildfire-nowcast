"""Add query indexes: fire_likelihood, is_archive, denoiser_labels_v2.fire_detection_id

Three btree indexes added to support common hot-path queries:

  1. ix_fire_detections_fire_likelihood
     -- fire_detections(fire_likelihood) WHERE fire_likelihood IS NOT NULL
     -- partial index: excludes NULL rows (never satisfy >= threshold) to reduce
     --   index size and improve range-scan selectivity
     -- used by GET /fires and mvt_fires with min_fire_likelihood filter:
     --   WHERE fire_likelihood IS NULL OR fire_likelihood >= :min_fire_likelihood
     -- NULL arm resolved by seq-scan fallback; non-NULL arm uses this index

  2. ix_fire_detections_is_archive
     -- fire_detections(is_archive)
     -- used by db_cleanup.py TTL sweeps:
     --   WHERE acq_time < :cutoff AND is_archive = false  (NRT sweep)
     --   WHERE acq_time < :cutoff AND is_archive = true   (archive sweep)
     -- complements the existing partial index
     -- ix_fire_detections_archive_acq_time (acq_time WHERE is_archive=true)

  3. ix_denoiser_labels_v2_fire_detection_id
     -- denoiser_labels_v2(fire_detection_id)
     -- used by snapshot export JOIN:
     --   JOIN denoiser_labels_v2 l ON d.id = l.fire_detection_id
     -- also accelerates CASCADE DELETE on fire_detections.id
     -- the existing composite ix_denoiser_labels_v2_rule_detection
     -- (rule_version, fire_detection_id) cannot serve single-column lookups

EXPLAIN validation queries (run against a populated DB):
  EXPLAIN SELECT id, acq_time, fire_likelihood FROM fire_detections
    WHERE fire_likelihood >= 0.5 ORDER BY acq_time DESC LIMIT 100;
  -- Expected: Index Scan using ix_fire_detections_fire_likelihood

  EXPLAIN SELECT id, acq_time FROM fire_detections
    WHERE acq_time < now() - interval '14 days' AND is_archive = false LIMIT 1000;
  -- Expected: Bitmap Index Scan on ix_fire_detections_is_archive

  EXPLAIN SELECT d.id, l.label FROM fire_detections d
    JOIN denoiser_labels_v2 l ON d.id = l.fire_detection_id
    WHERE d.acq_time > now() - interval '7 days';
  -- Expected: Index Scan using ix_denoiser_labels_v2_fire_detection_id

Revision ID: 20260325_add_query_indexes
Revises: 20260325_add_is_archive
Create Date: 2026-03-25 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260325_add_query_indexes"
down_revision: Union[str, Sequence[str], None] = "20260325_add_is_archive"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_fire_detections_fire_likelihood",
        "fire_detections",
        ["fire_likelihood"],
        postgresql_where=sa.text("fire_likelihood IS NOT NULL"),
    )
    op.create_index(
        "ix_fire_detections_is_archive",
        "fire_detections",
        ["is_archive"],
    )
    op.create_index(
        "ix_denoiser_labels_v2_fire_detection_id",
        "denoiser_labels_v2",
        ["fire_detection_id"],
    )
    op.execute("ANALYZE fire_detections")
    op.execute("ANALYZE denoiser_labels_v2")


def downgrade() -> None:
    op.drop_index("ix_denoiser_labels_v2_fire_detection_id", table_name="denoiser_labels_v2")
    op.drop_index("ix_fire_detections_is_archive", table_name="fire_detections")
    op.drop_index("ix_fire_detections_fire_likelihood", table_name="fire_detections")
