"""Add compound indexes for cursor-based (keyset) pagination on fire endpoints.

Cursor pagination on fire_detections uses ORDER BY acq_time, id.
A compound index on (acq_time, id) lets PostgreSQL satisfy both the keyset
predicate and the sort in a single index scan, eliminating sequential scans.

Cursor pagination on fire_events uses ORDER BY COALESCE(start_time, end_time) DESC,
event_id DESC.  A compound index on (start_time, event_id) supports the common case
where start_time is non-NULL (covering both sort key and predicate).

Cursor pagination on fire_fronts uses ORDER BY COALESCE(overpass_end, overpass_start)
DESC NULLS LAST, front_id DESC.  A compound index on (overpass_end, front_id) supports
the non-NULL case.

EXPLAIN validation queries (run against a populated DB):

  -- Detections forward page:
  EXPLAIN SELECT id, acq_time FROM fire_detections
    WHERE acq_time BETWEEN '2026-01-01' AND '2026-01-08'
      AND (acq_time > '2026-01-03' OR (acq_time = '2026-01-03' AND id > 5000))
    ORDER BY acq_time ASC, id ASC LIMIT 1001;
  -- Expected: Index Scan using ix_fire_detections_acq_time_id

  -- Events cursor page:
  EXPLAIN SELECT event_id, start_time FROM fire_events
    WHERE start_time <= '2026-01-08' AND end_time >= '2026-01-01'
      AND (COALESCE(start_time, end_time) < '2026-01-05'
           OR COALESCE(start_time, end_time) IS NULL
           OR (COALESCE(start_time, end_time) = '2026-01-05' AND event_id < 'evt_abc'))
    ORDER BY COALESCE(start_time, end_time) DESC, event_id DESC LIMIT 1001;
  -- Expected: Index Scan using ix_fire_events_start_time_event_id

  -- Fronts cursor page:
  EXPLAIN SELECT front_id, overpass_end FROM fire_fronts
    WHERE (COALESCE(overpass_end, overpass_start) < '2026-01-05'
           OR COALESCE(overpass_end, overpass_start) IS NULL
           OR (COALESCE(overpass_end, overpass_start) = '2026-01-05'
               AND front_id < 'front_xyz'))
    ORDER BY COALESCE(overpass_end, overpass_start) DESC NULLS LAST, front_id DESC LIMIT 801;
  -- Expected: Index Scan using ix_fire_fronts_overpass_end_front_id

Revision ID: 20260327_add_cursor_pagination_indexes
Revises: 20260326_add_lulc_freshness_index
Create Date: 2026-03-27 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260327_add_cursor_pagination_indexes"
down_revision: Union[str, Sequence[str], None] = "20260326_add_lulc_freshness_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # fire_detections: compound index on (acq_time, id) for cursor keyset scan.
    # ORDER BY acq_time ASC, id ASC / DESC — both directions covered by this B-tree.
    op.create_index(
        "ix_fire_detections_acq_time_id",
        "fire_detections",
        ["acq_time", "id"],
    )

    # fire_events: compound index on (start_time, event_id) to support the common
    # (non-NULL start_time) cursor predicate and ORDER BY COALESCE(start_time, end_time).
    op.create_index(
        "ix_fire_events_start_time_event_id",
        "fire_events",
        ["start_time", "event_id"],
        postgresql_ops={"start_time": "DESC", "event_id": "DESC"},
    )

    # fire_fronts: compound index on (overpass_end DESC NULLS LAST, front_id DESC) to support
    # cursor predicate and ORDER BY COALESCE(overpass_end, overpass_start) DESC NULLS LAST.
    # Raw SQL is used here because Alembic's create_index does not support NULLS LAST
    # expressions in the columns list.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_fire_fronts_overpass_end_front_id "
        "ON fire_fronts (overpass_end DESC NULLS LAST, front_id DESC)"
    )

    op.execute("ANALYZE fire_detections")
    op.execute("ANALYZE fire_events")
    op.execute("ANALYZE fire_fronts")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_fire_fronts_overpass_end_front_id")
    op.drop_index("ix_fire_events_start_time_event_id", table_name="fire_events")
    op.drop_index("ix_fire_detections_acq_time_id", table_name="fire_detections")
