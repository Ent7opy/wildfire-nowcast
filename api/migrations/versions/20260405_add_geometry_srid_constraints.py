"""Add SRID validation constraints for all geometry columns

This migration enforces that all geometry columns have SRID 4326 (WGS84).
A geometry stored with SRID -1 (unknown) would bypass spatial indexes
and return silently incorrect results.

Constraints added to tables:
  - fire_detections.geom
  - fire_events.geom (from denoiser v2)
  - fire_fronts.geom (from denoiser v2)
  - fire_perimeters.geom
  - perimeter_coverage_masks.geom
  - authoritative_perimeters.geom
  - aois.geom
  - aois.bbox
  - industrial_sources.geom
  - industrial_no_go_zones.geom
  - ne_populated_places.geom
  - spread_forecast_runs.bbox
  - spread_forecast_contours.geom
  - terrain_metadata.bbox
  - terrain_features_metadata.bbox
  - jit_forecast_jobs.bbox

Revision ID: 20260405_geometry_srid_constraints
Revises: 20260404_merge_three_heads
Create Date: 2026-04-05 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260405_geometry_srid_constraints"
down_revision: Union[str, Sequence[str], None] = "20260404_merge_three_heads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add CHECK constraints enforcing SRID 4326 for all geometry columns."""
    # All tables and their geometry columns requiring SRID validation
    geometry_columns = [
        ("fire_detections", "geom"),
        ("fire_events", "geom"),
        ("fire_fronts", "geom"),
        ("fire_perimeters", "geom"),
        ("perimeter_coverage_masks", "geom"),
        ("authoritative_perimeters", "geom"),
        ("aois", "geom"),
        ("aois", "bbox"),
        ("industrial_sources", "geom"),
        ("industrial_no_go_zones", "geom"),
        ("ne_populated_places", "geom"),
        ("spread_forecast_runs", "bbox"),
        ("spread_forecast_contours", "geom"),
        ("terrain_metadata", "bbox"),
        ("terrain_features_metadata", "bbox"),
        ("jit_forecast_jobs", "bbox"),
    ]

    for table, column in geometry_columns:
        constraint_name = f"ck_{table}_{column}_srid_4326"
        # Add CHECK constraint enforcing SRID 4326 for the geometry column
        # This prevents insertion of geometries with SRID -1 (unknown) or other non-4326 SRIDs
        op.create_check_constraint(
            constraint_name,
            table,
            f"ST_SRID({column}) = 4326",
        )


def downgrade() -> None:
    """Remove SRID validation constraints."""
    geometry_columns = [
        ("fire_detections", "geom"),
        ("fire_events", "geom"),
        ("fire_fronts", "geom"),
        ("fire_perimeters", "geom"),
        ("perimeter_coverage_masks", "geom"),
        ("authoritative_perimeters", "geom"),
        ("aois", "geom"),
        ("aois", "bbox"),
        ("industrial_sources", "geom"),
        ("industrial_no_go_zones", "geom"),
        ("ne_populated_places", "geom"),
        ("spread_forecast_runs", "bbox"),
        ("spread_forecast_contours", "geom"),
        ("terrain_metadata", "bbox"),
        ("terrain_features_metadata", "bbox"),
        ("jit_forecast_jobs", "bbox"),
    ]

    for table, column in geometry_columns:
        constraint_name = f"ck_{table}_{column}_srid_4326"
        op.drop_constraint(constraint_name, table, type_="check")
