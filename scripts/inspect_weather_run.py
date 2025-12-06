"""Helper to inspect the latest completed weather_run and its NetCDF payload."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint

import sqlalchemy as sa
import xarray as xr

from ingest.config import weather_settings
from ingest.repository import get_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect latest completed weather_run and optionally align run_time."
    )
    parser.add_argument(
        "--sync-run-time",
        action="store_true",
        help="Update run_time column to match metadata.run_time when they differ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = get_engine()
    with engine.connect() as conn:
        row = (
            conn.execute(
                sa.text(
                    "SELECT * FROM weather_runs "
                    "WHERE status = 'completed' "
                    "ORDER BY created_at DESC LIMIT 1"
                )
            )
            .mappings()
            .first()
        )
    row = dict(row) if row else None

    print("Latest completed weather_run:")
    pprint(dict(row) if row else None)
    print(
        "Configured defaults:",
        {
            "model": weather_settings.model_name,
            "run_time": weather_settings.run_time,
            "horizon_hours": weather_settings.horizon_hours,
            "step_hours": weather_settings.step_hours,
            "bbox": weather_settings.bbox,
        },
    )

    if not row:
        return

    metadata_run_time = None
    if row.get("metadata"):
        meta_rt = row["metadata"].get("run_time")
        if meta_rt:
            try:
                metadata_run_time = datetime.fromisoformat(meta_rt)
                if metadata_run_time.tzinfo is None:
                    metadata_run_time = metadata_run_time.replace(tzinfo=timezone.utc)
                else:
                    metadata_run_time = metadata_run_time.astimezone(timezone.utc)
            except Exception:
                metadata_run_time = None

    print("run_time column:", row["run_time"])
    print("metadata run_time:", metadata_run_time)

    if args.sync_run_time and metadata_run_time and metadata_run_time != row["run_time"]:
        with engine.begin() as conn:
            conn.execute(
                sa.text("UPDATE weather_runs SET run_time = :run_time WHERE id = :run_id"),
                {"run_time": metadata_run_time, "run_id": row["id"]},
            )
        row["run_time"] = metadata_run_time
        print("run_time updated to match metadata.")

    storage_path = Path(row["storage_path"])
    print(f"storage_path: {storage_path}")
    print("storage_path exists?", storage_path.exists())

    resolved_path = storage_path if storage_path.is_absolute() else (Path.cwd() / storage_path).resolve()
    if not storage_path.is_absolute():
        print("resolved path:", resolved_path)
        print("resolved path exists?", resolved_path.exists())

    if not resolved_path.exists():
        print("Dataset file is missing on disk; skipping open.")
        return

    ds = xr.open_dataset(resolved_path)
    print(ds)
    print("data_vars:", list(ds.data_vars))
    print("dims:", {k: int(v) for k, v in ds.sizes.items()})
    print(
        "lat range:",
        float(ds["lat"].min().values),
        float(ds["lat"].max().values),
    )
    print(
        "lon range:",
        float(ds["lon"].min().values),
        float(ds["lon"].max().values),
    )
    print("u10 @ time=0:")
    print(ds["u10"].isel(time=0))
    ds.close()


if __name__ == "__main__":
    main()

