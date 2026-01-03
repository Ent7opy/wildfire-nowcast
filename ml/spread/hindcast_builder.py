"""Hindcast builder for producing predicted vs observed spread datasets."""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from api.db import get_engine
from api.fires.service import get_fire_cells_heatmap
from ml.spread.contract import SpreadModel
from ml.spread.heuristic_v0 import HeuristicSpreadModelV0, HeuristicSpreadV0Config
from ml.spread.hindcast_dataset import sample_fire_reference_times
from ml.spread_features import build_spread_inputs

LOGGER = logging.getLogger(__name__)


def group_reference_times_into_events(
    ref_times: List[datetime], interval_hours: int
) -> List[List[datetime]]:
    """Group contiguous reference times into 'events'."""
    if not ref_times:
        return []

    # Sort to be sure
    sorted_times = sorted(ref_times)
    events = []
    current_event = [sorted_times[0]]

    # Gap threshold: 1.5x interval to allow for tiny drifts or single missing buckets
    gap_threshold = timedelta(hours=interval_hours * 1.5)

    for i in range(1, len(sorted_times)):
        if sorted_times[i] - sorted_times[i - 1] <= gap_threshold:
            current_event.append(sorted_times[i])
        else:
            events.append(current_event)
            current_event = [sorted_times[i]]
    events.append(current_event)

    return events


def build_hindcast_case(
    region_name: str,
    bbox: Tuple[float, float, float, float],
    ref_time: datetime,
    horizons_hours: List[int],
    model: SpreadModel,
    label_window_hours: int = 3,
) -> xr.Dataset:
    """Build a single hindcast case: inputs + predictions + observations."""
    # 1. Build inputs
    inputs = build_spread_inputs(
        region_name=region_name,
        bbox=bbox,
        forecast_reference_time=ref_time,
        horizons_hours=horizons_hours,
    )

    # 2. Run prediction
    forecast = model.predict(inputs.to_model_input())

    # 3. Gather observations
    obs_list = []
    for h in horizons_hours:
        target_time = ref_time + timedelta(hours=h)
        target_start = target_time - timedelta(hours=label_window_hours)
        target_end = target_time + timedelta(hours=label_window_hours)

        obs_heatmap = get_fire_cells_heatmap(
            region_name=region_name,
            bbox=bbox,
            start_time=target_start,
            end_time=target_end,
            mode="presence",
            clip=True,
        ).heatmap
        obs_list.append(obs_heatmap)

    obs_array = np.stack(obs_list, axis=0)  # (time, lat, lon)

    # 4. Package into xarray Dataset
    # We want to preserve input features if possible for calibration (e.g. wind)
    # But for MVP we focus on y_pred and y_obs.
    
    # Coordinates from inputs.window
    ds = xr.Dataset(
        data_vars={
            "y_pred": (["time", "lat", "lon"], forecast.probabilities.values.astype(np.float32)),
            "y_obs": (["time", "lat", "lon"], obs_array.astype(np.float32)),
            "fire_t0": (["lat", "lon"], inputs.active_fires.heatmap.astype(np.float32)),
            "slope": (["lat", "lon"], inputs.terrain.slope.astype(np.float32)),
            "aspect": (["lat", "lon"], inputs.terrain.aspect.astype(np.float32)),
        },
        coords={
            "time": pd.to_datetime([ref_time + timedelta(hours=h) for h in horizons_hours]),
            "lat": inputs.window.lat,
            "lon": inputs.window.lon,
            "lead_time_hours": ("time", list(horizons_hours)),
        },
        attrs={
            "region": region_name,
            "bbox": list(bbox),
            "ref_time": ref_time.isoformat(),
            "model": model.__class__.__name__,
        }
    )

    # Add weather if available in cube
    if "u10" in inputs.weather_cube:
        ds["u10"] = (["time", "lat", "lon"], inputs.weather_cube.u10.values.astype(np.float32))
    if "v10" in inputs.weather_cube:
        ds["v10"] = (["time", "lat", "lon"], inputs.weather_cube.v10.values.astype(np.float32))

    return ds


def run_hindcast_builder(config: Dict[str, Any]):
    """Run the full hindcast building pipeline."""
    # 1. Setup
    region_name = config["region_name"]
    bbox = tuple(config["bbox"])
    start_time = datetime.fromisoformat(config["start_time"]).replace(tzinfo=timezone.utc)
    end_time = datetime.fromisoformat(config["end_time"]).replace(tzinfo=timezone.utc)
    horizons_hours = config["horizons_hours"]
    output_root = Path(config.get("output_root", "data/hindcasts"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Selection criteria
    min_detections = config.get("min_detections", 5)
    interval_hours = config.get("interval_hours", 24)
    min_active_cells_t0 = config.get("min_active_cells_t0", 5)
    min_event_buckets = config.get("min_event_buckets", 1)
    label_window_h = config.get("label_window_hours", 3)

    # 3. Sample times
    engine = get_engine()
    candidate_times = sample_fire_reference_times(
        engine, bbox, start_time, end_time, min_detections, interval_hours
    )
    LOGGER.info(f"Found {len(candidate_times)} candidate reference times.")

    # 4. Group into events
    events = group_reference_times_into_events(candidate_times, interval_hours)
    LOGGER.info(f"Grouped into {len(events)} potential events.")

    # 5. Filter events by duration
    filtered_events = [e for e in events if len(e) >= min_event_buckets]
    LOGGER.info(f"Filtered to {len(filtered_events)} events meeting min_event_buckets={min_event_buckets}.")

    # 6. Build model
    # For now, only support HeuristicSpreadModelV0
    # In the future we can add model loading logic if needed
    model_name = config.get("model_name", "HeuristicSpreadModelV0")
    model_params = config.get("model_params", {})

    if model_name == "HeuristicSpreadModelV0":
        # Check for unknown params to avoid silent failures
        valid_fields = set(HeuristicSpreadV0Config.__annotations__.keys())
        unknown = set(model_params.keys()) - valid_fields
        if unknown:
            LOGGER.warning(f"Ignoring unknown model_params for HeuristicSpreadModelV0: {unknown}")
        
        # Filter params
        valid_params = {k: v for k, v in model_params.items() if k in valid_fields}
        model_config = HeuristicSpreadV0Config(**valid_params)
        model = HeuristicSpreadModelV0(config=model_config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    manifest = []

    # 7. Process cases
    for event_idx, event_times in enumerate(filtered_events):
        event_id = f"event_{event_idx:03d}"
        event_dir = run_dir / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        for ref_time in event_times:
            try:
                LOGGER.info(f"Processing {event_id} ref_time={ref_time.isoformat()}...")
                ds = build_hindcast_case(
                    region_name, bbox, ref_time, horizons_hours, model, label_window_h
                )

                # Filter by min_active_cells_t0
                n_active_t0 = (ds.fire_t0 > 0).sum().item()
                if n_active_t0 < min_active_cells_t0:
                    LOGGER.info(f"Skipping case with only {n_active_t0} active cells at T=0.")
                    continue

                filename = f"ref_{ref_time.strftime('%Y%m%dT%H%M%S')}Z.nc"
                out_path = event_dir / filename
                ds.to_netcdf(out_path)

                # Add to manifest
                manifest.append({
                    "event_id": event_id,
                    "ref_time": ref_time.isoformat(),
                    "path": str(out_path.relative_to(output_root.parent.parent if output_root.is_absolute() else Path.cwd())),
                    "n_active_t0": int(n_active_t0),
                    "n_obs_positive": int((ds.y_obs > 0).sum().item()),
                })

            except Exception:
                LOGGER.exception(f"Failed to build hindcast case for {ref_time}; skipping.")

    # 8. Save manifest
    manifest_path = run_dir / "index.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "config": config,
            "cases": manifest
        }, f, indent=2)

    LOGGER.info(f"Hindcast build complete. Saved {len(manifest)} cases to {run_dir}.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build predicted vs observed hindcast dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_hindcast_builder(config)


if __name__ == "__main__":
    main()

