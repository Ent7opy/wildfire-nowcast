# Ingest Data Quality Checks

Quick reference for the lightweight validation and logging added to the ingest pipelines. Log messages use the shared pattern `[event] <message> | {json context}` and are emitted to the CLI stdout/stderr when running the ingest commands.

## FIRMS
- Reject rows with missing/non-numeric/out-of-range lat/lon; logged with `event=firms.validation`.
- Warn and drop confidence values outside 0–100; count missing confidence and brightness fields.
- Warn and drop brightness values outside 200–500 K (obvious outliers).
- Summary log (`firms.validation_summary`) after parsing includes total rows, parsed rows, skip counts, sensor counts, and confidence buckets (`low`, `nominal`, `high`, `unknown`).

## Weather (GFS)
- Verify expected variables are present for the run (u10, v10, t2m, rh2m, optional tp).
- Check that `lead_time_hours` spans the requested horizon (capped at 72h) and that the time dimension count matches the configured step.
- Log min/max/mean per variable plus the time range via `weather.stats`; coverage issues are flagged with `weather.validation`.

## DEM
- Compute coverage fraction (finite cells / total) and min/max elevation for the configured region.
- Warn via `dem.validation` when coverage drops below 90% or contains no finite cells.
- Stats are logged with `dem.stats` alongside the region label and bbox.

## Where to look
- Run `python -m ingest.firms_ingest ...`, `python -m ingest.weather_ingest ...`, or `python -m ingest.dem_preprocess ...` and watch the console output.
- The `event` tag at the start of each message makes it easy to filter for validation-related lines (e.g., grep for `firms.validation`).

## Related docs
- Pipelines: `ingest_firms.md`, `ingest_weather.md`, `ingest_dem.md`
- Outputs: `../data/data_formats.md`
- Doc hub: `../README.md`

