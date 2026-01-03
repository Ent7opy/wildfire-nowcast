"""Feature engineering for FIRMS hotspot denoiser."""

import numpy as np
import pandas as pd
import xarray as xr
from typing import List
from sqlalchemy import text
from sqlalchemy.engine import Engine
from api.core.grid import DEFAULT_CELL_SIZE_DEG
from api.terrain.dem_loader import load_dem_for_bbox
from api.terrain.features_loader import load_slope_aspect_for_bbox

def add_firms_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical FIRMS signals."""
    df = df.copy()
    
    # confidence mapping
    if "confidence" in df.columns:
        # high/nominal/low -> numeric proxy
        # If it's already numeric, we keep it. If it's string, we map it.
        if df["confidence"].dtype == object:
            mapping = {"high": 90.0, "nominal": 60.0, "low": 30.0}
            df["confidence_norm"] = df["confidence"].map(mapping).fillna(30.0)
        else:
            df["confidence_norm"] = df["confidence"].fillna(30.0)
    
    # FRP and brightness
    # Note: frp, brightness, bright_t31 are already in the table
    if "brightness" in df.columns and "bright_t31" in df.columns:
        df["brightness_minus_t31"] = df["brightness"] - df["bright_t31"]
    
    # daynight - check raw_properties if not in columns
    if "daynight" not in df.columns and "raw_properties" in df.columns:
        df["daynight"] = df["raw_properties"].apply(lambda x: x.get("daynight") if isinstance(x, dict) else None)
    
    if "daynight" in df.columns:
        df["is_day"] = (df["daynight"] == "D").astype(int)
    
    # instrument/satellite
    for col in ["instrument", "satellite"]:
        if col not in df.columns and "raw_properties" in df.columns:
            df[col] = df["raw_properties"].apply(
                lambda x: x.get(col) if isinstance(x, dict) else None
            )
             
        if col in df.columns:
            df[f"{col}_code"] = df[col].astype("category").cat.codes

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features (cyclical encoding)."""
    df = df.copy()
    if "acq_time" not in df.columns:
        return df
    
    ts = pd.to_datetime(df["acq_time"])
    
    # Hour of day
    hour = ts.dt.hour + ts.dt.minute / 60.0
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    
    # Day of year
    doy = ts.dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    
    return df

def add_spatiotemporal_context(
    df: pd.DataFrame, 
    engine: Engine,
    radii_km: List[float] = [2.0, 5.0],
    windows_hours: List[int] = [6, 24],
    grid_size: float = DEFAULT_CELL_SIZE_DEG
) -> pd.DataFrame:
    """Add local spatiotemporal context using past-only detections."""
    df = df.copy()
    df["acq_time"] = pd.to_datetime(df["acq_time"])
    
    results = []
    
    for _, row in df.iterrows():
        detection_id = row.get("id")
        lat = row["lat"]
        lon = row["lon"]
        ts = row["acq_time"]
        
        row_context = {}
        
        with engine.connect() as conn:
            # Radius queries
            for r_km in radii_km:
                for w_h in windows_hours:
                    query = text("""
                        SELECT COUNT(*) 
                        FROM fire_detections 
                        WHERE acq_time <= :ts 
                          AND acq_time >= :ts - interval ':w_h hours'
                          AND id != :id
                          AND ST_DWithin(
                              geom, 
                              ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, 
                              :r_m
                          )
                    """)
                    # Using string interpolation for interval because bindparam with interval can be tricky
                    query_str = str(query).replace(":w_h hours", f"{w_h} hours")
                    count = conn.execute(text(query_str), {
                        "ts": ts,
                        "id": detection_id,
                        "lon": lon,
                        "lat": lat,
                        "r_m": r_km * 1000
                    }).scalar()
                    row_context[f"n_{int(r_km)}km_{w_h}h"] = count
            
            # Additional required counts
            if "n_5km_24h" not in row_context:
                # This should be covered if 5.0 and 24 are in defaults
                pass

            # Grid-based counts
            i_lat = int(np.floor(lat / grid_size))
            j_lon = int(np.floor(lon / grid_size))
            
            for w_d, label in [(1, "24h"), (7, "7d")]:
                query_grid = text(f"""
                    SELECT COUNT(*) 
                    FROM fire_detections 
                    WHERE acq_time <= :ts 
                      AND acq_time >= :ts - interval '{w_d} days'
                      AND id != :id
                      AND floor(lat / :grid_size) = :i_lat
                      AND floor(lon / :grid_size) = :j_lon
                """)
                count_grid = conn.execute(query_grid, {
                    "ts": ts,
                    "id": detection_id,
                    "grid_size": grid_size,
                    "i_lat": i_lat,
                    "j_lon": j_lon
                }).scalar()
                row_context[f"n_same_cell_{label}"] = count_grid
            
            # dist_nn_24h_km
            query_nn = text("""
                SELECT ST_Distance(
                    geom, 
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
                ) / 1000.0 as dist_km
                FROM fire_detections 
                WHERE acq_time <= :ts 
                  AND acq_time >= :ts - interval '24 hours'
                  AND id != :id
                ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
                LIMIT 1
            """)
            dist_nn = conn.execute(query_nn, {
                "ts": ts,
                "id": detection_id,
                "lon": lon,
                "lat": lat
            }).scalar()
            row_context["dist_nn_24h_km"] = dist_nn if dist_nn is not None else 999.0
            
            # seen_same_cell_past_3d
            query_seen = text("""
                SELECT EXISTS (
                    SELECT 1 FROM fire_detections 
                    WHERE acq_time < :ts 
                      AND acq_time >= :ts - interval '3 days'
                      AND floor(lat / :grid_size) = :i_lat
                      AND floor(lon / :grid_size) = :j_lon
                )
            """)
            seen = conn.execute(query_seen, {
                "ts": ts,
                "grid_size": grid_size,
                "i_lat": i_lat,
                "j_lon": j_lon
            }).scalar()
            row_context["seen_same_cell_past_3d"] = int(seen)
            
            # days_with_detection_past_30d_in_cell
            query_days = text("""
                SELECT COUNT(DISTINCT date(acq_time))
                FROM fire_detections 
                WHERE acq_time < :ts 
                  AND acq_time >= :ts - interval '30 days'
                  AND floor(lat / :grid_size) = :i_lat
                  AND floor(lon / :grid_size) = :j_lon
            """)
            days_count = conn.execute(query_days, {
                "ts": ts,
                "grid_size": grid_size,
                "i_lat": i_lat,
                "j_lon": j_lon
            }).scalar()
            row_context["days_with_detection_past_30d_in_cell"] = days_count

        results.append(row_context)
    
    context_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), context_df], axis=1)

def add_spatiotemporal_context_batch(
    df: pd.DataFrame,
    engine: Engine,
    radii_km: List[float] = [2.0, 5.0],
    windows_hours: List[int] = [6, 24],
    grid_size: float = DEFAULT_CELL_SIZE_DEG,
) -> pd.DataFrame:
    """
    Add local spatiotemporal context using past-only detections.
    Uses set-based SQL for efficiency on larger batches.
    """
    if df.empty:
        return df

    # We need 'id', 'lat', 'lon', 'acq_time' to compute features
    required = {"id", "lat", "lon", "acq_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for spatiotemporal context: {missing}")

    df = df.copy()
    df["acq_time"] = pd.to_datetime(df["acq_time"])

    # For small batches, we could still use the per-row one, but let's stick to set-based.
    # We'll build a temporary table or a large CTE with the batch IDs/coords.
    # To keep it simple and portable without temp tables, we'll use a JOIN with a VALUES list.

    # 1. Prepare the values list for SQL
    # We use numeric lats/lons and timestamps
    values_list = []
    for idx, row in df.iterrows():
        values_list.append(
            f"({row['id']}, {row['lat']}, {row['lon']}, '{row['acq_time'].isoformat()}'::timestamptz)"
        )
    values_sql = ",\n".join(values_list)

    # 2. Build the big query.
    # We'll compute counts and distances for all points in the batch in a single pass.
    # This uses a lateral join to compute local stats per input point.

    # Build parts of the query for each radius/window combination
    count_cols = []
    for r_km in radii_km:
        for w_h in windows_hours:
            col_name = f"n_{int(r_km)}km_{w_h}h"
            count_cols.append(f"""
                (
                    SELECT COUNT(*)
                    FROM fire_detections fd
                    WHERE fd.acq_time <= b.acq_time
                      AND fd.acq_time >= b.acq_time - interval '{w_h} hours'
                      AND fd.id != b.id
                      AND ST_DWithin(
                          fd.geom,
                          ST_SetSRID(ST_MakePoint(b.lon, b.lat), 4326)::geography,
                          {r_km * 1000}
                      )
                ) AS {col_name}
            """)

    # Grid-based counts
    for w_d, label in [(1, "24h"), (7, "7d")]:
        col_name = f"n_same_cell_{label}"
        count_cols.append(f"""
            (
                SELECT COUNT(*)
                FROM fire_detections fd
                WHERE fd.acq_time <= b.acq_time
                  AND fd.acq_time >= b.acq_time - interval '{w_d} days'
                  AND fd.id != b.id
                  AND floor(fd.lat / {grid_size}) = floor(b.lat / {grid_size})
                  AND floor(fd.lon / {grid_size}) = floor(b.lon / {grid_size})
            ) AS {col_name}
        """)

    # dist_nn_24h_km
    count_cols.append("""
        COALESCE(
            (
                SELECT ST_Distance(
                    fd.geom,
                    ST_SetSRID(ST_MakePoint(b.lon, b.lat), 4326)::geography
                ) / 1000.0
                FROM fire_detections fd
                WHERE fd.acq_time <= b.acq_time
                  AND fd.acq_time >= b.acq_time - interval '24 hours'
                  AND fd.id != b.id
                ORDER BY fd.geom <-> ST_SetSRID(ST_MakePoint(b.lon, b.lat), 4326)
                LIMIT 1
            ),
            999.0
        ) AS dist_nn_24h_km
    """)

    # seen_same_cell_past_3d
    count_cols.append(f"""
        (
            SELECT EXISTS (
                SELECT 1 FROM fire_detections fd
                WHERE fd.acq_time < b.acq_time
                  AND fd.acq_time >= b.acq_time - interval '3 days'
                  AND floor(fd.lat / {grid_size}) = floor(b.lat / {grid_size})
                  AND floor(fd.lon / {grid_size}) = floor(b.lon / {grid_size})
            )
        )::int AS seen_same_cell_past_3d
    """)

    # days_with_detection_past_30d_in_cell
    count_cols.append(f"""
        (
            SELECT COUNT(DISTINCT date(fd.acq_time))
            FROM fire_detections fd
            WHERE fd.acq_time < b.acq_time
              AND fd.acq_time >= b.acq_time - interval '30 days'
              AND floor(fd.lat / {grid_size}) = floor(b.lat / {grid_size})
              AND floor(fd.lon / {grid_size}) = floor(b.lon / {grid_size})
        ) AS days_with_detection_past_30d_in_cell
    """)

    count_cols_sql = ",\n            ".join(count_cols)
    query = text(f"""
        WITH batch(id, lat, lon, acq_time) AS (
            VALUES {values_sql}
        )
        SELECT
            b.id,
            {count_cols_sql}
        FROM batch b
    """)

    with engine.connect() as conn:
        context_df = pd.read_sql(query, conn)

    return pd.merge(df, context_df, on="id", how="left")

def add_terrain_features(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    """Add terrain features (elevation, slope)."""
    df = df.copy()
    if df.empty:
        df["elevation_m"] = []
        df["slope_deg"] = []
        return df

    min_lon, min_lat = df["lon"].min(), df["lat"].min()
    max_lon, max_lat = df["lon"].max(), df["lat"].max()
    # Add a small buffer to the bbox
    bbox = (min_lon - 0.05, min_lat - 0.05, max_lon + 0.05, max_lat + 0.05)
    
    try:
        dem = load_dem_for_bbox(region_name, bbox)
        slope, aspect = load_slope_aspect_for_bbox(region_name, bbox)
        
        # Interpolate
        lats = xr.DataArray(df["lat"].values, dims="z")
        lons = xr.DataArray(df["lon"].values, dims="z")
        
        df["elevation_m"] = dem.interp(lat=lats, lon=lons, method="linear").values
        df["slope_deg"] = slope.interp(lat=lats, lon=lons, method="linear").values
        
        # aspect sin/cos
        asp_val = aspect.interp(lat=lats, lon=lons, method="linear").values
        df["aspect_sin"] = np.sin(np.radians(asp_val))
        df["aspect_cos"] = np.cos(np.radians(asp_val))
        
    except Exception as e:
        print(f"Warning: Could not load terrain features for region {region_name}: {e}")
        df["elevation_m"] = np.nan
        df["slope_deg"] = np.nan
        df["aspect_sin"] = np.nan
        df["aspect_cos"] = np.nan
        
    return df
