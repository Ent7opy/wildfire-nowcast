"""Ingest industrial sources for denoiser masking."""

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import pandas as pd
import httpx
from sqlalchemy import text, JSON, bindparam
from ingest.repository import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("industrial_ingest")

WRI_POWER_PLANT_URL = "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv"

def download_wri_power_plants() -> pd.DataFrame:
    """Download the WRI Global Power Plant Database."""
    LOGGER.info(f"Downloading WRI Power Plant DB from {WRI_POWER_PLANT_URL}...")
    try:
        response = httpx.get(WRI_POWER_PLANT_URL, timeout=60.0)
        response.raise_for_status()
        # Save to temp file
        os.makedirs("data/ingest", exist_ok=True)
        temp_path = "data/ingest/global_power_plant_database.csv"
        with open(temp_path, "wb") as f:
            f.write(response.content)
        
        df = pd.read_csv(temp_path, low_memory=False)
        return df
    except Exception as e:
        LOGGER.error(f"Failed to download WRI Power Plant DB: {e}")
        return pd.DataFrame()

def ingest_industrial_sources(
    df: pd.DataFrame,
    source_name: str,
    source_version: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    type_col: str = "primary_fuel",
    name_col: str = "name",
    lat_col: str = "latitude",
    lon_col: str = "longitude"
):
    """Filter and load industrial sources into DB."""
    if df.empty:
        LOGGER.warning(f"No data to ingest for {source_name}")
        return

    # Filter by BBox if provided
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        df = df[
            (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat) &
            (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon)
        ]
        LOGGER.info(f"Filtered to bbox {bbox}: {len(df)} sources remaining.")

    if df.empty:
        LOGGER.warning(f"No sources in {source_name} intersect with bbox.")
        return

    engine = get_engine()
    
    # Prepare batch
    batch = []
    for _, row in df.iterrows():
        # IMPORTANT: pass a Python dict to SQLAlchemy's JSON binder.
        # Using row.to_json() would double-serialize (JSON string -> JSON), storing a quoted string.
        meta = row.where(pd.notnull(row), None).to_dict()
        # Normalize numpy/pandas scalars to plain Python types for JSON serialization.
        for k, v in list(meta.items()):
            if hasattr(v, "item"):
                try:
                    meta[k] = v.item()
                except Exception:
                    pass
        batch.append({
            "name": str(row[name_col]),
            "type": str(row[type_col]),
            "source": source_name,
            "source_version": source_version,
            "lat": float(row[lat_col]),
            "lon": float(row[lon_col]),
            "meta": meta
        })

    insert_stmt = text("""
        INSERT INTO industrial_sources (name, type, source, source_version, geom, meta)
        VALUES (:name, :type, :source, :source_version, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), :meta)
    """).bindparams(bindparam("meta", type_=JSON))

    with engine.begin() as conn:
        # Clear existing for this source/version to avoid duplicates if re-running
        conn.execute(text("DELETE FROM industrial_sources WHERE source = :s AND source_version = :v"), 
                     {"s": source_name, "v": source_version})
        
        # Batch insert
        batch_size = 1000
        for i in range(0, len(batch), batch_size):
            conn.execute(insert_stmt, batch[i:i+batch_size])
    
    LOGGER.info(f"Successfully ingested {len(batch)} sources from {source_name}")

def main():
    parser = argparse.ArgumentParser(description="Ingest industrial sources for denoiser.")
    parser.add_argument("--bbox", type=float, nargs=4, help="min_lon min_lat max_lon max_lat")
    parser.add_argument("--wri", action="store_true", help="Ingest WRI Power Plant DB")
    parser.add_argument("--csv", type=str, help="Path to custom CSV file")
    parser.add_argument("--source-name", type=str, help="Name of custom source")
    parser.add_argument("--source-version", type=str, default="v1", help="Version of custom source")
    
    args = parser.parse_args()
    bbox = tuple(args.bbox) if args.bbox else None

    if args.wri:
        df = download_wri_power_plants()
        ingest_industrial_sources(df, "WRI_Power_Plants", "v1.3.2", bbox=bbox)
    
    if args.csv:
        if not args.source_name:
            LOGGER.error("--source-name is required when using --csv")
            sys.exit(1)
        df = pd.read_csv(args.csv)
        ingest_industrial_sources(df, args.source_name, args.source_version, bbox=bbox)

if __name__ == "__main__":
    main()

