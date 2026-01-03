"""CLI for building FIRMS denoiser features."""

import argparse
import sys
import pandas as pd
from api.db import get_engine
from .dataset import load_labeled_data, build_dataset

def main():
    parser = argparse.ArgumentParser(description="Build features for FIRMS hotspot denoiser.")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--aoi", type=str, required=True, help="Region name for terrain features")
    parser.add_argument("--out", type=str, default="data/denoiser/features.parquet", help="Output path")
    parser.add_argument(
        "--label-table",
        type=str,
        default="fire_labels",
        help="Labels table to join (identifier only: <table> or <schema>.<table>)",
    )
    
    args = parser.parse_args()
    
    engine = get_engine()
    
    print(f"Loading data from {args.start} to {args.end}...")
    try:
        df_labeled = load_labeled_data(engine, args.start, args.end, label_table=args.label_table)
    except Exception as e:
        print(f"Error loading labeled data: {e}")
        # For Demo/Smoke test, if table doesn't exist, we might want to handle it or just fail.
        # If it's a first run, the user might need to create the labels table first.
        sys.exit(1)
        
    if df_labeled.empty:
        print("No labeled detections found in the specified range.")
        sys.exit(0)
        
    print(f"Building features for {len(df_labeled)} detections...")
    X, y, meta = build_dataset(df_labeled, engine, region_name=args.aoi)
    
    if X.empty:
        print("No training samples (POSITIVE/NEGATIVE) found after filtering.")
        sys.exit(0)
        
    # Combine into a single dataframe for Parquet export
    # We want X + label + meta
    out_df = pd.concat([X, meta], axis=1)
    
    print(f"Writing {len(out_df)} rows to {args.out}...")
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
