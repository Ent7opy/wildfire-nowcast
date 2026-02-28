#!/usr/bin/env python3
"""Load authoritative perimeter coverage masks with required provenance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.coverage_mask_builder import (  # noqa: E402
    _latest_successful_run_id,
    _parse_dt,
    build_coverage_masks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load denoiser coverage masks from authoritative geometry + provenance."
    )
    parser.add_argument("--input", required=True, help="Path to authoritative GeoJSON geometry")
    parser.add_argument("--authority-profile", required=True, help="Authority profile id (e.g., wfigs_us)")
    parser.add_argument(
        "--tier-policy",
        default="silver_gold",
        choices=["gold_only", "silver_only", "silver_gold"],
    )
    parser.add_argument("--source-uri", required=True, help="Authoritative source URI")
    parser.add_argument("--source-version", required=True, help="Source dataset/API version")
    parser.add_argument(
        "--run-source-profile",
        default=None,
        help="Source profile used to resolve latest successful ingest run id.",
    )
    parser.add_argument("--run-id", default=None, help="Optional ingest run id")
    parser.add_argument("--valid-from", default=None, help="Default ISO datetime for all rows")
    parser.add_argument("--valid-to", default=None, help="Default ISO datetime for all rows")
    parser.add_argument("--inactive", action="store_true", help="Load rows with is_active=false by default")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_id = str(args.run_id).strip() if args.run_id else None
    if not run_id:
        source_profile = str(args.run_source_profile).strip() if args.run_source_profile else None
        run_id = _latest_successful_run_id(source_profile)
    if not run_id:
        raise SystemExit(
            "No successful authoritative ingest run found. "
            "Run ingest.wfigs_authority_ingest first or pass --run-id/--run-source-profile."
        )

    summary = build_coverage_masks(
        input_path=Path(args.input),
        authority_profile=str(args.authority_profile),
        tier_policy=str(args.tier_policy),
        source_uri=str(args.source_uri),
        source_version=str(args.source_version),
        run_id=run_id,
        default_valid_from=_parse_dt(args.valid_from),
        default_valid_to=_parse_dt(args.valid_to),
        default_is_active=not bool(args.inactive),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main(sys.argv[1:])
