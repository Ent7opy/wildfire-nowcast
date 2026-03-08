#!/usr/bin/env python3
"""Freeze denoiser v2 baseline artifacts for reproducible comparisons."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        shutil.copy2(src, dst)


def freeze_baseline(model_run_dir: Path, snapshot_path: Path | None, out_root: Path) -> Path:
    run_id = model_run_dir.name
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{ts}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=False)

    _copy_if_exists(model_run_dir / "metrics.json", out_dir / "metrics.json")
    _copy_if_exists(model_run_dir / "gate_report.json", out_dir / "gate_report.json")
    _copy_if_exists(model_run_dir / "config_resolved.yaml", out_dir / "config_resolved.yaml")
    _copy_if_exists(model_run_dir / "metadata.json", out_dir / "metadata.json")
    _copy_if_exists(model_run_dir / "feature_list.json", out_dir / "feature_list.json")

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model_run_dir": str(model_run_dir),
        "run_id": run_id,
        "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze denoiser v2 baseline artifacts")
    parser.add_argument("--model-run", required=True, help="models/denoiser_v2/<run_id>")
    parser.add_argument("--snapshot", default=None, help="Optional snapshot parquet path")
    parser.add_argument("--out-root", default="reports/denoiser_v2/baselines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = freeze_baseline(
        model_run_dir=Path(args.model_run),
        snapshot_path=Path(args.snapshot) if args.snapshot else None,
        out_root=Path(args.out_root),
    )
    print(str(out_dir))


if __name__ == "__main__":
    main()
