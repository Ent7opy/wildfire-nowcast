#!/usr/bin/env python3
"""Run constrained denoiser v2 PU-bagging sweep and rank candidates."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SweepPoint:
    num_bags: int
    unlabeled_multiplier: int
    pos_threshold: float
    neg_threshold: float
    pos_class_weight: int


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=True,
    )
    out = (proc.stdout or "").strip().splitlines()
    return out[-1].strip() if out else ""


def _build_grid(args: argparse.Namespace) -> list[SweepPoint]:
    grid = itertools.product(
        _parse_int_list(args.num_bags),
        _parse_int_list(args.unlabeled_multiplier),
        _parse_float_list(args.pos_threshold),
        _parse_float_list(args.neg_threshold),
        _parse_int_list(args.pos_class_weight),
    )
    return [SweepPoint(*combo) for combo in grid]


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(bool(row.get("gate_pass", False))),
        float(row.get("covered_f1", -1.0)),
        int(bool(row.get("global_gate_pass", False))),
        float(row.get("global_f1", -1.0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained denoiser v2 sweep")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--num-bags", default="10,15,20")
    parser.add_argument("--unlabeled-multiplier", default="3,4,5")
    parser.add_argument("--pos-threshold", default="0.65,0.70,0.75")
    parser.add_argument("--neg-threshold", default="0.25,0.30,0.35")
    parser.add_argument("--pos-class-weight", default="4,8,16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else workspace / "reports" / "denoiser_v2" / f"sweep_{run_ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_root = out_root / "configs"
    cfg_root.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    points = _build_grid(args)

    if not args.execute:
        payload = {
            "mode": "dry_run",
            "count": len(points),
            "points": [point.__dict__ for point in points],
        }
        (out_root / "plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(str(out_root / "plan.json"))
        return

    leaderboard: list[dict[str, Any]] = []

    for idx, point in enumerate(points, start=1):
        cfg = dict(base_cfg)
        cfg["snapshot_path"] = args.snapshot
        cfg["model_backend"] = "xgboost_pu_bagging"
        cfg["coverage_scope"] = "covered"
        cfg["coverage_mask_source"] = "db_mask"
        cfg["pos_class_weight"] = point.pos_class_weight
        pu = dict(cfg.get("pu_bagging", {}))
        pu.update(
            {
                "num_bags": point.num_bags,
                "unlabeled_multiplier": point.unlabeled_multiplier,
                "pos_threshold": point.pos_threshold,
                "neg_threshold": point.neg_threshold,
            }
        )
        cfg["pu_bagging"] = pu

        cfg_path = cfg_root / f"{idx:03d}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        run_dir = _run(
            ["uv", "run", "--project", "ml", "-m", "ml.train_denoiser_v2", "--config", str(cfg_path)],
            cwd=workspace,
        )
        report_dir = out_root / "reports" / Path(run_dir).name
        report_dir.parent.mkdir(parents=True, exist_ok=True)

        _run(
            [
                "uv",
                "run",
                "--project",
                "ml",
                "-m",
                "ml.eval_denoiser_v2",
                "--model_run",
                run_dir,
                "--snapshot",
                args.snapshot,
                "--out",
                str(report_dir),
                "--gate-scope",
                "both",
            ],
            cwd=workspace,
        )

        summary = json.loads((report_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        row = {
            "config_path": str(cfg_path),
            "run_dir": run_dir,
            "report_dir": str(report_dir),
            "gate_pass": bool(summary.get("gate_pass", False)),
            "covered_f1": float((summary.get("covered") or {}).get("default_metrics", {}).get("f1", -1.0)),
            "global_f1": float((summary.get("global") or {}).get("default_metrics", {}).get("f1", -1.0)),
            "global_gate_pass": bool((summary.get("global") or {}).get("gate_pass", False)),
            "num_bags": point.num_bags,
            "unlabeled_multiplier": point.unlabeled_multiplier,
            "pos_threshold": point.pos_threshold,
            "neg_threshold": point.neg_threshold,
            "pos_class_weight": point.pos_class_weight,
        }
        leaderboard.append(row)

    leaderboard.sort(key=_rank_key, reverse=True)
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank

    (out_root / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    with (out_root / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(leaderboard[0].keys()) if leaderboard else ["rank"])
        writer.writeheader()
        for row in leaderboard:
            writer.writerow(row)

    print(str(out_root / "leaderboard.json"))


if __name__ == "__main__":
    main()
