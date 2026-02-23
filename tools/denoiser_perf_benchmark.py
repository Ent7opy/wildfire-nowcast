"""Denoiser v2 performance benchmark helper.

Collects:
- step latency
- rows processed
- rows/sec
- explain plan snippets for hot SQL paths
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class StepMetric:
    name: str
    latency_seconds: float
    rows_processed: int | None
    rows_per_second: float | None
    status: str
    details: dict[str, Any]


def _run_explain(sql: str, params: dict[str, Any]) -> list[str]:
    from api.db import get_engine

    with get_engine().begin() as conn:
        rows = conn.execute(text(f"EXPLAIN (FORMAT TEXT) {sql}"), params).fetchall()
    return [str(row[0]) for row in rows]


def _scalar(sql: str, params: dict[str, Any]) -> int:
    from api.db import get_engine

    with get_engine().begin() as conn:
        value = conn.execute(text(sql), params).scalar_one()
    return int(value or 0)


def _run_command(name: str, cmd: str) -> StepMetric:
    started = time.perf_counter()
    completed = subprocess.run(
        shlex.split(cmd),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started

    rows_processed: int | None = None
    stdout = completed.stdout.strip()
    if stdout:
        last_line = stdout.splitlines()[-1]
        if last_line.startswith("{") and last_line.endswith("}"):
            try:
                payload = json.loads(last_line)
                for key in ("rows", "count", "event_total"):
                    if key in payload:
                        rows_processed = int(payload[key])
                        break
            except Exception:
                rows_processed = None

    rows_per_second = None
    if rows_processed is not None and elapsed > 0:
        rows_per_second = float(rows_processed / elapsed)

    status = "ok" if completed.returncode == 0 else "failed"
    return StepMetric(
        name=name,
        latency_seconds=elapsed,
        rows_processed=rows_processed,
        rows_per_second=rows_per_second,
        status=status,
        details={
            "return_code": completed.returncode,
            "stdout_tail": "\n".join(stdout.splitlines()[-20:]),
            "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-20:]),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark denoiser query performance")
    parser.add_argument("--start", default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--end", default="2025-03-31T23:59:59+00:00")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=[-125.0, 24.0, -66.0, 49.0],
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    )
    parser.add_argument("--eventize-cmd", default="")
    parser.add_argument("--label-cmd", default="")
    parser.add_argument("--snapshot-cmd", default="")
    parser.add_argument("--train-cmd", default="")
    parser.add_argument("--eval-cmd", default="")
    parser.add_argument("--out", default="data/denoiser/perf_benchmark_q1_2025.json")
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    min_lon, min_lat, max_lon, max_lat = [float(x) for x in args.bbox]

    params = {
        "start_time": start_dt,
        "end_time": end_dt,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }

    report: dict[str, Any] = {
        "window": {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "bbox": [min_lon, min_lat, max_lon, max_lat],
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset": {},
        "plans": {},
        "steps": [],
    }

    report["dataset"]["detections_in_window"] = _scalar(
        """
        SELECT COUNT(*)
        FROM fire_detections
        WHERE acq_time BETWEEN :start_time AND :end_time
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
        """,
        params,
    )
    report["dataset"]["perimeters_in_window"] = _scalar(
        """
        SELECT COUNT(*)
        FROM fire_perimeters
        WHERE fire_start IS NOT NULL
          AND fire_start <= :end_time
          AND (fire_end IS NULL OR fire_end >= :start_time)
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
        """,
        params,
    )

    report["plans"]["eventize_selection"] = _run_explain(
        """
        SELECT id, source, sensor, acq_time, lat, lon
        FROM fire_detections
        WHERE acq_time BETWEEN :start_time AND :end_time
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
        """,
        params,
    )

    plan_params = dict(params)
    plan_params.update(
        {
            "buffer_m": 2315.0,
            "buffer_deg": 2315.0 / 111000.0,
            "pad_h": 48,
            "confidence_floor": 30.0,
        }
    )

    report["plans"]["label_positive"] = _run_explain(
        """
        SELECT DISTINCT d.id
        FROM fire_detections d
        JOIN fire_perimeters fp
          ON fp.geom && ST_Expand(d.geom, :buffer_deg)
         AND ST_DWithin(d.geom::geography, fp.geom::geography, :buffer_m)
        WHERE d.acq_time BETWEEN :start_time AND :end_time
          AND d.acq_time >= fp.fire_start - make_interval(hours => :pad_h)
          AND (fp.fire_end IS NULL OR d.acq_time <= fp.fire_end + make_interval(hours => :pad_h))
          AND COALESCE(d.confidence, 0) >= :confidence_floor
          AND d.geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
        """,
        plan_params,
    )

    if args.eventize_cmd:
        report["steps"].append(asdict(_run_command("eventize", args.eventize_cmd)))
    if args.label_cmd:
        report["steps"].append(asdict(_run_command("label_v2", args.label_cmd)))
    if args.snapshot_cmd:
        report["steps"].append(asdict(_run_command("snapshot_v2", args.snapshot_cmd)))
    if args.train_cmd:
        report["steps"].append(asdict(_run_command("train_v2", args.train_cmd)))
    if args.eval_cmd:
        report["steps"].append(asdict(_run_command("eval_v2", args.eval_cmd)))

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(str(output_path))


if __name__ == "__main__":
    main()
