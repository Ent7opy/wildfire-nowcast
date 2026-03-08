"""Baseline-vs-updated event association quality report for denoiser v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from typing import Iterable

import pandas as pd
from sqlalchemy import text

from api.db import get_engine
from ml.denoiser.eventize import EventizeParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("eval_event_association")

_BASELINE_FRONT_TIME_BUCKET_MINUTES = 30
_BASELINE_FRONT_CELL_DEG = 0.05
_BASELINE_EVENT_CELL_DEG = 0.2
_BASELINE_EVENT_LINK_DAYS = 3


@dataclass
class UnionFind:
    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, n: int) -> "UnionFind":
        return cls(parent=list(range(n)), rank=[0] * n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _stable_md5(parts: Iterable[object]) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()  # noqa: S324 - deterministic ID only


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.strip()
    if len(cleaned) == 10:
        return datetime.strptime(cleaned, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = radians(lat1)
    p2 = radians(lat2)
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2.0) ** 2 + cos(p1) * cos(p2) * sin(d_lon / 2.0) ** 2
    return 2.0 * r * asin(sqrt(a))


def _normalize_detection_rows(df: pd.DataFrame, *, params: EventizeParams) -> pd.DataFrame:
    out = df.copy()
    out["acq_time"] = pd.to_datetime(out["acq_time"], utc=True)
    out["source"] = out["source"].fillna("")
    out["sensor"] = out["sensor"].fillna("")
    out["false_source_masked"] = out["false_source_masked"].fillna(False).astype(bool)
    out["persistence_score"] = out["persistence_score"].fillna(0.0).astype(float)
    out["is_static_like"] = (
        out["false_source_masked"] | (out["persistence_score"] >= float(params.static_persistence_threshold))
    )
    return out.sort_values(["source", "sensor", "acq_time", "id"]).reset_index(drop=True)


def compute_baseline_assignments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    epoch = out["acq_time"].astype("int64") // 1_000_000_000

    front_bucket = (epoch // (_BASELINE_FRONT_TIME_BUCKET_MINUTES * 60)).astype(int)
    front_i_lat = (out["lat"] / _BASELINE_FRONT_CELL_DEG).apply(lambda v: int(v // 1))
    front_j_lon = (out["lon"] / _BASELINE_FRONT_CELL_DEG).apply(lambda v: int(v // 1))

    event_bucket = (epoch // (86400 * _BASELINE_EVENT_LINK_DAYS)).astype(int)
    event_i_lat = (out["lat"] / _BASELINE_EVENT_CELL_DEG).apply(lambda v: int(v // 1))
    event_j_lon = (out["lon"] / _BASELINE_EVENT_CELL_DEG).apply(lambda v: int(v // 1))

    out["front_id"] = [
        _stable_md5((src, sensor, tb, il, jl))
        for src, sensor, tb, il, jl in zip(
            out["source"],
            out["sensor"],
            front_bucket,
            front_i_lat,
            front_j_lon,
            strict=False,
        )
    ]
    out["event_id"] = [
        _stable_md5((src, sensor, eb, il, jl))
        for src, sensor, eb, il, jl in zip(
            out["source"],
            out["sensor"],
            event_bucket,
            event_i_lat,
            event_j_lon,
            strict=False,
        )
    ]

    return out[["id", "front_id", "event_id", "is_static_like"]].copy()


def _cluster_fronts(df: pd.DataFrame, *, params: EventizeParams) -> pd.DataFrame:
    front_frames: list[pd.DataFrame] = []
    gap_seconds = int(params.front_max_gap_minutes) * 60

    group_cols = ["source", "sensor"]
    if params.strict_static_split:
        group_cols.append("is_static_like")

    for _, grp in df.groupby(group_cols, dropna=False):
        g = grp.sort_values(["acq_time", "id"]).reset_index(drop=True)
        if g.empty:
            continue

        times = (g["acq_time"].astype("int64") // 1_000_000_000).to_numpy()
        lats = g["lat"].astype(float).to_numpy()
        lons = g["lon"].astype(float).to_numpy()
        ids = g["id"].astype(int).to_numpy()

        uf = UnionFind.create(len(g))
        for i in range(len(g)):
            j = i + 1
            while j < len(g) and (times[j] - times[i]) <= gap_seconds:
                if _haversine_m(lats[i], lons[i], lats[j], lons[j]) <= float(params.front_link_radius_m):
                    uf.union(i, j)
                j += 1

        root_to_anchor: dict[int, int] = {}
        for i, det_id in enumerate(ids):
            root = uf.find(i)
            anchor = root_to_anchor.get(root)
            if anchor is None or int(det_id) < anchor:
                root_to_anchor[root] = int(det_id)

        g["front_component_anchor"] = [root_to_anchor[uf.find(i)] for i in range(len(g))]

        start_by_anchor = g.groupby("front_component_anchor")["acq_time"].min().to_dict()
        g["front_id"] = [
            _stable_md5(
                (
                    "front_v2",
                    src,
                    sensor,
                    anchor,
                    start_by_anchor[int(anchor)].isoformat(),
                )
            )
            for src, sensor, anchor in zip(g["source"], g["sensor"], g["front_component_anchor"], strict=False)
        ]
        front_frames.append(g)

    if not front_frames:
        return pd.DataFrame(columns=["id", "front_id", "front_component_anchor", "is_static_like"])

    merged = pd.concat(front_frames, ignore_index=True)
    return merged[["id", "front_id", "front_component_anchor", "is_static_like"]].copy()


def _cluster_events(df: pd.DataFrame, fronts: pd.DataFrame, *, params: EventizeParams) -> pd.DataFrame:
    merged = df.merge(fronts[["id", "front_id", "front_component_anchor"]], on="id", how="inner")
    front_summary = (
        merged.groupby("front_id", as_index=False)
        .agg(
            source=("source", "first"),
            sensor=("sensor", "first"),
            front_anchor=("front_component_anchor", "min"),
            overpass_start=("acq_time", "min"),
            overpass_end=("acq_time", "max"),
            lat_centroid=("lat", "mean"),
            lon_centroid=("lon", "mean"),
            static_ratio=("is_static_like", "mean"),
        )
    )
    front_summary["front_static_like"] = front_summary["static_ratio"] >= 0.5

    group_cols = ["source", "sensor"]
    if params.strict_static_split:
        group_cols.append("front_static_like")

    event_frames: list[pd.DataFrame] = []
    gap = timedelta(days=int(params.event_max_gap_days))

    for _, grp in front_summary.groupby(group_cols, dropna=False):
        g = grp.sort_values(["overpass_start", "front_anchor", "front_id"]).reset_index(drop=True)
        if g.empty:
            continue

        uf = UnionFind.create(len(g))
        starts = g["overpass_start"].tolist()
        ends = g["overpass_end"].tolist()
        lats = g["lat_centroid"].astype(float).to_numpy()
        lons = g["lon_centroid"].astype(float).to_numpy()

        for i in range(len(g)):
            j = i + 1
            while j < len(g) and starts[j] <= (ends[i] + gap):
                if _haversine_m(lats[i], lons[i], lats[j], lons[j]) <= float(params.event_link_radius_m):
                    uf.union(i, j)
                j += 1

        root_to_anchor: dict[int, int] = {}
        for i, anchor in enumerate(g["front_anchor"].astype(int).tolist()):
            root = uf.find(i)
            prev = root_to_anchor.get(root)
            if prev is None or anchor < prev:
                root_to_anchor[root] = anchor

        g["event_component_anchor"] = [root_to_anchor[uf.find(i)] for i in range(len(g))]
        start_by_anchor = g.groupby("event_component_anchor")["overpass_start"].min().to_dict()
        g["event_id"] = [
            _stable_md5(
                (
                    "event_v2",
                    src,
                    sensor,
                    anchor,
                    start_by_anchor[int(anchor)].isoformat(),
                )
            )
            for src, sensor, anchor in zip(g["source"], g["sensor"], g["event_component_anchor"], strict=False)
        ]
        event_frames.append(g[["front_id", "event_id"]])

    if not event_frames:
        return fronts.assign(event_id="")[["id", "front_id", "event_id", "is_static_like"]]

    front_events = pd.concat(event_frames, ignore_index=True)
    out = fronts.merge(front_events, on="front_id", how="left")
    return out[["id", "front_id", "event_id", "is_static_like"]].copy()


def compute_updated_assignments(df: pd.DataFrame, *, params: EventizeParams) -> pd.DataFrame:
    normalized = _normalize_detection_rows(df, params=params)
    fronts = _cluster_fronts(normalized, params=params)
    if fronts.empty:
        return pd.DataFrame(columns=["id", "front_id", "event_id", "is_static_like"])
    assignments = _cluster_events(normalized, fronts, params=params)
    assignments["event_id"] = assignments["event_id"].fillna("")
    return assignments.sort_values("id").reset_index(drop=True)


def compute_event_metrics(detections: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, float | int]:
    merged = detections[["id", "acq_time", "is_static_like"]].merge(assignments, on="id", how="inner")
    if merged.empty:
        return {
            "event_count": 0,
            "detection_count": 0,
            "median_event_duration_hours": 0.0,
            "multi_day_event_share": 0.0,
            "singleton_event_share": 0.0,
            "mixed_static_dynamic_event_share": 0.0,
        }

    event_summary = (
        merged.groupby("event_id", dropna=False)
        .agg(
            start_time=("acq_time", "min"),
            end_time=("acq_time", "max"),
            detection_count=("id", "count"),
            static_count=("is_static_like", "sum"),
        )
        .reset_index()
    )
    event_summary["duration_hours"] = (
        (event_summary["end_time"] - event_summary["start_time"]).dt.total_seconds() / 3600.0
    )
    event_summary["dynamic_count"] = event_summary["detection_count"] - event_summary["static_count"]

    event_count = int(len(event_summary))
    if event_count == 0:
        return {
            "event_count": 0,
            "detection_count": int(len(merged)),
            "median_event_duration_hours": 0.0,
            "multi_day_event_share": 0.0,
            "singleton_event_share": 0.0,
            "mixed_static_dynamic_event_share": 0.0,
        }

    return {
        "event_count": event_count,
        "detection_count": int(len(merged)),
        "median_event_duration_hours": float(event_summary["duration_hours"].median()),
        "multi_day_event_share": float((event_summary["duration_hours"] >= 24.0).mean()),
        "singleton_event_share": float((event_summary["detection_count"] == 1).mean()),
        "mixed_static_dynamic_event_share": float(
            ((event_summary["static_count"] > 0) & (event_summary["dynamic_count"] > 0)).mean()
        ),
    }


def compute_replay_diff_rate(assign_a: pd.DataFrame, assign_b: pd.DataFrame) -> float:
    merged = assign_a[["id", "event_id"]].merge(
        assign_b[["id", "event_id"]],
        on="id",
        suffixes=("_a", "_b"),
        how="inner",
    )
    if merged.empty:
        return 0.0
    return float((merged["event_id_a"] != merged["event_id_b"]).mean())


def evaluate_no_regression_gates(baseline: dict[str, float | int], updated: dict[str, float | int]) -> dict[str, bool]:
    gates = {
        "median_event_duration_hours_non_decreasing": float(updated["median_event_duration_hours"])
        >= float(baseline["median_event_duration_hours"]),
        "singleton_event_share_non_increasing": float(updated["singleton_event_share"])
        <= float(baseline["singleton_event_share"]),
        "mixed_static_dynamic_event_share_non_increasing": float(updated["mixed_static_dynamic_event_share"])
        <= float(baseline["mixed_static_dynamic_event_share"]),
    }
    gates["pass"] = all(gates.values())
    return gates


def _load_detections(
    *,
    start_time: datetime,
    end_time: datetime,
    source_like: str | None,
) -> pd.DataFrame:
    source_predicate = ""
    params: dict[str, object] = {
        "start_time": start_time,
        "end_time": end_time,
    }
    if source_like:
        source_predicate = "AND source LIKE :source_like"
        params["source_like"] = source_like

    stmt = text(
        f"""
        SELECT
            id,
            source,
            sensor,
            acq_time,
            lat,
            lon,
            COALESCE(false_source_masked, FALSE) AS false_source_masked,
            COALESCE(persistence_score, 0.0) AS persistence_score
        FROM fire_detections
        WHERE acq_time >= :start_time
          AND acq_time <= :end_time
          {source_predicate}
        ORDER BY id
        """
    )

    with get_engine().begin() as conn:
        return pd.read_sql(stmt, conn, params=params)


def _render_summary_md(
    *,
    start_time: datetime,
    end_time: datetime,
    source_like: str | None,
    baseline_metrics: dict[str, float | int],
    updated_metrics: dict[str, float | int],
    replay_diff_rate: float,
    gates: dict[str, bool],
) -> str:
    return "\n".join(
        [
            "# Event Association Comparison",
            "",
            f"- window_start: {start_time.isoformat()}",
            f"- window_end: {end_time.isoformat()}",
            f"- source_like: {source_like or 'ALL'}",
            "",
            "## Baseline Metrics",
            "",
            f"- median_event_duration_hours: {baseline_metrics['median_event_duration_hours']:.3f}",
            f"- multi_day_event_share: {baseline_metrics['multi_day_event_share']:.6f}",
            f"- singleton_event_share: {baseline_metrics['singleton_event_share']:.6f}",
            f"- mixed_static_dynamic_event_share: {baseline_metrics['mixed_static_dynamic_event_share']:.6f}",
            "",
            "## Updated Metrics",
            "",
            f"- median_event_duration_hours: {updated_metrics['median_event_duration_hours']:.3f}",
            f"- multi_day_event_share: {updated_metrics['multi_day_event_share']:.6f}",
            f"- singleton_event_share: {updated_metrics['singleton_event_share']:.6f}",
            f"- mixed_static_dynamic_event_share: {updated_metrics['mixed_static_dynamic_event_share']:.6f}",
            f"- deterministic_replay_diff_rate: {replay_diff_rate:.6f}",
            "",
            "## Gates",
            "",
            f"- pass: {bool(gates['pass'])}",
            f"- median_event_duration_hours_non_decreasing: {bool(gates['median_event_duration_hours_non_decreasing'])}",
            f"- singleton_event_share_non_increasing: {bool(gates['singleton_event_share_non_increasing'])}",
            "- mixed_static_dynamic_event_share_non_increasing: "
            f"{bool(gates['mixed_static_dynamic_event_share_non_increasing'])}",
        ]
    )


def evaluate_event_association(
    *,
    start_time: datetime,
    end_time: datetime,
    source_like: str | None,
    out_root: str,
    params: EventizeParams,
) -> str:
    detections = _load_detections(start_time=start_time, end_time=end_time, source_like=source_like)
    if detections.empty:
        raise SystemExit("No detections found for requested window.")

    detections = _normalize_detection_rows(detections, params=params)

    baseline_assignments = compute_baseline_assignments(detections)
    updated_assignments_a = compute_updated_assignments(detections, params=params)
    updated_assignments_b = compute_updated_assignments(detections, params=params)

    replay_diff_rate = compute_replay_diff_rate(updated_assignments_a, updated_assignments_b)

    baseline_metrics = compute_event_metrics(detections, baseline_assignments)
    updated_metrics = compute_event_metrics(detections, updated_assignments_a)
    updated_metrics["deterministic_replay_diff_rate"] = replay_diff_rate

    gates = evaluate_no_regression_gates(baseline_metrics, updated_metrics)
    gates["deterministic_replay_diff_rate_zero"] = replay_diff_rate == 0.0
    gates["pass"] = bool(gates["pass"] and gates["deterministic_replay_diff_rate_zero"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    metric_rows = []
    for metric in (
        "median_event_duration_hours",
        "multi_day_event_share",
        "singleton_event_share",
        "mixed_static_dynamic_event_share",
    ):
        baseline_value = float(baseline_metrics[metric])
        updated_value = float(updated_metrics[metric])
        metric_rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "updated": updated_value,
                "delta": updated_value - baseline_value,
            }
        )
    metric_rows.append(
        {
            "metric": "deterministic_replay_diff_rate",
            "baseline": 0.0,
            "updated": float(replay_diff_rate),
            "delta": float(replay_diff_rate),
        }
    )

    comparison_df = pd.DataFrame(metric_rows)
    comparison_df.to_csv(os.path.join(run_dir, "comparison_table.csv"), index=False)

    assignment_diff = baseline_assignments[["id", "event_id"]].merge(
        updated_assignments_a[["id", "event_id"]],
        on="id",
        how="inner",
        suffixes=("_baseline", "_updated"),
    )
    assignment_diff = assignment_diff[assignment_diff["event_id_baseline"] != assignment_diff["event_id_updated"]]
    assignment_diff.head(500).to_csv(os.path.join(run_dir, "sample_assignment_diff.csv"), index=False)

    summary_payload = {
        "window": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "source_like": source_like,
        },
        "params": {
            "front_link_radius_m": float(params.front_link_radius_m),
            "front_max_gap_minutes": int(params.front_max_gap_minutes),
            "event_link_radius_m": float(params.event_link_radius_m),
            "event_max_gap_days": int(params.event_max_gap_days),
            "static_persistence_threshold": float(params.static_persistence_threshold),
            "strict_static_split": bool(params.strict_static_split),
        },
        "counts": {
            "detections": int(len(detections)),
            "baseline_events": int(baseline_metrics["event_count"]),
            "updated_events": int(updated_metrics["event_count"]),
            "assignment_diffs": int(len(assignment_diff)),
        },
        "baseline_metrics": baseline_metrics,
        "updated_metrics": updated_metrics,
        "gates": gates,
    }

    with open(os.path.join(run_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    summary_md = _render_summary_md(
        start_time=start_time,
        end_time=end_time,
        source_like=source_like,
        baseline_metrics=baseline_metrics,
        updated_metrics=updated_metrics,
        replay_diff_rate=replay_diff_rate,
        gates=gates,
    )
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs updated event association quality.")
    parser.add_argument("--start", type=str, default=None, help="ISO datetime or YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="ISO datetime or YYYY-MM-DD")
    parser.add_argument("--source-like", type=str, default=None, help="Optional source LIKE filter")
    parser.add_argument("--out", type=str, default="reports/denoiser_v2/event_association")
    args = parser.parse_args()

    end_time = _parse_dt(args.end) or datetime.now(timezone.utc)
    start_time = _parse_dt(args.start) or (end_time - timedelta(days=30))

    if start_time > end_time:
        raise SystemExit("--start must be <= --end")

    run_dir = evaluate_event_association(
        start_time=start_time,
        end_time=end_time,
        source_like=args.source_like,
        out_root=args.out,
        params=EventizeParams(),
    )
    LOGGER.info("Wrote event association report: %s", run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
