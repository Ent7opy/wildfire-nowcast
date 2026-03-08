"""False-positive autopsy for denoiser MVP precision diagnostics.

This script is read-only: it loads an existing model bundle + eval snapshot,
reconstructs calibrated probabilities, isolates false positives, and reports:
1) landcover_class distribution
2) thermal-intensity bucket distribution
3) spatial contiguity (DBSCAN cluster-size categories)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.parquet_io import read_parquet_with_fallback

_EARTH_RADIUS_M = 6_371_000.0
_DEFAULT_DBSCAN_EPS_M = 2315.0
_FP_BASE_LABELS = {"NEGATIVE", "UNKNOWN"}
_POS_LABEL = "POSITIVE"
_FRP_CANDIDATES = ("frp_density", "frp", "frp_max", "frp_mean")
_LAT_CANDIDATES = ("lat_centroid", "lat")
_LON_CANDIDATES = ("lon_centroid", "lon")


@dataclass(frozen=True)
class RunContext:
    model_run_dir: str
    snapshot_path: str
    run_id: str
    created_at: datetime
    lat_col: str
    lon_col: str
    frp_col: str


def _load_snapshot(path: str, *, columns: list[str]) -> pd.DataFrame:
    read_columns = list(dict.fromkeys(columns))

    def _read_with_fallback(parquet_path: str) -> pd.DataFrame:
        try:
            return read_parquet_with_fallback(parquet_path, columns=read_columns)
        except Exception:
            full = read_parquet_with_fallback(parquet_path)
            keep = [c for c in read_columns if c in full.columns]
            return full[keep]

    if os.path.isdir(path):
        return _read_with_fallback(os.path.join(path, "eval.parquet"))
    return _read_with_fallback(path)


def _predict_raw(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    pred = np.asarray(model.predict(x), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def _apply_calibrator(cal: dict[str, Any] | None, scores: np.ndarray) -> np.ndarray:
    if cal is None:
        return np.asarray(scores, dtype=float)
    ctype = cal.get("type")
    model = cal.get("model")
    if ctype == "isotonic" and model is not None:
        return np.asarray(model.predict(scores), dtype=float)
    if ctype == "platt" and model is not None:
        return np.asarray(model.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)
    return np.asarray(scores, dtype=float)


def _slice_key(row: pd.Series, slice_cols: list[str]) -> str:
    return "|".join(f"{col}={row.get(col, 'unknown')}" for col in slice_cols)


def _feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError(
            "STOP: We are missing an authoritative source for required model features in the eval snapshot "
            f"({', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}). "
            "I cannot proceed without faking it, which we have agreed not to do. "
            "Should we export the matching snapshot for this model run?"
        )
    return df[features].astype(np.float32)


def _parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _candidate_parquet_columns(path: str) -> set[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        frame = read_parquet_with_fallback(path)
        return set(frame.columns)


def _pick_first(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    col_set = set(columns)
    for candidate in candidates:
        if candidate in col_set:
            return candidate
    return None


def _discover_latest_context(model_root: str) -> RunContext:
    if not os.path.isdir(model_root):
        raise RuntimeError(f"Model root not found: {model_root}")

    candidates: list[RunContext] = []
    for entry in sorted(os.listdir(model_root)):
        run_dir = os.path.join(model_root, entry)
        metadata_path = os.path.join(run_dir, "metadata.json")
        bundle_path = os.path.join(run_dir, "model_bundle.pkl")
        if not os.path.isdir(run_dir) or not os.path.isfile(metadata_path) or not os.path.isfile(bundle_path):
            continue
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            snapshot_path = str(metadata.get("config", {}).get("snapshot_path", "")).strip()
            run_id = str(metadata.get("run_id", entry))
            created_at_raw = str(metadata.get("created_at", "")).strip()
            created_at = _parse_iso8601(created_at_raw)
            if not snapshot_path:
                continue
            parquet_path = (
                os.path.join(snapshot_path, "eval.parquet")
                if os.path.isdir(snapshot_path)
                else snapshot_path
            )
            if not os.path.isfile(parquet_path):
                continue
            cols = _candidate_parquet_columns(parquet_path)
            if "event_label" not in cols or "landcover_class" not in cols:
                continue
            lat_col = _pick_first(cols, _LAT_CANDIDATES)
            lon_col = _pick_first(cols, _LON_CANDIDATES)
            frp_col = _pick_first(cols, _FRP_CANDIDATES)
            if lat_col is None or lon_col is None or frp_col is None:
                continue
            candidates.append(
                RunContext(
                    model_run_dir=run_dir,
                    snapshot_path=snapshot_path,
                    run_id=run_id,
                    created_at=created_at,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    frp_col=frp_col,
                )
            )
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            "STOP: We are missing an authoritative source for one or more required FP autopsy fields "
            "(event_label, landcover_class, FRP, latitude/longitude) in available recent model snapshots. "
            "I cannot proceed without faking it, which we have agreed not to do. "
            "Should we export an eval snapshot that includes these columns?"
        )
    return sorted(candidates, key=lambda c: c.created_at)[-1]


def _resolve_context(args: argparse.Namespace) -> RunContext:
    if args.model_run is None and args.snapshot is None:
        return _discover_latest_context(model_root=args.model_root)
    if args.model_run is None or args.snapshot is None:
        raise RuntimeError("Provide both --model-run and --snapshot, or neither for auto-discovery.")

    metadata_path = os.path.join(args.model_run, "metadata.json")
    run_id = os.path.basename(args.model_run.rstrip(os.sep))
    created_at = datetime.now().astimezone()
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            run_id = str(metadata.get("run_id", run_id))
            created_at = _parse_iso8601(str(metadata.get("created_at")))
        except Exception:
            pass

    parquet_path = (
        os.path.join(args.snapshot, "eval.parquet")
        if os.path.isdir(args.snapshot)
        else args.snapshot
    )
    cols = _candidate_parquet_columns(parquet_path)
    lat_col = _pick_first(cols, _LAT_CANDIDATES)
    lon_col = _pick_first(cols, _LON_CANDIDATES)
    frp_col = _pick_first(cols, _FRP_CANDIDATES)
    missing_fields = [c for c in ("event_label", "landcover_class") if c not in cols]
    if lat_col is None:
        missing_fields.append("lat/lat_centroid")
    if lon_col is None:
        missing_fields.append("lon/lon_centroid")
    if frp_col is None:
        missing_fields.append("frp_density|frp|frp_max|frp_mean")
    if missing_fields:
        raise RuntimeError(
            "STOP: We are missing an authoritative source for "
            f"{', '.join(missing_fields)} in the selected snapshot ({parquet_path}). "
            "I cannot proceed without faking it, which we have agreed not to do. "
            "Should we switch to a snapshot that carries these fields?"
        )

    return RunContext(
        model_run_dir=args.model_run,
        snapshot_path=args.snapshot,
        run_id=run_id,
        created_at=created_at,
        lat_col=lat_col,
        lon_col=lon_col,
        frp_col=frp_col,
    )


def _format_pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * count / total):.2f}%"


def _format_table(counts: pd.Series, total: int) -> list[str]:
    lines: list[str] = []
    for idx, value in counts.items():
        label = str(idx)
        count = int(value)
        lines.append(f"  - {label}: {count} ({_format_pct(count, total)})")
    return lines


def _contiguity_categories(cluster_sizes: pd.Series) -> dict[int, str]:
    out: dict[int, str] = {}
    for cluster_id, size in cluster_sizes.items():
        n = int(size)
        if n <= 1:
            bucket = "singleton_1px"
        elif n <= 4:
            bucket = "small_2_to_4px"
        elif n <= 19:
            bucket = "medium_5_to_19px"
        else:
            bucket = "large_20px_plus"
        out[int(cluster_id)] = bucket
    return out


def analyze_false_positives(args: argparse.Namespace) -> int:
    context = _resolve_context(args)
    bundle_path = os.path.join(context.model_run_dir, "model_bundle.pkl")
    if not os.path.isfile(bundle_path):
        raise RuntimeError(f"model_bundle.pkl not found in model run dir: {context.model_run_dir}")

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    features = list(bundle["features"])
    slice_cols = list(bundle.get("slice_cols", ["sensor_id", "biome_slice"]))
    global_calibrator = bundle.get("global_calibrator")
    slice_calibrators = dict(bundle.get("slice_calibrators", {}))

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(bundle.get("thresholds", {}).get("decision", 0.5))
    )

    needed_columns = list(
        dict.fromkeys(
            features
            + slice_cols
            + [
                "event_id",
                "event_label",
                "landcover_class",
                context.frp_col,
                context.lat_col,
                context.lon_col,
            ]
        )
    )
    eval_df = _load_snapshot(context.snapshot_path, columns=needed_columns).copy()
    if eval_df.empty:
        raise RuntimeError("Snapshot eval dataset is empty; no rows to analyze.")

    raw = _predict_raw(model, _feature_matrix(eval_df, features))
    calibrated = np.zeros(len(eval_df), dtype=float)
    for idx, row in enumerate(eval_df.itertuples(index=False)):
        row_series = pd.Series(row._asdict())
        key = _slice_key(row_series, slice_cols)
        calibrator = slice_calibrators.get(key, global_calibrator)
        calibrated[idx] = float(_apply_calibrator(calibrator, np.asarray([raw[idx]], dtype=float))[0])

    labels = eval_df["event_label"].astype(str).str.upper()
    pred_positive = calibrated >= threshold
    fp_mask = pred_positive & labels.isin(_FP_BASE_LABELS)
    false_positives = eval_df.loc[fp_mask].copy()
    total_fp = int(len(false_positives))

    total_pred_pos = int(pred_positive.sum())
    total_rows = int(len(eval_df))
    total_pos_gt = int((labels == _POS_LABEL).sum())
    total_neg_unk_gt = int(labels.isin(_FP_BASE_LABELS).sum())

    print("=== False Positive Autopsy ===")
    print(f"Model run: {context.run_id}")
    print(f"Model run dir: {context.model_run_dir}")
    print(f"Snapshot: {context.snapshot_path}")
    print(f"Decision threshold: {threshold:.6f}")
    print(f"FP definition: predicted_positive AND event_label in {sorted(_FP_BASE_LABELS)}")
    print("")
    print("=== Base Counts ===")
    print(f"  - eval_rows: {total_rows}")
    print(f"  - ground_truth_positive_rows: {total_pos_gt}")
    print(f"  - ground_truth_negative_or_unknown_rows: {total_neg_unk_gt}")
    print(f"  - predicted_positive_rows: {total_pred_pos}")
    print(f"  - false_positive_rows: {total_fp} ({_format_pct(total_fp, total_pred_pos)}) of predicted positives")

    if total_fp == 0:
        print("")
        print("No false positives at this threshold; stratification is empty.")
        return 0

    false_positives["landcover_class_norm"] = pd.to_numeric(
        false_positives["landcover_class"], errors="coerce"
    ).round().astype("Int64")
    landcover_labels = false_positives["landcover_class_norm"].astype("string").fillna("UNKNOWN")
    landcover_counts = landcover_labels.value_counts(dropna=False).sort_values(ascending=False)

    class_40_count = int((false_positives["landcover_class_norm"] == 40).sum())
    print("")
    print("=== FP by Land Cover (landcover_class) ===")
    print(f"  - class_40_cropland: {class_40_count} ({_format_pct(class_40_count, total_fp)}) of all FP")
    for line in _format_table(landcover_counts, total_fp):
        print(line)

    false_positives["frp_metric"] = pd.to_numeric(false_positives[context.frp_col], errors="coerce").fillna(0.0)
    frp_bins = [-1e-12, 1.0, 10.0, 50.0, 100.0, 500.0, float("inf")]
    frp_labels = ["<1", "1-10", "10-50", "50-100", "100-500", "500+"]
    false_positives["frp_bucket"] = pd.cut(
        false_positives["frp_metric"],
        bins=frp_bins,
        labels=frp_labels,
        include_lowest=True,
        right=False,
    ).astype("string")
    frp_counts = false_positives["frp_bucket"].fillna("UNKNOWN").value_counts(dropna=False)

    print("")
    print(f"=== FP by Thermal Intensity ({context.frp_col}) ===")
    for line in _format_table(frp_counts, total_fp):
        print(line)

    lat = pd.to_numeric(false_positives[context.lat_col], errors="coerce")
    lon = pd.to_numeric(false_positives[context.lon_col], errors="coerce")
    valid_coords = lat.notna() & lon.notna()
    cluster_label = np.full(total_fp, -1, dtype=int)
    if bool(valid_coords.any()):
        coords = np.column_stack(
            [
                lat.loc[valid_coords].to_numpy(dtype=float),
                lon.loc[valid_coords].to_numpy(dtype=float),
            ]
        )
        db = DBSCAN(
            eps=float(args.dbscan_eps_m) / _EARTH_RADIUS_M,
            min_samples=1,
            metric="haversine",
            algorithm="ball_tree",
        )
        cluster_label[valid_coords.to_numpy(dtype=bool)] = db.fit_predict(np.radians(coords))

    false_positives["cluster_id"] = cluster_label
    clustered = false_positives[false_positives["cluster_id"] >= 0]
    if clustered.empty:
        raise RuntimeError(
            "BLOCKER: I cannot verify FP spatial contiguity because no false-positive rows have usable "
            "latitude/longitude coordinates in the selected snapshot."
        )

    cluster_sizes = clustered["cluster_id"].value_counts().sort_index()
    cluster_bucket_map = _contiguity_categories(cluster_sizes)
    false_positives["contiguity_bucket"] = false_positives["cluster_id"].map(cluster_bucket_map).fillna("missing_coords")
    contiguity_counts = false_positives["contiguity_bucket"].value_counts(dropna=False)

    singleton_points = int((clustered["cluster_id"].map(cluster_sizes) == 1).sum())
    multi_pixel_points = int((clustered["cluster_id"].map(cluster_sizes) >= 2).sum())
    n_clusters = int(cluster_sizes.shape[0])

    print("")
    print(f"=== FP Spatial Contiguity (DBSCAN, eps={float(args.dbscan_eps_m):.1f}m) ===")
    print(f"  - fp_points_with_valid_coords: {int(valid_coords.sum())} ({_format_pct(int(valid_coords.sum()), total_fp)})")
    print(f"  - fp_clusters: {n_clusters}")
    print(f"  - singleton_fp_points: {singleton_points} ({_format_pct(singleton_points, total_fp)})")
    print(f"  - contiguous_multi_pixel_fp_points: {multi_pixel_points} ({_format_pct(multi_pixel_points, total_fp)})")
    for line in _format_table(contiguity_counts, total_fp):
        print(line)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only FP diagnostics for denoiser eval snapshots."
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default="models/denoiser_v2",
        help="Model run root used for auto-discovery mode.",
    )
    parser.add_argument(
        "--model-run",
        type=str,
        default=None,
        help="Explicit model run dir containing model_bundle.pkl.",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Explicit snapshot path (dir with eval.parquet or parquet file).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold. Defaults to model bundle decision threshold.",
    )
    parser.add_argument(
        "--dbscan-eps-m",
        type=float,
        default=_DEFAULT_DBSCAN_EPS_M,
        help="DBSCAN epsilon in meters for FP contiguity analysis.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = analyze_false_positives(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
