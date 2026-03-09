"""Deterministic region key helpers used across spread train/inference/runtime."""

from __future__ import annotations

import hashlib
from typing import Iterable


def bbox_region_name(bbox: Iterable[float]) -> str:
    """Return canonical bbox-based region_name used by terrain ingest defaults."""
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    return f"bbox_{min_lon:.2f}_{min_lat:.2f}_{max_lon:.2f}_{max_lat:.2f}"


def bbox_region_token(bbox: Iterable[float], *, precision: int = 4) -> str:
    """Return a stable bbox token for hashing/encoding."""
    vals = [float(v) for v in bbox]
    fmt = f"{{:.{int(precision)}f}}"
    return ",".join(fmt.format(v) for v in vals)


def deterministic_region_bucket(
    *,
    region_name: str | None = None,
    bbox: Iterable[float] | None = None,
    n_buckets: int = 1024,
) -> int:
    """Map region identity to a deterministic bucket id."""
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive.")

    if region_name:
        key = str(region_name)
    elif bbox is not None:
        key = f"bbox:{bbox_region_token(bbox, precision=4)}"
    else:
        key = "global"

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % int(n_buckets)
