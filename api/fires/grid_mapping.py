"""Map fire detections (lon/lat points) onto the canonical analysis grid.

This module intentionally matches `api.core.grid` conventions:
- CRS: EPSG:4326
- Indices: `(i, j) = (lat_index, lon_index)`
- Analysis order: `i` increases south → north, `j` increases west → east
- Index rule: `floor((coord - origin) / cell)`
- Boundary rule: points on the *upper* boundary are out-of-bounds (half-open extent)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np

from api.core.grid import GridSpec, latlon_to_index

Mode = Literal["count", "presence", "sum", "max"]


def normalize_lon(lon: np.ndarray | float) -> np.ndarray | float:
    """Normalize longitudes into the interval [-180, 180].

    Notes
    - `180` stays `-180` with this wrap rule.
    - If your database already constrains longitudes to [-180, 180], this is a no-op.
    """

    arr = np.asarray(lon, dtype=float)
    norm = ((arr + 180.0) % 360.0) - 180.0
    if arr.ndim == 0:
        return float(norm)
    return norm


def _in_bounds_mask(i: np.ndarray, j: np.ndarray, n_lat: int, n_lon: int) -> np.ndarray:
    return (i >= 0) & (i < n_lat) & (j >= 0) & (j < n_lon)


def _as_numpy_column(
    rows: Sequence[Mapping[str, Any]],
    col: str,
    *,
    dtype: type | np.dtype,
) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=dtype)
    return np.asarray([r[col] for r in rows], dtype=dtype)


@overload
def fires_to_indices(
    fires: Sequence[Mapping[str, Any]],
    grid: GridSpec,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    normalize_lons: bool = True,
    drop_outside: bool = True,
) -> list[dict[str, Any]]: ...


@overload
def fires_to_indices(  # type: ignore[overload-overlap]
    fires: Any,
    grid: GridSpec,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    normalize_lons: bool = True,
    drop_outside: bool = True,
) -> Any: ...


def fires_to_indices(
    fires: Any,
    grid: GridSpec,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    normalize_lons: bool = True,
    drop_outside: bool = True,
) -> Any:
    """Add stable `(i, j)` grid indices to detections.

    Supported inputs
    - **Sequence[Mapping]**: returns `list[dict]` with added keys `i`, `j`, `in_bounds`.
    - **pandas.DataFrame** (optional): if pandas is installed and `fires` is a DataFrame,
      returns a DataFrame with added columns.
    """

    # Optional pandas support without taking a hard dependency.
    try:  # pragma: no cover (pandas is not an api dependency)
        import pandas as pd  # type: ignore

        if isinstance(fires, pd.DataFrame):
            df = fires.copy()
            lat = df[lat_col].to_numpy(dtype=float, copy=False)
            lon = df[lon_col].to_numpy(dtype=float, copy=False)
            if normalize_lons:
                lon = normalize_lon(lon)  # type: ignore[assignment]
            i, j = latlon_to_index(grid, lat=lat, lon=lon)
            in_bounds = _in_bounds_mask(i, j, grid.n_lat, grid.n_lon)
            out = df.assign(i=i, j=j, in_bounds=in_bounds)
            if drop_outside:
                out = out.loc[out["in_bounds"]].reset_index(drop=True)
            return out
    except Exception:
        pass

    if not isinstance(fires, Sequence):
        raise TypeError("fires must be a pandas.DataFrame or a Sequence[Mapping].")
    if len(fires) == 0:
        return []
    if not isinstance(fires[0], Mapping):
        raise TypeError("fires must be a pandas.DataFrame or a Sequence[Mapping].")

    rows: Sequence[Mapping[str, Any]] = fires
    lat = _as_numpy_column(rows, lat_col, dtype=float)
    lon = _as_numpy_column(rows, lon_col, dtype=float)
    if normalize_lons:
        lon = normalize_lon(lon)  # type: ignore[assignment]
    i, j = latlon_to_index(grid, lat=lat, lon=lon)
    in_bounds = _in_bounds_mask(i, j, grid.n_lat, grid.n_lon)

    out_rows: list[dict[str, Any]] = []
    for k, r in enumerate(rows):
        rr = dict(r)
        rr["i"] = int(i[k])
        rr["j"] = int(j[k])
        rr["in_bounds"] = bool(in_bounds[k])
        if (not drop_outside) or rr["in_bounds"]:
            out_rows.append(rr)
    return out_rows


def aggregate_indices_to_grid(
    i: np.ndarray,
    j: np.ndarray,
    shape: tuple[int, int],
    *,
    mode: Mode = "count",
    values: np.ndarray | None = None,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Aggregate `(i, j)` hits into a dense 2D grid.

    Parameters
    - **shape**: `(n_lat, n_lon)` in analysis order.
    - **mode**:
      - `"count"`: increment 1 per hit
      - `"presence"`: set 1 for any hit
      - `"sum"`: sum `values` per cell
      - `"max"`: max of `values` per cell
    """

    n_lat, n_lon = shape
    if dtype is None:
        if mode == "count":
            dtype = np.int32
        elif mode == "presence":
            dtype = np.uint8
        else:
            dtype = np.float32

    if i.size == 0:
        return np.zeros(shape, dtype=dtype)

    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)

    # Safety: reject out-of-range indices (caller should filter, but we keep this strict).
    if np.any(i < 0) or np.any(i >= n_lat) or np.any(j < 0) or np.any(j >= n_lon):
        raise ValueError("aggregate_indices_to_grid received out-of-bounds indices.")

    out = np.zeros(shape, dtype=dtype)

    if mode == "count":
        np.add.at(out, (i, j), 1)
        return out

    flat = (i * n_lon + j).astype(np.int64, copy=False)

    if mode == "presence":
        out.ravel()[np.unique(flat)] = 1
        return out

    if values is None:
        raise ValueError(f"mode='{mode}' requires values.")
    values = np.asarray(values)
    if values.shape[0] != i.shape[0]:
        raise ValueError("values must have the same length as i/j.")

    if mode == "sum":
        np.add.at(out, (i, j), values.astype(out.dtype, copy=False))
        return out

    if mode == "max":
        # Max-reduce per flat index.
        order = np.argsort(flat, kind="stable")
        flat_s = flat[order]
        vals_s = values[order].astype(out.dtype, copy=False)
        uniq, start_idx = np.unique(flat_s, return_index=True)
        max_vals = np.maximum.reduceat(vals_s, start_idx)
        out.ravel()[uniq] = max_vals
        return out

    raise ValueError(f"Unsupported mode: {mode}")


def aggregate_to_grid(
    fires_with_indices: Any,
    grid: GridSpec,
    *,
    mode: Mode = "count",
    value_col: str | None = None,
    dtype: np.dtype | None = None,
    drop_outside: bool = True,
) -> np.ndarray:
    """Aggregate detections (with `i`/`j`) into a `(n_lat, n_lon)` heatmap."""

    # pandas path (optional)
    try:  # pragma: no cover (pandas is not an api dependency)
        import pandas as pd  # type: ignore

        if isinstance(fires_with_indices, pd.DataFrame):
            df = fires_with_indices
            if drop_outside and "in_bounds" in df.columns:
                df = df.loc[df["in_bounds"]]
            i = df["i"].to_numpy(dtype=int, copy=False)
            j = df["j"].to_numpy(dtype=int, copy=False)
            vals = None
            if mode in ("sum", "max"):
                if not value_col:
                    raise ValueError(f"mode='{mode}' requires value_col.")
                vals = df[value_col].to_numpy(copy=False)
            return aggregate_indices_to_grid(
                i=i,
                j=j,
                shape=(grid.n_lat, grid.n_lon),
                mode=mode,
                values=vals,
                dtype=dtype,
            )
    except Exception:
        pass

    if not isinstance(fires_with_indices, Sequence):
        raise TypeError("fires_with_indices must be a pandas.DataFrame or a Sequence[Mapping].")
    rows: Sequence[Mapping[str, Any]] = fires_with_indices
    if len(rows) == 0:
        return np.zeros((grid.n_lat, grid.n_lon), dtype=(dtype or np.int32))

    # Filter using in_bounds if present, otherwise assume already filtered.
    if drop_outside and "in_bounds" in rows[0]:
        rows = [r for r in rows if bool(r.get("in_bounds"))]
        if len(rows) == 0:
            return np.zeros((grid.n_lat, grid.n_lon), dtype=(dtype or np.int32))

    i = _as_numpy_column(rows, "i", dtype=int)
    j = _as_numpy_column(rows, "j", dtype=int)

    vals = None
    if mode in ("sum", "max"):
        if not value_col:
            raise ValueError(f"mode='{mode}' requires value_col.")
        vals = _as_numpy_column(rows, value_col, dtype=float)

    return aggregate_indices_to_grid(
        i=i,
        j=j,
        shape=(grid.n_lat, grid.n_lon),
        mode=mode,
        values=vals,
        dtype=dtype,
    )

