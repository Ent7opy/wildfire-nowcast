"""Parquet I/O helpers with deterministic engine fallback."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

_PARQUET_ENGINES: tuple[str, ...] = ("pyarrow", "fastparquet")


def read_parquet_with_fallback(
    path: str,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    errors: list[str] = []
    for engine in _PARQUET_ENGINES:
        try:
            return pd.read_parquet(path, columns=columns, engine=engine)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            errors.append(f"{engine}: {exc!r}")
    raise RuntimeError(f"Unable to read parquet at {path}. Tried engines: {', '.join(errors)}")


def write_parquet_with_fallback(df: pd.DataFrame, path: str, *, index: bool = False) -> None:
    errors: list[str] = []
    for engine in _PARQUET_ENGINES:
        try:
            df.to_parquet(path, index=index, engine=engine)
            return
        except Exception as exc:  # pragma: no cover - depends on optional deps
            errors.append(f"{engine}: {exc!r}")
    raise RuntimeError(f"Unable to write parquet at {path}. Tried engines: {', '.join(errors)}")
