"""Weather bias correction utilities.

This module implements a lightweight, dependency-free bias correction layer for
forecast weather fields (e.g. GFS) against a "truth" dataset (e.g. ERA5).

Design goals:
- Easy to fit on aligned xarray datasets (forecast + truth).
- Fast to apply at inference time (simple affine transforms).
- Serializable to JSON so API/spread inference can load it without ML frameworks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

WEATHER_BIAS_CORRECTOR_ENV = "WEATHER_BIAS_CORRECTOR_PATH"
WEATHER_BIAS_CORRECTOR_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class AffineCorrection:
    """A simple per-variable affine correction: corrected = alpha + beta * x."""

    alpha: float
    beta: float
    # Optional output range clamp (useful for bounded variables like RH).
    clamp_min: float | None = None
    clamp_max: float | None = None

    def apply(self, x: xr.DataArray) -> xr.DataArray:
        y = self.alpha + self.beta * x
        if self.clamp_min is not None or self.clamp_max is not None:
            y = y.clip(min=self.clamp_min, max=self.clamp_max)
        return y

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "clamp_min": None if self.clamp_min is None else float(self.clamp_min),
            "clamp_max": None if self.clamp_max is None else float(self.clamp_max),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "AffineCorrection":
        return AffineCorrection(
            alpha=float(d["alpha"]),
            beta=float(d["beta"]),
            clamp_min=None if d.get("clamp_min") is None else float(d["clamp_min"]),
            clamp_max=None if d.get("clamp_max") is None else float(d["clamp_max"]),
        )


@dataclass(frozen=True, slots=True)
class WeatherBiasCorrector:
    """A collection of per-variable affine corrections."""

    corrections: Mapping[str, AffineCorrection]

    @staticmethod
    def _fit_affine(forecast: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
        """Fit truth ≈ alpha + beta * forecast using least squares."""
        x = np.asarray(forecast, dtype=float).ravel()
        y = np.asarray(truth, dtype=float).ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            # No valid samples: identity transform.
            return 0.0, 1.0

        x = x[valid]
        y = y[valid]

        # Degenerate case: constant forecast -> best is alpha=mean(y), beta=0.
        x_var = float(np.var(x))
        if x_var < 1e-18:
            return float(np.mean(y)), 0.0

        # polyfit returns slope then intercept for deg=1.
        beta, alpha = np.polyfit(x, y, deg=1)
        return float(alpha), float(beta)

    @classmethod
    def fit(
        cls,
        *,
        forecast: xr.Dataset,
        truth: xr.Dataset,
        variables: Sequence[str],
        rh_var_names: Iterable[str] = ("rh2m",),
    ) -> "WeatherBiasCorrector":
        """Fit a corrector on aligned forecast/truth datasets.

        Notes
        -----
        - `forecast` and `truth` must already be aligned on dims/coords.
        - Fitting is done globally over all points (time/lat/lon) to keep the
          runtime + storage small and inference fast.
        """
        missing_fc = [v for v in variables if v not in forecast.data_vars]
        missing_tr = [v for v in variables if v not in truth.data_vars]
        if missing_fc or missing_tr:
            raise ValueError(
                "Cannot fit WeatherBiasCorrector: missing variables. "
                f"missing_forecast={missing_fc!r} missing_truth={missing_tr!r}"
            )

        # Basic alignment check: same dims and sizes for the fields we use.
        for v in variables:
            if tuple(forecast[v].dims) != tuple(truth[v].dims):
                raise ValueError(
                    f"Forecast/truth dims mismatch for {v!r}: "
                    f"forecast={forecast[v].dims} truth={truth[v].dims}"
                )
            for d in forecast[v].dims:
                if int(forecast.sizes[d]) != int(truth.sizes[d]):
                    raise ValueError(
                        f"Forecast/truth size mismatch for {v!r} dim {d!r}: "
                        f"forecast={int(forecast.sizes[d])} truth={int(truth.sizes[d])}"
                    )

        rh_vars = set(rh_var_names)
        out: dict[str, AffineCorrection] = {}
        for v in variables:
            alpha, beta = cls._fit_affine(forecast[v].values, truth[v].values)
            if v in rh_vars:
                out[v] = AffineCorrection(alpha=alpha, beta=beta, clamp_min=0.0, clamp_max=100.0)
            else:
                out[v] = AffineCorrection(alpha=alpha, beta=beta)
        return cls(corrections=out)

    def apply(self, ds: xr.Dataset, *, inplace: bool = False) -> xr.Dataset:
        """Apply corrections to a dataset (variables updated in-place or copied)."""
        out = ds if inplace else ds.copy(deep=False)
        for var, corr in self.corrections.items():
            if var not in out.data_vars:
                continue
            out[var] = corr.apply(out[var])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": WEATHER_BIAS_CORRECTOR_FORMAT_VERSION,
            "corrections": {k: v.to_dict() for k, v in sorted(self.corrections.items())},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WeatherBiasCorrector":
        version = int(d.get("format_version", 0))
        if version != WEATHER_BIAS_CORRECTOR_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported WeatherBiasCorrector format_version={version}; "
                f"expected {WEATHER_BIAS_CORRECTOR_FORMAT_VERSION}"
            )
        corr = d.get("corrections", {})
        if not isinstance(corr, dict):
            raise TypeError("Expected 'corrections' to be a dict.")
        return cls(corrections={k: AffineCorrection.from_dict(v) for k, v in corr.items()})

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "WeatherBiasCorrector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def resolve_weather_bias_corrector_path(
    path: str | Path | None = None, *, env_var: str = WEATHER_BIAS_CORRECTOR_ENV
) -> Path | None:
    """Resolve a corrector path from an explicit path or an environment variable."""
    if path is not None:
        p = Path(path)
        return p
    env = os.environ.get(env_var)
    if not env:
        return None
    return Path(env)


def _resolve_latest_run_dir(root: Path) -> Path | None:
    """Return the latest run directory under root (by mtime), if any."""
    if not root.exists() or not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_weather_bias_corrector_path_full(
    region_name: str | None,
    *,
    explicit_path_env: str = WEATHER_BIAS_CORRECTOR_ENV,
    root_env: str = "WEATHER_BIAS_CORRECTOR_ROOT",
    default_repo_subpath: str = "models/weather_bias_corrector",
) -> Path | None:
    """Resolve weather bias corrector path with full fallback chain.
    
    This centralizes the path resolution logic to ensure consistency between
    the spread service and other consumers.
    
    Resolution order:
    1. Explicit file path from environment variable
    2. Region-specific directory under root environment variable
    3. Global directory under root environment variable  
    4. Region-specific directory under conventional repo path
    5. Global directory under conventional repo path
    
    Parameters
    ----------
    region_name : str | None
        Region name for region-aware resolution. If None, skips region-specific paths.
    explicit_path_env : str
        Environment variable name for explicit file path.
    root_env : str
        Environment variable name for root directory.
    default_repo_subpath : str
        Default subpath under repo root to search.
        
    Returns
    -------
    Path | None
        Path to weather_bias_corrector.json if found, None otherwise.
    """
    # 1) Explicit file env var wins.
    if (p := os.environ.get(explicit_path_env)):
        return Path(p)

    # 2) Region-aware root, else global root.
    root_env_val = os.environ.get(root_env)
    roots: list[Path] = []
    if root_env_val:
        if region_name:
            roots.append(Path(root_env_val) / region_name)
        roots.append(Path(root_env_val))

    # 3) Conventional default under repo
    repo_root = Path(__file__).resolve().parents[1]
    if region_name:
        roots.append(repo_root / default_repo_subpath / region_name)
    roots.append(repo_root / default_repo_subpath)

    for root in roots:
        latest = _resolve_latest_run_dir(root)
        if latest is None:
            # Also allow the root itself to be a run directory.
            latest = root if root.is_dir() else None
        if latest is None:
            continue
        candidate = latest / "weather_bias_corrector.json"
        if candidate.exists():
            return candidate
        if latest.is_file() and latest.name.endswith(".json"):
            return latest
    return None

