"""Wildfire spread forecasting models."""

from ml.spread.contract import (
    DEFAULT_HORIZONS_HOURS,
    SpreadForecast,
    SpreadModel,
    SpreadModelInput,
)

try:
    # Optional dependency: heuristic model uses scipy for fast convolution.
    from ml.spread.heuristic_v0 import HeuristicSpreadModelV0, HeuristicSpreadV0Config
except Exception:  # pragma: no cover
    HeuristicSpreadModelV0 = None  # type: ignore[assignment]
    HeuristicSpreadV0Config = None  # type: ignore[assignment]
try:
    # Optional dependency: learned model may require extra packages.
    from ml.spread.learned_v1 import LearnedSpreadModelV1
except Exception:  # pragma: no cover
    LearnedSpreadModelV1 = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_HORIZONS_HOURS",
    "SpreadForecast",
    "SpreadModel",
    "SpreadModelInput",
]

if HeuristicSpreadModelV0 is not None:  # pragma: no cover
    __all__.append("HeuristicSpreadModelV0")
if HeuristicSpreadV0Config is not None:  # pragma: no cover
    __all__.append("HeuristicSpreadV0Config")

if LearnedSpreadModelV1 is not None:  # pragma: no cover
    __all__.append("LearnedSpreadModelV1")

