"""Wildfire spread forecasting models."""

from ml.spread.contract import (
    DEFAULT_HORIZONS_HOURS,
    SpreadForecast,
    SpreadModel,
    SpreadModelInput,
)
from ml.spread.heuristic_v0 import HeuristicSpreadModelV0, HeuristicSpreadV0Config
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
    "HeuristicSpreadModelV0",
    "HeuristicSpreadV0Config",
]

if LearnedSpreadModelV1 is not None:  # pragma: no cover
    __all__.append("LearnedSpreadModelV1")

