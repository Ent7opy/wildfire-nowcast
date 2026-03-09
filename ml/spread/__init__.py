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
try:
    # Optional dependency: v2 model uses ONNX Runtime.
    from ml.spread.learned_v2 import LearnedSpreadModelV2
except Exception:  # pragma: no cover
    LearnedSpreadModelV2 = None  # type: ignore[assignment]
try:
    # Optional dependency: v3 model uses ONNX Runtime.
    from ml.spread.learned_v3 import LearnedSpreadModelV3
except Exception:  # pragma: no cover
    LearnedSpreadModelV3 = None  # type: ignore[assignment]

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
if LearnedSpreadModelV2 is not None:  # pragma: no cover
    __all__.append("LearnedSpreadModelV2")
if LearnedSpreadModelV3 is not None:  # pragma: no cover
    __all__.append("LearnedSpreadModelV3")
