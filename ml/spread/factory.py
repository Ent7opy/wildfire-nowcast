"""Factory for creating spread models by name."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Type

from ml.spread.contract import SpreadModel
from ml.spread.heuristic_v0 import HeuristicSpreadModelV0, HeuristicSpreadV0Config
from ml.spread.learned_v1 import LearnedSpreadModelV1

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LearnedSpreadV1Config:
    """Configuration for LearnedSpreadModelV1."""
    model_run_dir: str
    calibrator_run_dir: str | None = None


# Registry mapping model names to (ModelClass, ConfigClass)
MODEL_REGISTRY: dict[str, tuple[Type[SpreadModel], Type[Any]]] = {
    "HeuristicSpreadModelV0": (HeuristicSpreadModelV0, HeuristicSpreadV0Config),
    "LearnedSpreadModelV1": (LearnedSpreadModelV1, LearnedSpreadV1Config),
}


def get_spread_model(name: str, params: dict[str, Any] | None = None) -> SpreadModel:
    """Instantiate a spread model by name with optional parameters.

    Parameters
    ----------
    name : str
        The name of the model to instantiate (must be in MODEL_REGISTRY).
    params : dict[str, Any] | None
        Dictionary of configuration parameters for the model.
        Unknown parameters will be filtered out with a warning.

    Returns
    -------
    SpreadModel
        The instantiated spread model.

    Raises
    ------
    ValueError
        If the model name is not found in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    if params is None:
        params = {}

    model_cls, config_cls = MODEL_REGISTRY[name]

    # Filter parameters based on the config class annotations
    valid_fields = set(config_cls.__annotations__.keys())
    unknown = set(params.keys()) - valid_fields
    if unknown:
        LOGGER.warning(
            "Ignoring unknown model_params for %s: %s",
            name,
            unknown,
        )

    valid_params = {k: v for k, v in params.items() if k in valid_fields}
    model_config = config_cls(**valid_params)

    # Instantiate model based on its specific requirements
    if name == "LearnedSpreadModelV1":
        # LearnedSpreadModelV1 takes model_run_dir directly
        return model_cls(
            model_run_dir=model_config.model_run_dir,  # type: ignore
            calibrator_run_dir=model_config.calibrator_run_dir,  # type: ignore
        )

    # Default: assume model_cls(config=model_config)
    return model_cls(config=model_config)  # type: ignore

