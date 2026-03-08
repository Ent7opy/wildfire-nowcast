"""Factory for creating spread models by name."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Type

from ml.spread.contract import SpreadModel
from ml.spread.heuristic_v0 import HeuristicSpreadModelV0, HeuristicSpreadV0Config
from ml.spread.learned_v1 import LearnedSpreadModelV1
from ml.spread.learned_v2 import LearnedSpreadModelV2

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LearnedSpreadV1Config:
    """Configuration for LearnedSpreadModelV1."""
    model_run_dir: str
    calibrator_run_dir: str | None = None


@dataclass(frozen=True, slots=True)
class LearnedSpreadV2Config:
    """Configuration for LearnedSpreadModelV2."""

    model_run_dir: str
    calibrator_run_dir: str | None = None
    onnx_filename: str = "model.int8.onnx"


# Registry mapping model names to (ModelClass, ConfigClass)
MODEL_REGISTRY: dict[str, tuple[Type[SpreadModel], Type[Any]]] = {
    "HeuristicSpreadModelV0": (HeuristicSpreadModelV0, HeuristicSpreadV0Config),
    "LearnedSpreadModelV1": (LearnedSpreadModelV1, LearnedSpreadV1Config),
    "LearnedSpreadModelV2": (LearnedSpreadModelV2, LearnedSpreadV2Config),
}

MODEL_DEFAULT_NAME = "HeuristicSpreadModelV0"
MODEL_VERSION_HINTS: dict[str, str] = {
    "HeuristicSpreadModelV0": "v0",
    "LearnedSpreadModelV1": "v1",
    "LearnedSpreadModelV2": "v2",
}


def normalize_model_selection(
    name: str | None,
    params: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Validate and normalize model selection from external inputs.

    Returns a concrete `(model_name, model_params)` pair suitable for
    `get_spread_model`.
    """
    selected_name = name or MODEL_DEFAULT_NAME
    if selected_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: {selected_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    if params is None:
        normalized_params: dict[str, Any] = {}
    elif isinstance(params, dict):
        normalized_params = dict(params)
    else:
        raise ValueError("model_params must be an object/dict when provided.")

    # Learned model requires an explicit artifacts directory.
    if selected_name in {"LearnedSpreadModelV1", "LearnedSpreadModelV2"} and not normalized_params.get(
        "model_run_dir"
    ):
        raise ValueError(
            f"model_params.model_run_dir is required for {selected_name}."
        )

    return selected_name, normalized_params


def get_model_version_hint(model_name: str) -> str:
    """Return a stable version hint for persistence metadata."""
    return MODEL_VERSION_HINTS.get(model_name, "")


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
    name, params = normalize_model_selection(name=name, params=params)

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
    if name == "LearnedSpreadModelV2":
        return model_cls(
            model_run_dir=model_config.model_run_dir,  # type: ignore
            calibrator_run_dir=model_config.calibrator_run_dir,  # type: ignore
            onnx_filename=model_config.onnx_filename,  # type: ignore
        )

    # Default: assume model_cls(config=model_config)
    return model_cls(config=model_config)  # type: ignore
