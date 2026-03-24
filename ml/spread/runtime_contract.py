"""Runtime contract for spread model feature channels.

This module is the single source of truth for channel schema, tensor specs,
and feature order across train and infer. Any drift between the hindcast
dataset builder and the inference feature builder is a hard stop.

Usage at training time:
    write_contract(model_run_dir / "runtime_contract.json",
                   SpreadRuntimeContract(channels=CANONICAL_V2_CHANNELS))

Usage at inference time:
    contract = load_contract(model_run_dir / "runtime_contract.json")
    validate_channel_alignment(model.channel_names, contract.channels)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Canonical channel definitions
# ---------------------------------------------------------------------------

#: Authoritative channel list and order for v2/v3 spatial spread models.
#: This tuple is imported by both hindcast_dataset.py (training) and
#: learned_v2.py (inference). Never redefine it locally elsewhere.
CANONICAL_V2_CHANNELS: tuple[str, ...] = (
    "fire_t0",
    "fire_t-6h",
    "fire_t-12h",
    "u10",
    "v10",
    "t2m",
    "rh2m",
    "precip_24h",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "elevation_m",
    "ruggedness",
    "tpi",
    "ndvi",
    "lfmc",
    "dfmc",
    "region_id_embedding_input",
)

#: v3 shares the same channel schema as v2.
CANONICAL_V3_CHANNELS: tuple[str, ...] = CANONICAL_V2_CHANNELS

#: Map model class name → canonical channels, for gate-time lookup.
CANONICAL_CHANNELS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "LearnedSpreadModelV2": CANONICAL_V2_CHANNELS,
    "LearnedSpreadModelV3": CANONICAL_V3_CHANNELS,
}

# ---------------------------------------------------------------------------
# Contract dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpreadRuntimeContract:
    """Immutable specification of the feature tensor a model was trained on."""

    channels: tuple[str, ...]
    dtype: str = "float32"
    layout: str = "CHW"  # channel-first spatial tensor

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def to_dict(self) -> dict[str, Any]:
        return {"channels": list(self.channels), "dtype": self.dtype, "layout": self.layout}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpreadRuntimeContract":
        return cls(
            channels=tuple(d["channels"]),
            dtype=d.get("dtype", "float32"),
            layout=d.get("layout", "CHW"),
        )


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ContractViolationError(RuntimeError):
    """Raised when train-time and infer-time feature channels diverge."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_channel_alignment(
    infer_channels: Sequence[str],
    dataset_channels: Sequence[str],
) -> None:
    """Raise ContractViolationError with a human-readable diff if channels diverge.

    Both name AND order must match — a reordering is a silent data corruption,
    not a recoverable warning.
    """
    infer = list(infer_channels)
    dataset = list(dataset_channels)

    if infer == dataset:
        return

    infer_set = set(infer)
    dataset_set = set(dataset)

    lines: list[str] = []
    missing_in_infer = sorted(dataset_set - infer_set)
    extra_in_infer = sorted(infer_set - dataset_set)

    if missing_in_infer:
        lines.append(f"  channels in dataset but missing from inference: {missing_in_infer}")
    if extra_in_infer:
        lines.append(f"  channels in inference but absent from dataset: {extra_in_infer}")

    if not lines:
        # Same set, but wrong order.
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(infer, dataset)) if a != b), len(infer)
        )
        lines.append(
            f"  channel order mismatch starting at index {first_diff}: "
            f"inference has {infer[first_diff]!r}, dataset expects {dataset[first_diff]!r}"
        )

    raise ContractViolationError(
        "STOP: feature channel mismatch between inference builder and training dataset.\n"
        + "\n".join(lines)
    )


def validate_feature_tensor(tensor: np.ndarray, contract: SpreadRuntimeContract) -> None:
    """Raise ContractViolationError if the tensor C-dimension doesn't match the contract.

    Expects layout CHW (ndim == 3) or NCHW (ndim == 4, validates dim 1).
    """
    arr = np.asarray(tensor)
    if arr.ndim == 3:
        c = arr.shape[0]
    elif arr.ndim == 4:
        c = arr.shape[1]
    else:
        raise ContractViolationError(
            f"STOP: expected 3-D (CHW) or 4-D (NCHW) tensor, got ndim={arr.ndim}"
        )

    if c != contract.n_channels:
        raise ContractViolationError(
            f"STOP: tensor has {c} channels, contract requires {contract.n_channels}."
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_contract(path: Path, contract: SpreadRuntimeContract) -> None:
    """Write runtime_contract.json to *path* (file path, not directory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_contract(path: Path) -> SpreadRuntimeContract:
    """Load runtime_contract.json from *path*.

    Raises FileNotFoundError if the file is absent — absence is a hard stop,
    not a condition to silently skip.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"STOP: runtime_contract.json not found at {path}. "
            "Re-export the model to generate a contract file."
        )
    return SpreadRuntimeContract.from_dict(payload)
