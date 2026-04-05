"""Runtime contract for denoiser v2 feature schema.

This module is the single source of truth for feature names and order
across train and infer. Any drift between training and inference is a hard stop.

Usage at training time:
    contract = DenoiserRuntimeContract(features=features)
    write_contract(model_run_dir / "runtime_contract.json", contract)

Usage at inference time:
    contract = load_contract(model_run_dir / "runtime_contract.json")
    validate_feature_alignment(inference_feature_list, contract.features)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Contract dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenoiserRuntimeContract:
    """Immutable specification of the feature list a model was trained on.

    ``dtype`` defaults to float32, the standard for denoiser feature tensors.
    Features are stored as an ordered tuple to enforce immutability and
    explicit ordering at validation time.
    """

    features: tuple[str, ...]
    dtype: str = "float32"

    @property
    def n_features(self) -> int:
        return len(self.features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DenoiserRuntimeContract":
        return cls(
            features=tuple(d["features"]),
            dtype=d.get("dtype", "float32"),
        )


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ContractViolationError(RuntimeError):
    """Raised when train-time and infer-time features diverge."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_feature_alignment(
    infer_features: Sequence[str],
    contract_features: Sequence[str],
) -> None:
    """Raise ContractViolationError with a human-readable diff if features diverge.

    Both name AND order must match — a reordering is a silent data corruption,
    not a recoverable warning.
    """
    infer = list(infer_features)
    contract = list(contract_features)

    if infer == contract:
        return

    infer_set = set(infer)
    contract_set = set(contract)

    lines: list[str] = []
    missing_in_infer = sorted(contract_set - infer_set)
    extra_in_infer = sorted(infer_set - contract_set)

    if missing_in_infer:
        lines.append(f"  features in contract but missing from inference: {missing_in_infer}")
    if extra_in_infer:
        lines.append(f"  features in inference but absent from contract: {extra_in_infer}")

    if not lines:
        # Same set, but wrong order.
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(infer, contract)) if a != b), len(infer)
        )
        lines.append(
            f"  feature order mismatch starting at index {first_diff}: "
            f"inference has {infer[first_diff]!r}, contract expects {contract[first_diff]!r}"
        )

    raise ContractViolationError(
        "STOP: feature mismatch between inference builder and training contract.\n"
        + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_contract(path: Path, contract: DenoiserRuntimeContract) -> None:
    """Write runtime_contract.json to *path* (file path, not directory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_contract(path: Path) -> DenoiserRuntimeContract:
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
    return DenoiserRuntimeContract.from_dict(payload)
