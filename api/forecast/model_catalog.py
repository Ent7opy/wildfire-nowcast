"""Approved spread-model catalog and request-side resolution helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from api.model_registry import resolve_active_model
from ml.spread.factory import normalize_model_selection

SPREAD_MODEL_CATALOG_JSON_ENV = "SPREAD_MODEL_CATALOG_JSON"
SPREAD_MODEL_CATALOG_SIGNATURE_ENV = "SPREAD_MODEL_CATALOG_SIGNATURE"
SPREAD_MODEL_CATALOG_SIGNING_KEY_ENV = "SPREAD_MODEL_CATALOG_SIGNING_KEY"
SPREAD_MODEL_CATALOG_REQUIRE_SIGNATURE_ENV = "SPREAD_MODEL_CATALOG_REQUIRE_SIGNATURE"

# Safe default: only heuristic baseline is selectable without explicit catalog config.
DEFAULT_MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "v0_default": {
        "model_name": "HeuristicSpreadModelV0",
        "model_params": {},
    }
}


def _resolve_promoted_spread_catalog_entry() -> tuple[str, tuple[str, dict[str, Any]]] | None:
    """Resolve promoted spread model as a catalog entry.

    Returns a `(model_id, (model_name, model_params))` tuple or None if unavailable.
    """
    active = resolve_active_model("spread")
    if not active:
        return None

    model_id = active.get("model_id")
    artifact_uri = active.get("artifact_uri")
    if not model_id or not artifact_uri:
        return None

    artifact_path = Path(str(artifact_uri))
    inferred_model_name = "LearnedSpreadModelV1"
    try:
        meta_path = artifact_path / "metadata.json"
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            name = payload.get("model_name")
            if name in {"LearnedSpreadModelV1", "LearnedSpreadModelV2"}:
                inferred_model_name = str(name)
        elif (artifact_path / "model.int8.onnx").exists() or (artifact_path / "model.onnx").exists():
            inferred_model_name = "LearnedSpreadModelV2"
    except Exception:
        inferred_model_name = "LearnedSpreadModelV1"

    model_name, model_params = normalize_model_selection(
        inferred_model_name,
        {"model_run_dir": str(artifact_uri)},
    )
    return str(model_id), (model_name, model_params)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _canonical_catalog_json(source: dict[str, Any]) -> str:
    return json.dumps(source, sort_keys=True, separators=(",", ":"))


def compute_catalog_signature(source: dict[str, Any], signing_key: str) -> str:
    canonical = _canonical_catalog_json(source)
    return hmac.new(
        signing_key.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _verify_catalog_signature_or_raise(source: dict[str, Any]) -> None:
    require_signature = _env_bool(SPREAD_MODEL_CATALOG_REQUIRE_SIGNATURE_ENV, default=False)
    provided_signature = os.getenv(SPREAD_MODEL_CATALOG_SIGNATURE_ENV)

    if not require_signature and (provided_signature is None or not provided_signature.strip()):
        return

    signing_key = os.getenv(SPREAD_MODEL_CATALOG_SIGNING_KEY_ENV)
    if signing_key is None or not signing_key.strip():
        raise ValueError(
            f"{SPREAD_MODEL_CATALOG_SIGNING_KEY_ENV} is required when catalog signature verification is enabled."
        )

    if provided_signature is None or not provided_signature.strip():
        raise ValueError(
            f"{SPREAD_MODEL_CATALOG_SIGNATURE_ENV} is required when signature verification is enabled."
        )

    expected_signature = compute_catalog_signature(source, signing_key.strip())
    if not hmac.compare_digest(expected_signature, provided_signature.strip()):
        raise ValueError(
            f"Invalid {SPREAD_MODEL_CATALOG_SIGNATURE_ENV}: signature mismatch."
        )


def _normalize_catalog_entry(raw_entry: dict[str, Any], model_id: str) -> tuple[str, dict[str, Any]]:
    model_name = raw_entry.get("model_name")
    model_params = raw_entry.get("model_params", {})
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"Invalid catalog entry for {model_id!r}: missing model_name.")
    if not isinstance(model_params, dict):
        raise ValueError(f"Invalid catalog entry for {model_id!r}: model_params must be an object.")
    return normalize_model_selection(model_name, model_params)


@lru_cache(maxsize=1)
def get_spread_model_catalog() -> dict[str, tuple[str, dict[str, Any]]]:
    """Load and validate the spread model catalog from environment JSON."""
    raw = os.getenv(SPREAD_MODEL_CATALOG_JSON_ENV)
    if raw is None or not raw.strip():
        source = DEFAULT_MODEL_CATALOG
    else:
        try:
            source = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid {SPREAD_MODEL_CATALOG_JSON_ENV}: {e}") from e
        if not isinstance(source, dict):
            raise ValueError(f"Invalid {SPREAD_MODEL_CATALOG_JSON_ENV}: expected JSON object.")
        _verify_catalog_signature_or_raise(source)

    catalog: dict[str, tuple[str, dict[str, Any]]] = {}
    for model_id, entry in source.items():
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Spread model catalog keys must be non-empty strings.")
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid catalog entry for {model_id!r}: expected object.")
        catalog[model_id] = _normalize_catalog_entry(entry, model_id)

    promoted = _resolve_promoted_spread_catalog_entry()
    if promoted is not None:
        promoted_model_id, promoted_entry = promoted
        catalog.setdefault(promoted_model_id, promoted_entry)
    return catalog


def resolve_request_model_selection(
    *,
    model_id: str | None,
    model_name: str | None,
    model_params: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], str | None]:
    """Resolve external request model selection to a safe concrete model config."""
    if model_id:
        catalog = get_spread_model_catalog()
        resolved = catalog.get(model_id)
        if resolved is None:
            raise ValueError(
                f"Unsupported model_id: {model_id}. Available: {sorted(catalog.keys())}"
            )
        if model_name is not None and model_name != resolved[0]:
            raise ValueError("model_name conflicts with selected model_id.")
        if model_params is not None and model_params != resolved[1]:
            raise ValueError("model_params conflicts with selected model_id.")
        return resolved[0], dict(resolved[1]), model_id

    if model_name is None and model_params is None:
        promoted = _resolve_promoted_spread_catalog_entry()
        if promoted is not None:
            promoted_model_id, promoted_entry = promoted
            return promoted_entry[0], dict(promoted_entry[1]), promoted_model_id

    normalized_params = dict(model_params or {})
    # Block raw artifact paths from request surface; require catalog model_id instead.
    if (
        "model_run_dir" in normalized_params
        or "calibrator_run_dir" in normalized_params
        or "onnx_filename" in normalized_params
    ):
        raise ValueError(
            "Direct model artifact paths are not allowed in requests. Use model_id from the approved catalog."
        )
    if model_name in {"LearnedSpreadModelV1", "LearnedSpreadModelV2"}:
        raise ValueError(
            f"{model_name} must be selected via model_id from the approved catalog."
        )
    name, params = normalize_model_selection(model_name, normalized_params)
    return name, params, None
