"""Model registry helpers for explicit model promotion/rollback workflows."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from api.db import get_engine

MODEL_FAMILIES = {"denoiser", "spread"}

LOGGER = logging.getLogger(__name__)


def _notify_model_event(
    action: str,
    family: str,
    model_id: str,
    promoted_by: str | None,
    notes: str | None,
) -> None:
    try:
        from api.notifications import notify  # noqa: PLC0415

        severity = "warning" if action == "rollback" else "info"
        body_parts = [f"model_id={model_id}", f"by={promoted_by or 'unknown'}"]
        if notes:
            body_parts.append(notes.rstrip("."))
        notify(
            f"model_{action}:{family}",
            title=f"Model {action}: {family}",
            body=". ".join(body_parts) + ".",
            severity=severity,
            family=family,
            model_id=model_id,
            action=action,
            promoted_by=promoted_by or "unknown",
        )
    except Exception:
        LOGGER.debug("Failed to send model-%s notification for family=%s", action, family)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_family(family: str) -> str:
    normalized = str(family or "").strip().lower()
    if normalized not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model family: {family}. Expected one of {sorted(MODEL_FAMILIES)}")
    return normalized


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned or "model"


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _build_model_id(family: str, artifact_uri: str) -> str:
    stem = _slugify(artifact_uri.rsplit("/", 1)[-1])[:32]
    stamp = _utc_now().strftime("%Y%m%d%H%M%S")
    suffix = uuid4().hex[:8]
    return f"{family}-{stem}-{stamp}-{suffix}"


def register_model(
    *,
    family: str,
    artifact_uri: str,
    metrics_json: dict[str, Any] | None = None,
    status: str = "registered",
    model_id: str | None = None,
    engine: Engine | None = None,
) -> str:
    """Register a model artifact in the model registry and return its model_id."""
    family_norm = _normalize_family(family)
    artifact = str(artifact_uri or "").strip()
    if not artifact:
        raise ValueError("artifact_uri is required")

    resolved_model_id = model_id or _build_model_id(family_norm, artifact)

    stmt = text(
        """
        INSERT INTO model_registry (
            model_id,
            family,
            artifact_uri,
            metrics_json,
            status,
            created_at,
            updated_at
        )
        VALUES (
            :model_id,
            :family,
            :artifact_uri,
            :metrics_json,
            :status,
            NOW(),
            NOW()
        )
        """
    ).bindparams(bindparam("metrics_json", type_=JSONB))

    db = engine or get_engine()
    with db.begin() as conn:
        conn.execute(
            stmt,
            {
                "model_id": resolved_model_id,
                "family": family_norm,
                "artifact_uri": artifact,
                "metrics_json": metrics_json or {},
                "status": status,
            },
        )

    return resolved_model_id


def update_model_metrics_json(
    *,
    family: str,
    model_id: str,
    metrics_json: dict[str, Any],
    merge: bool = True,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """Update metrics_json for a registered model and return the updated registry row."""
    family_norm = _normalize_family(family)
    if not isinstance(metrics_json, dict):
        raise ValueError("metrics_json must be a JSON object")

    db = engine or get_engine()
    current = _get_registry_row(family=family_norm, model_id=model_id, engine=db)
    if current is None:
        raise ValueError(f"Model not found for family={family_norm}: {model_id}")

    current_metrics = current.get("metrics_json") if isinstance(current.get("metrics_json"), dict) else {}
    if merge:
        updated_metrics = dict(current_metrics)
        updated_metrics.update(metrics_json)
    else:
        updated_metrics = dict(metrics_json)

    stmt = text(
        """
        UPDATE model_registry
        SET metrics_json = :metrics_json,
            updated_at = NOW()
        WHERE family = :family
          AND model_id = :model_id
        """
    ).bindparams(bindparam("metrics_json", type_=JSONB))

    with db.begin() as conn:
        conn.execute(
            stmt,
            {
                "family": family_norm,
                "model_id": model_id,
                "metrics_json": updated_metrics,
            },
        )

    refreshed = _get_registry_row(family=family_norm, model_id=model_id, engine=db)
    if refreshed is None:
        raise RuntimeError(f"Failed to refresh model row after metrics update: {model_id}")
    return refreshed


def _get_registry_row(
    *,
    family: str,
    model_id: str,
    engine: Engine | None = None,
) -> dict[str, Any] | None:
    stmt = text(
        """
        SELECT model_id, family, artifact_uri, metrics_json, status, created_at, updated_at
        FROM model_registry
        WHERE model_id = :model_id
          AND family = :family
        """
    )
    db = engine or get_engine()
    with db.begin() as conn:
        row = conn.execute(
            stmt,
            {
                "model_id": model_id,
                "family": family,
            },
        ).mappings().first()
    if row is None:
        return None
    payload = dict(row)
    payload["created_at"] = _as_utc(payload.get("created_at"))
    payload["updated_at"] = _as_utc(payload.get("updated_at"))
    return payload


def promote_model(
    *,
    family: str,
    model_id: str,
    promoted_by: str | None = None,
    notes: str | None = None,
    engine: Engine | None = None,
    _notify: bool = True,
) -> dict[str, Any]:
    """Promote a registered model to active champion for its family."""
    family_norm = _normalize_family(family)
    db = engine or get_engine()

    if _get_registry_row(family=family_norm, model_id=model_id, engine=db) is None:
        raise ValueError(f"Model not found for family={family_norm}: {model_id}")

    get_existing_stmt = text(
        """
        SELECT family, model_id, rollback_model_id
        FROM model_promotions
        WHERE family = :family
        """
    )
    promote_stmt = text(
        """
        INSERT INTO model_promotions (
            family,
            model_id,
            promoted_at,
            promoted_by,
            rollback_model_id,
            notes,
            updated_at
        )
        VALUES (
            :family,
            :model_id,
            NOW(),
            :promoted_by,
            :rollback_model_id,
            :notes,
            NOW()
        )
        ON CONFLICT (family)
        DO UPDATE SET
            model_id = EXCLUDED.model_id,
            promoted_at = EXCLUDED.promoted_at,
            promoted_by = EXCLUDED.promoted_by,
            rollback_model_id = EXCLUDED.rollback_model_id,
            notes = EXCLUDED.notes,
            updated_at = NOW()
        """
    )

    demote_previous_stmt = text(
        """
        UPDATE model_registry
        SET status = 'registered',
            updated_at = NOW()
        WHERE family = :family
          AND model_id != :model_id
          AND status = 'promoted'
        """
    )
    mark_promoted_stmt = text(
        """
        UPDATE model_registry
        SET status = 'promoted',
            updated_at = NOW()
        WHERE family = :family
          AND model_id = :model_id
        """
    )

    with db.begin() as conn:
        existing = conn.execute(get_existing_stmt, {"family": family_norm}).mappings().first()
        previous_model_id = existing.get("model_id") if existing else None
        rollback_model_id = previous_model_id if previous_model_id and previous_model_id != model_id else None

        conn.execute(
            promote_stmt,
            {
                "family": family_norm,
                "model_id": model_id,
                "promoted_by": promoted_by,
                "rollback_model_id": rollback_model_id,
                "notes": notes,
            },
        )
        conn.execute(
            demote_previous_stmt,
            {
                "family": family_norm,
                "model_id": model_id,
            },
        )
        conn.execute(
            mark_promoted_stmt,
            {
                "family": family_norm,
                "model_id": model_id,
            },
        )

    active = resolve_active_model(family_norm, engine=db)
    if active is None:
        raise RuntimeError(f"Promotion failed to resolve active model for family={family_norm}")
    if _notify:
        _notify_model_event("promoted", family_norm, model_id, promoted_by, notes)
    return active


def rollback_model(
    *,
    family: str,
    promoted_by: str | None = None,
    notes: str | None = None,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """Rollback to the previously promoted model for a family."""
    family_norm = _normalize_family(family)
    db = engine or get_engine()

    stmt = text(
        """
        SELECT model_id, rollback_model_id
        FROM model_promotions
        WHERE family = :family
        """
    )
    with db.begin() as conn:
        row = conn.execute(stmt, {"family": family_norm}).mappings().first()

    if row is None:
        raise ValueError(f"No promotion exists for family={family_norm}")

    rollback_model_id = row.get("rollback_model_id")
    if not rollback_model_id:
        raise ValueError(f"No rollback target recorded for family={family_norm}")

    merged_notes = (notes or "").strip()
    if not merged_notes:
        merged_notes = "rollback"

    _notify_model_event("rollback", family_norm, str(rollback_model_id), promoted_by, merged_notes)
    return promote_model(
        family=family_norm,
        model_id=str(rollback_model_id),
        promoted_by=promoted_by,
        notes=merged_notes,
        engine=db,
        _notify=False,
    )


def resolve_active_model(family: str, *, engine: Engine | None = None) -> dict[str, Any] | None:
    """Resolve the currently promoted model for a family.

    Returns None if no promoted model exists or registry tables are unavailable.
    """
    family_norm = _normalize_family(family)
    stmt = text(
        """
        SELECT
            p.family,
            p.model_id,
            p.promoted_at,
            p.promoted_by,
            p.rollback_model_id,
            p.notes,
            r.artifact_uri,
            r.metrics_json,
            r.status,
            r.created_at,
            r.updated_at
        FROM model_promotions p
        JOIN model_registry r ON r.model_id = p.model_id
        WHERE p.family = :family
        """
    )

    db = engine or get_engine()
    try:
        with db.begin() as conn:
            row = conn.execute(stmt, {"family": family_norm}).mappings().first()
    except SQLAlchemyError:
        return None

    if row is None:
        return None

    payload = dict(row)
    payload["created_at"] = _as_utc(payload.get("created_at"))
    payload["updated_at"] = _as_utc(payload.get("updated_at"))
    payload["promoted_at"] = _as_utc(payload.get("promoted_at"))
    return payload


def validate_gate_report(metrics_json: dict[str, Any] | None) -> tuple[bool, str]:
    """Validate that a model's gate report passes before promotion.

    Gate report is expected at ``metrics_json["gate_report"]`` with a ``"pass"`` boolean.
    Returns ``(is_valid, reason)`` — reason is empty string on success.

    Shared between the API and CLI promotion code paths.
    """
    if not metrics_json:
        return False, "No metrics_json found on model"

    gate_report = metrics_json.get("gate_report")
    if gate_report is None:
        return False, "No gate_report in metrics_json"

    if not isinstance(gate_report, dict):
        return False, "gate_report must be a JSON object"

    if not gate_report.get("pass"):
        reason = gate_report.get("reason") or gate_report.get("failure_reason") or "gate_report.pass is false"
        return False, str(reason)

    return True, ""


def validate_model_gate(family: str, model_id: str, *, engine: Engine | None = None) -> None:
    """Assert that a model exists and its gate report passes promotion criteria.

    Raises ``ValueError`` on any failure — invalid family, missing model, or
    gate report not passing.  Shared pre-condition for API and CLI promotion flows.
    """
    family_norm = _normalize_family(family)
    db = engine or get_engine()
    row = _get_registry_row(family=family_norm, model_id=model_id, engine=db)
    if row is None:
        raise ValueError(f"Model not found for family={family_norm}: {model_id}")
    is_valid, reason = validate_gate_report(row.get("metrics_json"))
    if not is_valid:
        raise ValueError(f"Gate report validation failed: {reason}")


def list_active_models(*, engine: Engine | None = None) -> dict[str, dict[str, Any]]:
    """Return active promoted models keyed by family."""
    db = engine or get_engine()
    payload: dict[str, dict[str, Any]] = {}
    for family in sorted(MODEL_FAMILIES):
        active = resolve_active_model(family, engine=db)
        if active is not None:
            payload[family] = active
    return payload
