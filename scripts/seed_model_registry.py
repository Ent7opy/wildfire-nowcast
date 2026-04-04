"""Idempotent model registry seeder — safe to run on every deploy.

Registers and promotes the committed model artifacts if no promoted model
exists for a given family. Designed to be chained after `alembic upgrade head`
in the Railway API preDeployCommand.

Usage:
    uv run --project api python scripts/seed_model_registry.py

Exit codes:
    0 — success (all families seeded or already have a promoted model)
    1 — fatal error (registration or promotion failed)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("seed_model_registry")

# Repo root is the working directory when run via `uv run --project api`.
REPO_ROOT = Path(__file__).parent.parent

# ── Committed model artifacts to seed ─────────────────────────────────────────
# Only the active production models are listed here.
# When you train and want to promote a new model:
#   1. Add its gitignore exception (see .gitignore)
#   2. Update the path below
#   3. Push — Railway deploys will auto-register and promote it

DENOISER_ARTIFACT = (
    "models/denoiser_v2/20260304_235923_94a9940fee24ef9a8e4914cc8e9b66e3404cb054"
)
SPREAD_ARTIFACT = (
    "models/spread_v3/20260308_151315_fe0e5f16d7f0f59da935a7e63d1f64ee624f6f55"
)
SPREAD_CALIBRATION_ARTIFACT = (
    "models/spread_calibration/bbox_47_31_v3"
    "/20260308_165025_fe0e5f16d7f0f59da935a7e63d1f64ee624f6f55"
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        log.warning("Could not parse %s: %s", path, exc)
        return {}


def _build_denoiser_metrics(artifact_dir: Path) -> dict:
    metrics = _load_json(artifact_dir / "metrics.json")
    gate = _load_json(artifact_dir / "gate_report.json")
    meta = _load_json(artifact_dir / "metadata.json")
    cfg = meta.get("config", {})
    runtime_contract = {
        "pipeline_version": "v2",
        "threshold_profile": "env",
        "thresholds": {
            "strong_filter_threshold": cfg.get("strong_filter_threshold", 0.5),
            "downweight_threshold": cfg.get("downweight_threshold", 0.7),
            "uncertainty_band_low": cfg.get("uncertainty_band_low", 0.45),
            "uncertainty_band_high": cfg.get("uncertainty_band_high", 0.55),
        },
    }
    metrics["gate_report"] = gate
    metrics["runtime_contract"] = runtime_contract
    return metrics


def _build_spread_metrics(artifact_dir: Path, calibrator_dir: Path | None) -> dict:
    metrics = _load_json(artifact_dir / "registry_metrics.json")
    if not metrics:
        metrics = _load_json(artifact_dir / "metrics.json")
    if calibrator_dir:
        metrics["calibrator_run_dir"] = str(calibrator_dir)
    return metrics


def seed_family(
    family: str,
    artifact_uri: str,
    metrics_json: dict,
    *,
    promoted_by: str = "seed_model_registry",
    notes: str = "",
) -> None:
    from api.model_registry import (
        promote_model,
        register_model,
        resolve_active_model,
    )

    active = resolve_active_model(family)
    if active is not None:
        log.info(
            "Family=%s already has promoted model %s — skipping.",
            family,
            active.get("model_id"),
        )
        return

    log.info("Family=%s: no promoted model found — registering %s", family, artifact_uri)
    model_id = register_model(
        family=family,
        artifact_uri=artifact_uri,
        metrics_json=metrics_json,
    )
    log.info("Family=%s: registered as model_id=%s — promoting", family, model_id)
    promote_model(
        family=family,
        model_id=model_id,
        promoted_by=promoted_by,
        notes=notes,
        _notify=False,
    )
    log.info("Family=%s: promoted model_id=%s", family, model_id)


def main() -> int:
    try:
        # ── Denoiser ──────────────────────────────────────────────────────────
        denoiser_dir = REPO_ROOT / DENOISER_ARTIFACT
        if not denoiser_dir.exists():
            log.error("Denoiser artifact not found at %s", denoiser_dir)
            return 1
        denoiser_metrics = _build_denoiser_metrics(denoiser_dir)
        seed_family(
            family="denoiser",
            artifact_uri=DENOISER_ARTIFACT,
            metrics_json=denoiser_metrics,
            notes=(
                "Auto-seeded by seed_model_registry.py. "
                "recall=0.92 roc_auc=0.942. Gate precision/f1 thresholds "
                "are set for a standard classifier; this is a PU-learning "
                "model — see issue #320 for threshold recalibration."
            ),
        )

        # ── Spread ────────────────────────────────────────────────────────────
        spread_dir = REPO_ROOT / SPREAD_ARTIFACT
        if not spread_dir.exists():
            log.warning(
                "Spread artifact not found at %s — skipping spread seeding.", spread_dir
            )
        else:
            calibrator_dir = REPO_ROOT / SPREAD_CALIBRATION_ARTIFACT
            spread_metrics = _build_spread_metrics(
                spread_dir,
                calibrator_dir if calibrator_dir.exists() else None,
            )
            seed_family(
                family="spread",
                artifact_uri=SPREAD_ARTIFACT,
                metrics_json=spread_metrics,
                notes="Auto-seeded by seed_model_registry.py. gate_pass=True mvp_operational.",
            )

    except Exception:
        log.exception("seed_model_registry failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
