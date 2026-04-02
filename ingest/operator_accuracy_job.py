"""Orchestrator wrapper for the weekly operator label-quality accuracy job.

This module is the thin ingest-layer adapter for ml.denoiser.operator_accuracy,
following the same pattern as ingest/denoiser_drift_monitor.py.

The job queries denoiser_labels_v2 (source='review_queue') and
authoritative_perimeters to compute per-operator label accuracy, then upserts
results into operator_label_quality.  This feeds back into the review queue
feedback loop by adjusting the label_weight of future review queue labels.

Invoked by the orchestrator as JOB_OPERATOR_ACCURACY (weekly by default).
Can also be run standalone:
    uv run --project ingest -m ingest.operator_accuracy_job
"""

from __future__ import annotations

import json
import logging

from api.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("operator_accuracy_job")


def run_operator_accuracy_job() -> int:
    """Run one operator accuracy computation cycle.

    Returns 0 on success, 1 on failure.
    """
    # Lazy import keeps api.db out of module scope at ingest startup.
    from ml.denoiser.operator_accuracy import compute_operator_accuracy  # noqa: PLC0415

    engine = get_engine()
    result = compute_operator_accuracy(engine)
    n_operators = len(result.get("operators", []))
    LOGGER.info(
        "Operator accuracy job complete: %d operator(s) updated at %s",
        n_operators,
        result.get("computed_at"),
    )
    print(json.dumps(result))
    return 0


def main() -> None:
    import sys

    sys.exit(run_operator_accuracy_job())


if __name__ == "__main__":
    main()
