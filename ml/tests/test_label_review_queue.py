"""Tests for ml.denoiser.label_review_queue.

Acceptance criteria covered:
  - resolved 'confirmed_fire' queue item produces label='POSITIVE' row
  - resolved 'marked_noise' queue item produces label='NEGATIVE' row
  - auto-resolved items (resolved_by LIKE 'auto:%') are excluded
  - conflict: operator label disagrees with perimeter label → logged to
    label_conflicts, NOT inserted into denoiser_labels_v2
  - items with unrecognised resolved_notes are skipped (counted, not inserted)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.denoiser.label_review_queue import (
    REVIEW_QUEUE_RULE_VERSION,
    ingest_review_queue_labels,
    warn_if_unprocessed_labels_exist,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(**kwargs: Any) -> MagicMock:
    """Build a MagicMock that behaves like a SQLAlchemy Row (dict-style access)."""
    row = MagicMock()
    row.__getitem__ = lambda self, key: kwargs[key]
    row._mapping = kwargs
    return row


def _make_engine(fetch_rows: list[Any]) -> tuple[MagicMock, MagicMock]:
    """Build a mock engine whose connection returns the given fetch_rows.

    After the initial fetch, further execute() calls (batch inserts) return
    a generic mock — we inspect call_args_list to assert what was inserted.
    """
    conn = MagicMock()
    engine = MagicMock()
    engine.begin.return_value.__enter__ = lambda self: conn
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    fetch_result = MagicMock()
    fetch_result.fetchall.return_value = fetch_rows

    # First execute() is always the fetch; subsequent calls are batch inserts.
    conn.execute.side_effect = [fetch_result, MagicMock(), MagicMock()]
    return engine, conn


# ---------------------------------------------------------------------------
# Tests: label mapping
# ---------------------------------------------------------------------------


def test_confirmed_fire_produces_positive_label() -> None:
    """A resolved 'confirmed_fire' queue item must emit label='POSITIVE'."""
    candidate = _row(
        queue_id=1,
        event_id="ev_001",
        resolved_by="operator",
        resolved_notes="confirmed_fire",
        fire_detection_id=42,
        existing_perimeter_label=None,  # no conflict
    )
    engine, conn = _make_engine(fetch_rows=[candidate])

    counts = ingest_review_queue_labels(engine, label_weight=0.8)

    assert counts["inserted"] == 1
    assert counts["conflicts"] == 0
    assert counts["skipped_invalid_notes"] == 0

    # The label batch insert is the second execute() call (index 1).
    # Verify the params list passed to it contains the correct label.
    label_call = conn.execute.call_args_list[1]
    label_params_list = label_call[0][1]  # list of param dicts
    assert len(label_params_list) == 1
    p = label_params_list[0]
    assert p["label"] == "POSITIVE"
    assert p["rule_version"] == REVIEW_QUEUE_RULE_VERSION
    assert p["label_weight"] == 0.8
    assert p["fire_detection_id"] == 42


def test_marked_noise_produces_negative_label() -> None:
    """A resolved 'marked_noise' queue item must emit label='NEGATIVE'."""
    candidate = _row(
        queue_id=2,
        event_id="ev_002",
        resolved_by="operator",
        resolved_notes="marked_noise",
        fire_detection_id=99,
        existing_perimeter_label=None,
    )
    engine, conn = _make_engine(fetch_rows=[candidate])

    counts = ingest_review_queue_labels(engine, label_weight=0.8)

    assert counts["inserted"] == 1
    label_params_list = conn.execute.call_args_list[1][0][1]
    assert label_params_list[0]["label"] == "NEGATIVE"
    assert label_params_list[0]["fire_detection_id"] == 99


# ---------------------------------------------------------------------------
# Tests: auto-resolved exclusion
# ---------------------------------------------------------------------------


def test_auto_resolved_items_excluded() -> None:
    """Items with resolved_by='auto:perimeter:wfigs' must NOT reach denoiser_labels_v2.

    The SQL fetch query already filters these out via WHERE resolved_by NOT LIKE 'auto:%',
    so the result set is empty — no insert calls should happen.
    """
    engine, conn = _make_engine(fetch_rows=[])

    counts = ingest_review_queue_labels(engine, label_weight=0.8)

    assert counts["inserted"] == 0
    assert counts["conflicts"] == 0
    # Only the fetch query was executed; no batch inserts.
    assert conn.execute.call_count == 1


def test_auto_resolved_pattern_verified_in_fetch_sql() -> None:
    """Confirm the fetch SQL contains the auto-resolve exclusion clause."""
    import inspect

    from ml.denoiser.label_review_queue import ingest_review_queue_labels

    src = inspect.getsource(ingest_review_queue_labels)
    assert "NOT LIKE 'auto:%'" in src or "resolved_by NOT LIKE" in src


# ---------------------------------------------------------------------------
# Tests: conflict resolution
# ---------------------------------------------------------------------------


def test_conflict_skips_label_insert_and_logs_to_label_conflicts() -> None:
    """When the operator says 'confirmed_fire' but perimeter says 'NEGATIVE',
    the conflict is logged to label_conflicts and no label is inserted."""
    candidate = _row(
        queue_id=3,
        event_id="ev_003",
        resolved_by="operator",
        resolved_notes="confirmed_fire",
        fire_detection_id=77,
        existing_perimeter_label="NEGATIVE",  # conflict
    )
    engine, conn = _make_engine(fetch_rows=[candidate])

    counts = ingest_review_queue_labels(engine, label_weight=0.8)

    assert counts["inserted"] == 0
    assert counts["conflicts"] == 1

    # The conflict batch insert is the second execute() call (index 1).
    conflict_call = conn.execute.call_args_list[1]
    conflict_params_list = conflict_call[0][1]
    assert len(conflict_params_list) == 1
    p = conflict_params_list[0]
    assert p["perimeter_label"] == "NEGATIVE"
    assert p["operator_label"] == "POSITIVE"
    assert p["fire_detection_id"] == 77


def test_no_conflict_when_labels_agree() -> None:
    """When operator and perimeter agree (both POSITIVE), no conflict is logged."""
    candidate = _row(
        queue_id=4,
        event_id="ev_004",
        resolved_by="operator",
        resolved_notes="confirmed_fire",
        fire_detection_id=55,
        existing_perimeter_label="POSITIVE",  # agreement
    )
    engine, conn = _make_engine(fetch_rows=[candidate])

    counts = ingest_review_queue_labels(engine, label_weight=0.8)

    assert counts["inserted"] == 1
    assert counts["conflicts"] == 0


# ---------------------------------------------------------------------------
# Tests: invalid / unknown resolved_notes
# ---------------------------------------------------------------------------


def test_unrecognised_resolved_notes_skipped() -> None:
    """Items with notes outside ('confirmed_fire', 'marked_noise') are skipped."""
    candidate = _row(
        queue_id=5,
        event_id="ev_005",
        resolved_by="operator",
        resolved_notes="needs_more_info",
        fire_detection_id=33,
        existing_perimeter_label=None,
    )
    engine, conn = _make_engine(fetch_rows=[candidate])

    counts = ingest_review_queue_labels(engine, label_weight=0.8)

    assert counts["inserted"] == 0
    assert counts["skipped_invalid_notes"] == 1
    # No batch inserts — only the initial fetch.
    assert conn.execute.call_count == 1


# ---------------------------------------------------------------------------
# Tests: warn_if_unprocessed_labels_exist
# ---------------------------------------------------------------------------


def test_warn_logs_when_pending_labels_exist(caplog: pytest.LogCaptureFixture) -> None:
    """warn_if_unprocessed_labels_exist must log a WARNING when count > 0."""
    count_row = _row(n=3)
    conn = MagicMock()
    engine = MagicMock()
    engine.connect.return_value.__enter__ = lambda self: conn
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    result = MagicMock()
    result.fetchone.return_value = count_row
    conn.execute.return_value = result

    import logging

    with caplog.at_level(logging.WARNING, logger="denoiser_label_review_queue"):
        count = warn_if_unprocessed_labels_exist(engine)

    assert count == 3
    assert any("REVIEW_QUEUE_FEEDBACK_ENABLED" in r.message for r in caplog.records)


def test_warn_silent_when_no_pending_labels(caplog: pytest.LogCaptureFixture) -> None:
    """warn_if_unprocessed_labels_exist must not log when count == 0."""
    count_row = _row(n=0)
    conn = MagicMock()
    engine = MagicMock()
    engine.connect.return_value.__enter__ = lambda self: conn
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    result = MagicMock()
    result.fetchone.return_value = count_row
    conn.execute.return_value = result

    import logging

    with caplog.at_level(logging.WARNING, logger="denoiser_label_review_queue"):
        count = warn_if_unprocessed_labels_exist(engine)

    assert count == 0
    assert not any("REVIEW_QUEUE_FEEDBACK_ENABLED" in r.message for r in caplog.records)
