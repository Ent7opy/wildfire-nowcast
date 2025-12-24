from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from api.fires import repo


def test_list_fire_detections_bbox_time_filters_noise_by_default(monkeypatch):
    """Ensure the is_noise predicate is added by default."""
    mock_engine = MagicMock()
    monkeypatch.setattr(repo, "get_engine", lambda: mock_engine)

    # Capture the statement executed
    executed_stmt = None

    def mock_begin():
        nonlocal executed_stmt
        context = MagicMock()

        def execute(stmt, params):
            nonlocal executed_stmt
            executed_stmt = stmt
            return MagicMock()

        context.execute = execute
        return context

    mock_engine.begin.return_value.__enter__.side_effect = mock_begin

    repo.list_fire_detections_bbox_time(
        bbox=(0, 0, 1, 1),
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    assert executed_stmt is not None
    assert "is_noise IS NOT TRUE" in str(executed_stmt)


def test_list_fire_detections_bbox_time_includes_noise_when_requested(monkeypatch):
    """Ensure the is_noise predicate is omitted when include_noise=True."""
    mock_engine = MagicMock()
    monkeypatch.setattr(repo, "get_engine", lambda: mock_engine)

    executed_stmt = None

    def mock_begin():
        nonlocal executed_stmt
        context = MagicMock()

        def execute(stmt, params):
            nonlocal executed_stmt
            executed_stmt = stmt
            return MagicMock()

        context.execute = execute
        return context

    mock_engine.begin.return_value.__enter__.side_effect = mock_begin

    repo.list_fire_detections_bbox_time(
        bbox=(0, 0, 1, 1),
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        include_noise=True,
    )

    assert executed_stmt is not None
    assert "is_noise IS NOT TRUE" not in str(executed_stmt)

