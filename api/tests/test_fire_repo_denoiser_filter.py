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


def test_list_fire_fronts_bbox_time_uses_authoritative_event_links(monkeypatch):
    """Ensure fronts query is event-linked and applies score/review predicates."""
    mock_engine = MagicMock()
    monkeypatch.setattr(repo, "get_engine", lambda: mock_engine)

    executed_stmt = None
    executed_params = None

    def mock_begin():
        nonlocal executed_stmt, executed_params
        context = MagicMock()

        def execute(stmt, params=None):
            nonlocal executed_stmt, executed_params
            stmt_text = str(stmt)
            if "SET LOCAL statement_timeout" in stmt_text:
                return MagicMock()
            executed_stmt = stmt
            executed_params = params
            rows = MagicMock()
            rows.mappings.return_value.all.return_value = []
            return rows

        context.execute = execute
        return context

    mock_engine.begin.return_value.__enter__.side_effect = mock_begin

    repo.list_fire_fronts_bbox_time(
        bbox=(0, 0, 1, 1),
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        min_event_score=0.6,
        include_review_required=False,
        limit=400,
    )

    assert executed_stmt is not None
    sql = str(executed_stmt)
    assert "WITH candidate_events AS" in sql
    assert "FROM fire_event_memberships fem" in sql
    assert "fem.front_id IS NOT NULL" in sql
    assert "fe.review_required IS NOT TRUE" in sql
    assert "fe.event_score >= :min_event_score" in sql
    assert "ST_Intersects(ff.geom, ST_MakeEnvelope" in sql
    assert "ff.geom_source" in sql
    assert "ff.geom_method" in sql
    assert "ff.geom_quality" in sql
    assert "ff.authority_profile" in sql
    assert "ff.authoritative_perimeter_id" in sql
    assert executed_params is not None
    assert executed_params["min_event_score"] == 0.6
    # limit is sent as limit+1 to detect has_more; effective_limit caps fronts at 800
    assert executed_params["limit"] == 401


def test_list_fire_events_bbox_time_selects_geom_provenance_fields(monkeypatch):
    """Ensure events query returns geometry provenance fields for API consumers."""
    mock_engine = MagicMock()
    monkeypatch.setattr(repo, "get_engine", lambda: mock_engine)

    executed_stmt = None
    executed_params = None

    def mock_begin():
        nonlocal executed_stmt, executed_params
        context = MagicMock()

        def execute(stmt, params=None):
            nonlocal executed_stmt, executed_params
            stmt_text = str(stmt)
            if "SET LOCAL statement_timeout" in stmt_text:
                return MagicMock()
            executed_stmt = stmt
            executed_params = params
            rows = MagicMock()
            rows.mappings.return_value.all.return_value = []
            return rows

        context.execute = execute
        return context

    mock_engine.begin.return_value.__enter__.side_effect = mock_begin

    repo.list_fire_events_bbox_time(
        bbox=(0, 0, 1, 1),
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        min_event_score=0.5,
        include_review_required=False,
        limit=250,
    )

    assert executed_stmt is not None
    sql = str(executed_stmt)
    assert "geom_source" in sql
    assert "geom_method" in sql
    assert "geom_quality" in sql
    assert "authority_profile" in sql
    assert "authoritative_perimeter_id" in sql
    assert executed_params is not None
    assert executed_params["min_event_score"] == 0.5
    # limit is sent as limit+1 to detect has_more
    assert executed_params["limit"] == 251
