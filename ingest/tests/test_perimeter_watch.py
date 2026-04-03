"""Tests for perimeter_breach_watch and perimeter_growth_watch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ingest.perimeter_breach_watch import (
    _cluster_detections,
    check_perimeter_breach,
    run_perimeter_breach_checks,
)
from ingest.perimeter_growth_watch import (
    check_perimeter_growth,
    run_perimeter_growth_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PERIMETER = {
    "id": 1,
    "source_id": "wfigs-abc123",
    "fire_name": "Oak Fire",
    "geom": "MULTIPOLYGON(((-119.5 37.5, -119.0 37.5, -119.0 38.0, -119.5 38.0, -119.5 37.5)))",
    "source": "nifc_wfigs",
    "acres": 5000.0,
}


def _make_detection(lat: float, lon: float, score: float = 0.85) -> SimpleNamespace:
    """Return a fake DB row with the columns expected by check_perimeter_breach."""
    return SimpleNamespace(lat=lat, lon=lon, denoised_score=score, dist_deg=0.02)


def _make_session(rows: list) -> MagicMock:
    """Return a mock SQLAlchemy session whose execute().fetchall() returns *rows*."""
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = rows
    return session


# ---------------------------------------------------------------------------
# _cluster_detections unit tests
# ---------------------------------------------------------------------------


def test_cluster_groups_nearby_points():
    detections = [
        {"lat": 37.6, "lon": -119.2, "denoised_score": 0.8, "dist_deg": 0.02},
        {"lat": 37.601, "lon": -119.2, "denoised_score": 0.9, "dist_deg": 0.02},
        {"lat": 37.602, "lon": -119.2, "denoised_score": 0.85, "dist_deg": 0.02},
    ]
    clusters = _cluster_detections(detections, radius_deg=0.01)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_separates_distant_points():
    detections = [
        {"lat": 37.6, "lon": -119.2, "denoised_score": 0.8, "dist_deg": 0.02},
        {"lat": 38.5, "lon": -120.5, "denoised_score": 0.9, "dist_deg": 0.02},
    ]
    clusters = _cluster_detections(detections, radius_deg=0.01)
    assert len(clusters) == 2


# ---------------------------------------------------------------------------
# Feature 3: check_perimeter_breach
# ---------------------------------------------------------------------------


def _outside_rows(n: int, score: float = 0.85) -> list[SimpleNamespace]:
    """Return *n* detection rows clearly outside the perimeter (large dist_deg)."""
    base_lat = 39.0  # well north of the dummy perimeter polygon
    return [
        _make_detection(lat=base_lat + i * 0.001, lon=-119.3, score=score)
        for i in range(n)
    ]


class TestCheckPerimeterBreach:
    def test_cluster_of_3_triggers_notify(self):
        """Cluster of 3+ qualifying detections → notify called with severity=critical."""
        rows = _outside_rows(3)
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_breach_watch.notify", mock_notify):
            result = check_perimeter_breach(_PERIMETER, session)

        mock_notify.assert_called_once()
        call_args = mock_notify.call_args
        # First positional arg is event_type
        event_type = call_args.args[0]
        assert event_type == f"perimeter_breach:{_PERIMETER['source_id']}"
        assert call_args.kwargs["severity"] == "critical"
        assert call_args.kwargs["detection_count"] == 3
        assert call_args.kwargs["denoised_score"] == pytest.approx(0.85, abs=1e-3)
        assert result is not None
        assert result["source_id"] == _PERIMETER["source_id"]

    def test_cluster_of_5_triggers_notify(self):
        """Cluster of 5 detections also triggers — validate count in body."""
        rows = _outside_rows(5, score=0.9)
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_breach_watch.notify", mock_notify):
            result = check_perimeter_breach(_PERIMETER, session)

        mock_notify.assert_called_once()
        assert result is not None
        assert result["detection_count"] == 5

    def test_cluster_of_2_does_not_trigger_notify(self):
        """Cluster of only 2 detections → notify NOT called (below min cluster size)."""
        rows = _outside_rows(2)
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_breach_watch.notify", mock_notify):
            result = check_perimeter_breach(_PERIMETER, session)

        mock_notify.assert_not_called()
        assert result is None

    def test_no_detections_returns_none(self):
        """Zero candidate detections from DB → no notify, returns None."""
        session = _make_session([])
        mock_notify = MagicMock()

        with patch("ingest.perimeter_breach_watch.notify", mock_notify):
            result = check_perimeter_breach(_PERIMETER, session)

        mock_notify.assert_not_called()
        assert result is None

    def test_detections_inside_perimeter_excluded(self):
        """If the SQL filter (NOT ST_Within) means no rows returned → no alert.

        We simulate this by having the session return an empty list — the
        PostGIS WHERE clause would have excluded detections inside the
        perimeter before they reach Python.
        """
        # The spatial filter is in SQL; we test the Python path that receives
        # zero rows after the inside-perimeter filter.
        session = _make_session([])
        mock_notify = MagicMock()

        with patch("ingest.perimeter_breach_watch.notify", mock_notify):
            result = check_perimeter_breach(_PERIMETER, session)

        mock_notify.assert_not_called()
        assert result is None

    def test_noise_detections_excluded(self):
        """Detections flagged as noise never reach Python (excluded in SQL).

        We simulate the SQL filter returning empty — as noise rows have
        is_noise=True so they wouldn't appear in the result set.
        """
        session = _make_session([])
        mock_notify = MagicMock()

        with patch("ingest.perimeter_breach_watch.notify", mock_notify):
            result = check_perimeter_breach(_PERIMETER, session)

        mock_notify.assert_not_called()
        assert result is None

    def test_edge_detections_within_500m_excluded(self):
        """Detections within 500 m of boundary are filtered out by SQL (dist < 0.004°).

        Simulate with empty return — the ST_Distance >= 0.004 predicate would
        exclude them.
        """
        session = _make_session([])
        mock_notify = MagicMock()

        with patch("ingest.perimeter_breach_watch.notify", mock_notify):
            result = check_perimeter_breach(_PERIMETER, session)

        mock_notify.assert_not_called()
        assert result is None

    def test_exception_returns_none(self):
        """Any unexpected exception is caught and returns None."""
        session = MagicMock()
        session.execute.side_effect = RuntimeError("DB error")
        mock_notify = MagicMock()

        with patch("ingest.perimeter_breach_watch.notify", mock_notify):
            result = check_perimeter_breach(_PERIMETER, session)

        mock_notify.assert_not_called()
        assert result is None

    def test_notify_receives_required_context_fields(self):
        """notify() must include denoised_score and aoi_id per contract."""
        rows = _outside_rows(3, score=0.92)
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_breach_watch.notify", mock_notify):
            check_perimeter_breach(_PERIMETER, session)

        kwargs = mock_notify.call_args.kwargs
        assert "denoised_score" in kwargs
        assert "aoi_id" in kwargs
        assert kwargs["aoi_id"] is None

    def test_two_separate_clusters_fires_on_first_qualifying(self):
        """When two spatially separate clusters exist and first qualifies, notify is called once."""
        # Cluster A: 4 detections near lat=39.0
        # Cluster B: 1 detection far away — below threshold alone
        cluster_a = [
            SimpleNamespace(lat=39.0 + i * 0.001, lon=-119.3, denoised_score=0.8, dist_deg=0.02)
            for i in range(4)
        ]
        cluster_b = [
            SimpleNamespace(lat=40.5, lon=-120.0, denoised_score=0.8, dist_deg=0.03)
        ]
        session = _make_session(cluster_a + cluster_b)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_breach_watch.notify", mock_notify):
            result = check_perimeter_breach(_PERIMETER, session)

        mock_notify.assert_called_once()
        assert result is not None


# ---------------------------------------------------------------------------
# run_perimeter_breach_checks
# ---------------------------------------------------------------------------


class TestRunPerimeterBreachChecks:
    def _active_perimeter_row(self, source_id: str = "wfigs-001") -> SimpleNamespace:
        return SimpleNamespace(
            id=1,
            source_id=source_id,
            fire_name="Test Fire",
            geom="MULTIPOLYGON(((-119.5 37.5, -119.0 37.5, -119.0 38.0, -119.5 38.0, -119.5 37.5)))",
            source="nifc_wfigs",
            acres=5000.0,
        )

    def test_no_active_perimeters_returns_empty(self):
        session = MagicMock()
        session.execute.return_value.fetchall.return_value = []

        result = run_perimeter_breach_checks(session)

        assert result == []

    def test_calls_check_for_each_active_perimeter(self):
        rows = [
            self._active_perimeter_row("src-1"),
            self._active_perimeter_row("src-2"),
        ]
        session = MagicMock()
        session.execute.return_value.fetchall.return_value = rows

        with patch(
            "ingest.perimeter_breach_watch.check_perimeter_breach",
            side_effect=[{"source_id": "src-1"}, None],
        ) as mock_check:
            result = run_perimeter_breach_checks(session)

        assert mock_check.call_count == 2
        assert len(result) == 1
        assert result[0]["source_id"] == "src-1"

    def test_exception_returns_empty(self):
        session = MagicMock()
        session.execute.side_effect = RuntimeError("DB error")

        result = run_perimeter_breach_checks(session)

        assert result == []


# ---------------------------------------------------------------------------
# Feature 6: check_perimeter_growth
# ---------------------------------------------------------------------------


def _make_growth_row(
    fire_name: str,
    current_acres: float,
    prev_acres: float,
    source_id: str = "wfigs-g1",
) -> SimpleNamespace:
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        fire_name=fire_name,
        current_source_id=source_id,
        current_acres=current_acres,
        current_created_at=now,
        prev_acres=prev_acres,
        prev_created_at=now - timedelta(hours=6),
    )


class TestCheckPerimeterGrowth:
    def test_growth_30pct_200acres_is_critical(self):
        """30% growth with 200 acres absolute → critical severity."""
        rows = [_make_growth_row("Elk Fire", current_acres=1000.0, prev_acres=800.0)]  # 25%
        # 25% is exactly the critical threshold — use 30% to be clearly above
        rows = [_make_growth_row("Elk Fire", current_acres=1040.0, prev_acres=800.0)]  # 30%
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            result = check_perimeter_growth("nifc_wfigs", session)

        mock_notify.assert_called_once()
        kwargs = mock_notify.call_args.kwargs
        assert kwargs["severity"] == "critical"
        assert len(result) == 1
        assert result[0]["severity"] == "critical"

    def test_growth_exactly_25pct_is_critical(self):
        """Exactly 25% growth → critical (boundary case)."""
        rows = [_make_growth_row("Cedar Fire", current_acres=1000.0, prev_acres=800.0)]
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            result = check_perimeter_growth("nifc_wfigs", session)

        mock_notify.assert_called_once()
        assert result[0]["severity"] == "critical"

    def test_growth_12pct_150acres_is_warning(self):
        """12% growth with 150 acres absolute → warning severity."""
        rows = [_make_growth_row("Pine Fire", current_acres=1400.0, prev_acres=1250.0)]  # 12%
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            result = check_perimeter_growth("cwfis", session)

        mock_notify.assert_called_once()
        kwargs = mock_notify.call_args.kwargs
        assert kwargs["severity"] == "warning"
        assert len(result) == 1
        assert result[0]["severity"] == "warning"

    def test_growth_5pct_does_not_trigger(self):
        """5% growth → below 10% threshold → no notify."""
        rows = [_make_growth_row("Ash Fire", current_acres=1050.0, prev_acres=1000.0)]
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            result = check_perimeter_growth("nifc_wfigs", session)

        mock_notify.assert_not_called()
        assert result == []

    def test_growth_15pct_but_only_50_acres_does_not_trigger(self):
        """15% growth but only 50 acres absolute → below 100-acre minimum → no notify."""
        rows = [_make_growth_row("Brush Fire", current_acres=383.0, prev_acres=333.0)]  # ~15%, ~50 acres
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            result = check_perimeter_growth("nifc_wfigs", session)

        mock_notify.assert_not_called()
        assert result == []

    def test_only_one_perimeter_record_no_notify(self):
        """Single perimeter (no previous snapshot) → no notify.

        The SQL JOIN requires rn=1 and rn=2 to both exist; if only one row
        exists the JOIN returns nothing, so check_perimeter_growth receives [].
        """
        session = _make_session([])  # SQL returns empty — JOIN produced no pairs
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            result = check_perimeter_growth("nifc_wfigs", session)

        mock_notify.assert_not_called()
        assert result == []

    def test_no_fires_returns_empty(self):
        """No rows from DB → empty result."""
        session = _make_session([])
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            result = check_perimeter_growth("copernicus_ems", session)

        mock_notify.assert_not_called()
        assert result == []

    def test_event_type_contains_source_and_slug(self):
        """Event type must be perimeter_growth:{source}:{fire_name_slug}."""
        rows = [_make_growth_row("Big Creek Fire", current_acres=1500.0, prev_acres=1000.0)]
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            check_perimeter_growth("nifc_wfigs", session)

        event_type = mock_notify.call_args.args[0]
        assert event_type == "perimeter_growth:nifc_wfigs:big_creek_fire"

    def test_notify_receives_required_context_fields(self):
        """aoi_id must be present per contract; denoised_score must be absent."""
        rows = [_make_growth_row("Ridge Fire", current_acres=1200.0, prev_acres=1000.0)]
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            check_perimeter_growth("nifc_wfigs", session)

        kwargs = mock_notify.call_args.kwargs
        assert "aoi_id" in kwargs
        assert kwargs["aoi_id"] is None
        # denoised_score must NOT be passed for this event (not applicable)
        assert "denoised_score" not in kwargs

    def test_exception_returns_empty_list(self):
        """Unexpected exception is caught and returns []."""
        session = MagicMock()
        session.execute.side_effect = RuntimeError("DB timeout")
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            result = check_perimeter_growth("nifc_wfigs", session)

        mock_notify.assert_not_called()
        assert result == []

    def test_multiple_fires_independent(self):
        """Each fire is evaluated independently; only qualifying ones alert."""
        rows = [
            _make_growth_row("Big Fire", current_acres=2000.0, prev_acres=1000.0),  # 100% → critical
            _make_growth_row("Small Fire", current_acres=105.0, prev_acres=100.0),   # 5% → skip
        ]
        session = _make_session(rows)
        mock_notify = MagicMock()

        with patch("ingest.perimeter_growth_watch.notify", mock_notify):
            result = check_perimeter_growth("nifc_wfigs", session)

        assert mock_notify.call_count == 1
        assert len(result) == 1
        assert result[0]["fire_name"] == "Big Fire"


# ---------------------------------------------------------------------------
# run_perimeter_growth_checks
# ---------------------------------------------------------------------------


class TestRunPerimeterGrowthChecks:
    def test_runs_all_three_sources(self):
        """run_perimeter_growth_checks calls check_perimeter_growth for each source."""
        session = MagicMock()

        with patch(
            "ingest.perimeter_growth_watch.check_perimeter_growth",
            return_value=[],
        ) as mock_check:
            run_perimeter_growth_checks(session)

        sources_checked = [c.args[0] for c in mock_check.call_args_list]
        assert set(sources_checked) == {"nifc_wfigs", "cwfis", "copernicus_ems"}

    def test_aggregates_results_from_all_sources(self):
        """Results from all sources are combined into a single list."""
        session = MagicMock()

        def fake_check(source: str, _session: object) -> list:
            if source == "nifc_wfigs":
                return [{"fire_name": "Oak Fire", "source": source}]
            if source == "cwfis":
                return [{"fire_name": "Maple Fire", "source": source}]
            return []

        with patch("ingest.perimeter_growth_watch.check_perimeter_growth", side_effect=fake_check):
            result = run_perimeter_growth_checks(session)

        assert len(result) == 2
        fire_names = {r["fire_name"] for r in result}
        assert fire_names == {"Oak Fire", "Maple Fire"}
