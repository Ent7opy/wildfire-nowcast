"""Tests for spread_trajectory_watch and weather_threshold_watch."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from ingest.spread_trajectory_watch import (
    _last_alerted_state,
    check_spread_trajectory,
    run_spread_trajectory_checks,
)
from ingest.weather_threshold_watch import (
    check_weather_thresholds,
    run_weather_threshold_checks,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_T0 = datetime(2026, 3, 27, 6, 0, 0, tzinfo=timezone.utc)
_T1 = _T0 + timedelta(hours=6)
_T2 = _T1 + timedelta(hours=6)


def _make_aoi(**overrides: object) -> dict:
    base: dict = {
        "id": uuid4(),
        "name": "Test AOI",
        "bbox": {
            "type": "Polygon",
            "coordinates": [[[-120.0, 37.0], [-119.0, 37.0], [-119.0, 38.0], [-120.0, 38.0], [-120.0, 37.0]]],
        },
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4: Spread Trajectory Deviation
# ─────────────────────────────────────────────────────────────────────────────

def _make_spread_session(
    *,
    run_rows: list,
    centroid_rows: dict[int, tuple[float, float] | None],
    older_run_row: tuple | None = None,
    dist_deg: float = 0.1,
    horizons: list[int] | None = None,
) -> MagicMock:
    """Build a mock SQLAlchemy session for spread trajectory tests.

    centroid_rows maps run_id -> (cx, cy) or None (applies to every horizon).
    horizons controls which horizons get real centroid data.  Horizons NOT in
    this list return None for the curr centroid (simulating a missing contour).

    New SQL call order (matches refactored implementation):
      1.  runs query        → fetchall → run_rows
      2.  older run query   → fetchone → older_run_row
      For each horizon in _REQUIRED_HORIZONS:
        If horizon IS in horizons:
          3.  centroid for curr_run_id  → fetchone
          4.  centroid for prev_run_id  → fetchone
          5.  ST_Distance               → fetchone
          If older_run_row is not None:
            6.  centroid for older_run_id → fetchone
        Else (missing horizon):
          3.  centroid for curr_run_id  → fetchone (None) → code skips via continue
    """
    from ingest.spread_trajectory_watch import _REQUIRED_HORIZONS

    if horizons is None:
        horizons = list(_REQUIRED_HORIZONS)

    session = MagicMock()
    call_results: list = []

    # 1. runs
    runs_mock = MagicMock()
    runs_mock.fetchall.return_value = run_rows
    call_results.append(runs_mock)

    if len(run_rows) >= 2:
        curr_run_id = run_rows[0][0]
        prev_run_id = run_rows[1][0]

        # 2. older run query
        older_mock = MagicMock()
        older_mock.fetchone.return_value = older_run_row
        call_results.append(older_mock)

        older_run_id = older_run_row[0] if older_run_row is not None else None

        for h in _REQUIRED_HORIZONS:
            if h not in horizons:
                # Missing horizon: only the curr centroid query fires (returns None)
                # The code then logs a warning and calls `continue`.
                missing_mock = MagicMock()
                missing_mock.fetchone.return_value = None
                call_results.append(missing_mock)
                continue

            # curr centroid
            curr_cent = centroid_rows.get(curr_run_id)
            curr_mock = MagicMock()
            curr_mock.fetchone.return_value = (
                (curr_cent[0], curr_cent[1]) if curr_cent else None
            )
            call_results.append(curr_mock)

            if curr_cent is None:
                # If curr is None the code also logs warning and continues
                continue

            # prev centroid
            prev_cent = centroid_rows.get(prev_run_id)
            prev_mock = MagicMock()
            prev_mock.fetchone.return_value = (
                (prev_cent[0], prev_cent[1]) if prev_cent else None
            )
            call_results.append(prev_mock)

            if prev_cent is None:
                continue

            # ST_Distance
            dist_mock = MagicMock()
            dist_mock.fetchone.return_value = (dist_deg,)
            call_results.append(dist_mock)

            # older centroid (only if older run exists)
            if older_run_id is not None:
                older_cent = centroid_rows.get(older_run_id)
                older_cent_mock = MagicMock()
                older_cent_mock.fetchone.return_value = (
                    (older_cent[0], older_cent[1]) if older_cent else None
                )
                call_results.append(older_cent_mock)

    session.execute.side_effect = call_results
    return session


def _bearing(from_lon: float, from_lat: float, to_lon: float, to_lat: float) -> float:
    return math.degrees(math.atan2(to_lon - from_lon, to_lat - from_lat)) % 360.0


class TestSpreadTrajectory:
    """Feature 4: spread trajectory deviation."""

    def setup_method(self) -> None:
        """Clear module-level gate state before every test."""
        _last_alerted_state.clear()

    def test_direction_40deg_is_warning(self) -> None:
        """Direction rotates 40° → notify called with severity='warning'."""
        aoi = _make_aoi()
        aoi_id = aoi["id"]

        # We need the older→prev bearing and the prev→curr bearing to differ by ~40°.
        # Place points such that the angular delta is exactly 40°.
        # older centroid = (0, 0) lon/lat
        # prev centroid  = (0, 0.1) — heading due north (0°)
        # curr centroid  = (0.1*sin40°, 0.1*cos40°) — heading 40° from north
        angle_rad = math.radians(40.0)
        older_cent = (-119.5, 37.4)
        prev_cent = (-119.5, 37.5)  # due north from older
        curr_cent = (
            -119.5 + 0.1 * math.sin(angle_rad),
            37.5 + 0.1 * math.cos(angle_rad),
        )

        run_rows = [(1, _T2), (2, _T1)]
        older_run_row = (3, _T0)
        centroids = {1: curr_cent, 2: prev_cent, 3: older_cent}

        session = _make_spread_session(
            run_rows=run_rows,
            centroid_rows=centroids,
            older_run_row=older_run_row,
            dist_deg=0.1,
        )

        with patch("ingest.spread_trajectory_watch.notify") as mock_notify:
            result = check_spread_trajectory(aoi, session)

        assert result is not None
        assert len(result) > 0
        mock_notify.assert_called()
        # event_type includes aoi_id and horizon
        event_types = [c.args[0] for c in mock_notify.call_args_list]
        assert any(str(aoi_id) in et for et in event_types)
        call_kw = mock_notify.call_args_list[0].kwargs
        assert call_kw["severity"] == "warning"
        assert call_kw["aoi_id"] == str(aoi_id)

    def test_direction_50deg_is_critical(self) -> None:
        """Direction rotates 50° → notify called with severity='critical'."""
        aoi = _make_aoi()

        angle_rad = math.radians(50.0)
        older_cent = (-119.5, 37.4)
        prev_cent = (-119.5, 37.5)
        curr_cent = (
            -119.5 + 0.1 * math.sin(angle_rad),
            37.5 + 0.1 * math.cos(angle_rad),
        )

        run_rows = [(1, _T2), (2, _T1)]
        older_run_row = (3, _T0)
        centroids = {1: curr_cent, 2: prev_cent, 3: older_cent}

        session = _make_spread_session(
            run_rows=run_rows,
            centroid_rows=centroids,
            older_run_row=older_run_row,
            dist_deg=0.1,
        )

        with patch("ingest.spread_trajectory_watch.notify") as mock_notify:
            result = check_spread_trajectory(aoi, session)

        assert result is not None
        assert len(result) > 0
        mock_notify.assert_called()
        call_kw = mock_notify.call_args_list[0].kwargs
        assert call_kw["severity"] == "critical"

    def test_speed_increase_60pct_is_warning(self) -> None:
        """Speed increases 60% → notify called with severity='warning'."""
        aoi = _make_aoi()

        # Keep direction same (0° change) but vary distances to create speed increase.
        # older→prev: 0.1 deg in 6h → speed_prev = 0.1*111/6 = 1.85 km/h
        # prev→curr:  0.16 deg in 6h → speed_curr = 0.16*111/6 = 2.96 km/h → +60%
        older_cent = (-119.5, 37.3)
        prev_cent = (-119.5, 37.4)   # 0.1 deg north of older
        curr_cent = (-119.5, 37.56)  # 0.16 deg north of prev (same direction)

        run_rows = [(1, _T2), (2, _T1)]
        older_run_row = (3, _T0)
        centroids = {1: curr_cent, 2: prev_cent, 3: older_cent}

        # ST_Distance for prev→curr: 0.16 deg
        session = _make_spread_session(
            run_rows=run_rows,
            centroid_rows=centroids,
            older_run_row=older_run_row,
            dist_deg=0.16,
        )

        with patch("ingest.spread_trajectory_watch.notify") as mock_notify:
            result = check_spread_trajectory(aoi, session)

        assert result is not None
        assert len(result) > 0
        mock_notify.assert_called()
        call_kw = mock_notify.call_args_list[0].kwargs
        assert call_kw["severity"] == "warning"

    def test_only_one_run_returns_none(self) -> None:
        """Only one forecast run → notify NOT called, returns None."""
        aoi = _make_aoi()
        run_rows = [(1, _T2)]  # only one

        session = MagicMock()
        runs_mock = MagicMock()
        runs_mock.fetchall.return_value = run_rows
        session.execute.return_value = runs_mock

        with patch("ingest.spread_trajectory_watch.notify") as mock_notify:
            result = check_spread_trajectory(aoi, session)

        assert result is None
        mock_notify.assert_not_called()

    def test_no_significant_change_does_not_notify(self) -> None:
        """Direction < 30° and speed change < 50% → notify NOT called, returns []."""
        aoi = _make_aoi()

        # 10° direction change, same speed
        angle_rad = math.radians(10.0)
        older_cent = (-119.5, 37.4)
        prev_cent = (-119.5, 37.5)
        curr_cent = (
            -119.5 + 0.1 * math.sin(angle_rad),
            37.5 + 0.1 * math.cos(angle_rad),
        )

        run_rows = [(1, _T2), (2, _T1)]
        older_run_row = (3, _T0)
        centroids = {1: curr_cent, 2: prev_cent, 3: older_cent}

        session = _make_spread_session(
            run_rows=run_rows,
            centroid_rows=centroids,
            older_run_row=older_run_row,
            dist_deg=0.1,  # same distance as older→prev (0.1 deg)
        )

        with patch("ingest.spread_trajectory_watch.notify") as mock_notify:
            result = check_spread_trajectory(aoi, session)

        # Runs exist but no horizon triggered → empty list, not None
        assert result == []
        mock_notify.assert_not_called()

    def test_run_spread_trajectory_checks_aggregates(self) -> None:
        """run_spread_trajectory_checks collects results from each AOI."""
        aoi1 = _make_aoi(name="AOI 1")
        aoi2 = _make_aoi(name="AOI 2")

        with patch(
            "ingest.spread_trajectory_watch.check_spread_trajectory"
        ) as mock_check:
            # check_spread_trajectory now returns list[dict] | None
            mock_check.side_effect = [[{"aoi_name": "AOI 1"}], None]
            session = MagicMock()
            results = run_spread_trajectory_checks([aoi1, aoi2], session)

        assert len(results) == 1
        assert results[0]["aoi_name"] == "AOI 1"

    # ── Transition gate tests (Fix 1) ─────────────────────────────────────────

    def test_trajectory_suppresses_repeat_with_same_bearing(self) -> None:
        """First alert at bearing 45°, second call with same bearing → suppressed."""
        aoi = _make_aoi()
        aoi_id = str(aoi["id"])

        # Pre-populate gate state as if a prior alert fired at bearing 45° for h=12
        _last_alerted_state[f"{aoi_id}:12"] = {
            "bearing": 45.0,
            "run_id": 99,
            "severity": "warning",
        }

        # Construct a 40° direction change (triggers 'warning') landing at ~45° bearing
        angle_rad = math.radians(45.0)
        older_cent = (-119.5, 37.4)
        prev_cent = (-119.5, 37.5)
        curr_cent = (
            -119.5 + 0.1 * math.sin(angle_rad),
            37.5 + 0.1 * math.cos(angle_rad),
        )

        run_rows = [(1, _T2), (2, _T1)]
        older_run_row = (3, _T0)
        centroids = {1: curr_cent, 2: prev_cent, 3: older_cent}

        # Only h=12 is served with real data; 24/48/72 missing (return None)
        session = _make_spread_session(
            run_rows=run_rows,
            centroid_rows=centroids,
            older_run_row=older_run_row,
            dist_deg=0.1,
            horizons=[12],
        )

        with patch("ingest.spread_trajectory_watch.notify") as mock_notify:
            result = check_spread_trajectory(aoi, session)

        # The 12h horizon is suppressed (same bearing, same severity) → no notify
        mock_notify.assert_not_called()
        assert result == []

    def test_trajectory_re_alerts_on_additional_shift(self) -> None:
        """First alert at 45°, second call rotated to 90° (45° additional) → re-alerts."""
        aoi = _make_aoi()
        aoi_id = str(aoi["id"])

        # Prior alert was at bearing 45°
        _last_alerted_state[f"{aoi_id}:12"] = {
            "bearing": 45.0,
            "run_id": 99,
            "severity": "warning",
        }

        # Now the trajectory has rotated another ~45° (from 45° to ~90°, due east).
        # older→prev leg points NE (bearing ~45°)
        # prev→curr leg points E (bearing ~90°)
        # direction_change ≈ 45° > _DIR_CRIT_DEG (45°) → critical
        # additional_shift from prior bearing (45°) to current bearing (~90°) = ~45° ≥ 30°
        older_cent = (-119.5, 37.4)
        prev_cent = (
            -119.5 + 0.1 * math.sin(math.radians(45.0)),
            37.4 + 0.1 * math.cos(math.radians(45.0)),
        )
        curr_cent = (
            prev_cent[0] + 0.1,  # due east from prev
            prev_cent[1],
        )

        run_rows = [(1, _T2), (2, _T1)]
        older_run_row = (3, _T0)
        centroids = {1: curr_cent, 2: prev_cent, 3: older_cent}

        session = _make_spread_session(
            run_rows=run_rows,
            centroid_rows=centroids,
            older_run_row=older_run_row,
            dist_deg=0.1,
            horizons=[12],
        )

        with patch("ingest.spread_trajectory_watch.notify") as mock_notify:
            result = check_spread_trajectory(aoi, session)

        # additional_shift ≥ _DIR_WARN_DEG → should re-alert
        mock_notify.assert_called_once()
        assert result is not None
        assert len(result) == 1

    def test_trajectory_multi_horizon_12h_triggers_24h_does_not(self) -> None:
        """12h contour exceeds threshold, 24h absent → only one notify() call (for 12h)."""
        aoi = _make_aoi()

        # 40° direction change — exceeds warning threshold
        angle_rad = math.radians(40.0)
        older_cent = (-119.5, 37.4)
        prev_cent = (-119.5, 37.5)
        curr_cent = (
            -119.5 + 0.1 * math.sin(angle_rad),
            37.5 + 0.1 * math.cos(angle_rad),
        )

        run_rows = [(1, _T2), (2, _T1)]
        older_run_row = (3, _T0)
        centroids = {1: curr_cent, 2: prev_cent, 3: older_cent}

        # Only h=12 gets real data; 24/48/72 are missing
        session = _make_spread_session(
            run_rows=run_rows,
            centroid_rows=centroids,
            older_run_row=older_run_row,
            dist_deg=0.1,
            horizons=[12],
        )

        with patch("ingest.spread_trajectory_watch.notify") as mock_notify:
            result = check_spread_trajectory(aoi, session)

        # Only h=12 should have triggered → exactly one notify call
        mock_notify.assert_called_once()
        assert result is not None
        assert len(result) == 1
        assert result[0]["horizon_hours"] == 12

    def test_trajectory_horizon_missing_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """48h contour absent → WARNING logged, 48h skipped, other horizons proceed."""
        aoi = _make_aoi()

        # 40° direction change at h=12
        angle_rad = math.radians(40.0)
        older_cent = (-119.5, 37.4)
        prev_cent = (-119.5, 37.5)
        curr_cent = (
            -119.5 + 0.1 * math.sin(angle_rad),
            37.5 + 0.1 * math.cos(angle_rad),
        )

        run_rows = [(1, _T2), (2, _T1)]
        older_run_row = (3, _T0)
        centroids = {1: curr_cent, 2: prev_cent, 3: older_cent}

        # Serve only h=12; 24/48/72 will be missing (returns None → WARNING logged)
        session = _make_spread_session(
            run_rows=run_rows,
            centroid_rows=centroids,
            older_run_row=older_run_row,
            dist_deg=0.1,
            horizons=[12],
        )

        with patch("ingest.spread_trajectory_watch.notify"):
            with caplog.at_level(logging.WARNING, logger="ingest.spread_trajectory_watch"):
                result = check_spread_trajectory(aoi, session)

        # At least one WARNING about missing ICS outlook must appear
        warning_msgs = [
            r.message for r in caplog.records
            if r.levelno == logging.WARNING
            and "ICS" in r.message
            and "outlook unavailable" in r.message
        ]
        assert warning_msgs, (
            "Expected at least one WARNING about missing ICS outlook. "
            f"Got records: {[r.message for r in caplog.records]}"
        )
        # 48h should be explicitly mentioned
        assert any("48h" in m for m in warning_msgs)
        # h=12 still ran; result is not None and has an entry
        assert result is not None


    def test_gate_not_advanced_when_notify_suppressed(self) -> None:
        """Gate state must NOT update when notify() returns False (burst/rate-limit)."""
        aoi = _make_aoi()
        aoi_id = str(aoi["id"])

        # 40° direction change triggers warning
        angle_rad = math.radians(40.0)
        older_cent = (-119.5, 37.4)
        prev_cent = (-119.5, 37.5)
        curr_cent = (
            -119.5 + 0.1 * math.sin(angle_rad),
            37.5 + 0.1 * math.cos(angle_rad),
        )
        run_rows = [(1, _T2), (2, _T1)]
        older_run_row = (3, _T0)
        centroids = {1: curr_cent, 2: prev_cent, 3: older_cent}
        session = _make_spread_session(
            run_rows=run_rows,
            centroid_rows=centroids,
            older_run_row=older_run_row,
            dist_deg=0.1,
            horizons=[12],
        )

        gate_key = f"{aoi_id}:12"
        assert gate_key not in _last_alerted_state

        # Simulate burst cap suppression: notify() returns False
        with patch("ingest.spread_trajectory_watch.notify", return_value=False):
            result = check_spread_trajectory(aoi, session)

        # Gate state must remain empty — notify was not delivered
        assert gate_key not in _last_alerted_state
        # Result still contains the triggered horizon (detection occurred)
        assert result is not None
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5: Critical Weather Threshold at Fire Location
# ─────────────────────────────────────────────────────────────────────────────

def _make_weather_session(
    *,
    curr_metadata: dict | None,
    prev_metadata: dict | None = None,
    has_curr_run: bool = True,
) -> MagicMock:
    """Build a mock session for weather threshold tests."""
    session = MagicMock()

    if not has_curr_run:
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        session.execute.return_value = result_mock
        return session

    rows: list = []
    if has_curr_run:
        rows.append((101, _T2, curr_metadata))
    if prev_metadata is not None:
        rows.append((100, _T1, prev_metadata))

    result_mock = MagicMock()
    result_mock.fetchall.return_value = rows
    session.execute.return_value = result_mock
    return session


class TestWeatherThresholds:
    """Feature 5: critical weather threshold at fire location."""

    def test_rh20_is_warning(self) -> None:
        """rh2m_min=20 (< 25%) → notify called with severity='warning'."""
        aoi = _make_aoi()
        aoi_id = aoi["id"]

        metadata = {"summary": {"rh2m_min": 20.0, "wind_bearing_deg": 180.0}}
        session = _make_weather_session(curr_metadata=metadata)

        with patch("ingest.weather_threshold_watch.notify") as mock_notify:
            result = check_weather_thresholds(aoi, session)

        assert result is not None
        mock_notify.assert_called_once()
        # event_type is the first positional arg
        assert mock_notify.call_args.args[0] == f"weather_threshold:{aoi_id}"
        call_kw = mock_notify.call_args.kwargs
        assert call_kw["severity"] == "warning"
        assert call_kw["aoi_id"] == str(aoi_id)
        assert call_kw["rh_pct"] == 20.0

    def test_rh12_is_critical(self) -> None:
        """rh2m_min=12 (< 15%) → notify called with severity='critical'."""
        aoi = _make_aoi()

        metadata = {"summary": {"rh2m_min": 12.0, "wind_bearing_deg": 90.0}}
        session = _make_weather_session(curr_metadata=metadata)

        with patch("ingest.weather_threshold_watch.notify") as mock_notify:
            result = check_weather_thresholds(aoi, session)

        assert result is not None
        mock_notify.assert_called_once()
        call_kw = mock_notify.call_args.kwargs
        assert call_kw["severity"] == "critical"

    def test_wind_shift_35deg_is_warning(self) -> None:
        """Wind shift 35° → notify called with severity='warning'."""
        aoi = _make_aoi()

        curr_metadata = {"summary": {"rh2m_min": 40.0, "wind_bearing_deg": 215.0}}
        prev_metadata = {"summary": {"rh2m_min": 40.0, "wind_bearing_deg": 180.0}}
        session = _make_weather_session(
            curr_metadata=curr_metadata,
            prev_metadata=prev_metadata,
        )

        with patch("ingest.weather_threshold_watch.notify") as mock_notify:
            result = check_weather_thresholds(aoi, session)

        assert result is not None
        mock_notify.assert_called_once()
        call_kw = mock_notify.call_args.kwargs
        assert call_kw["severity"] == "warning"
        conditions = call_kw["conditions"]
        assert any("wind_shift" in c for c in conditions)

    def test_no_summary_key_skips_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """weather_runs metadata has no 'summary' key → notify NOT called, WARNING logged."""
        aoi = _make_aoi()

        # metadata present but no 'summary' key
        metadata = {"variables": ["u10", "v10", "rh2m"]}
        session = _make_weather_session(curr_metadata=metadata)

        with patch("ingest.weather_threshold_watch.notify") as mock_notify:
            with caplog.at_level(logging.WARNING, logger="ingest.weather_threshold_watch"):
                result = check_weather_thresholds(aoi, session)

        assert result is None
        mock_notify.assert_not_called()
        assert any("weather summary not available" in r.message for r in caplog.records)

    def test_rh30_wind_shift_20_does_not_notify(self) -> None:
        """RH=30% and wind shift 20° → no threshold exceeded, notify NOT called."""
        aoi = _make_aoi()

        curr_metadata = {"summary": {"rh2m_min": 30.0, "wind_bearing_deg": 200.0}}
        prev_metadata = {"summary": {"rh2m_min": 32.0, "wind_bearing_deg": 180.0}}
        session = _make_weather_session(
            curr_metadata=curr_metadata,
            prev_metadata=prev_metadata,
        )

        with patch("ingest.weather_threshold_watch.notify") as mock_notify:
            result = check_weather_thresholds(aoi, session)

        assert result is None
        mock_notify.assert_not_called()

    def test_run_weather_threshold_checks_aggregates(self) -> None:
        """run_weather_threshold_checks collects non-None results."""
        aoi1 = _make_aoi(name="AOI 1")
        aoi2 = _make_aoi(name="AOI 2")

        with patch(
            "ingest.weather_threshold_watch.check_weather_thresholds"
        ) as mock_check:
            mock_check.side_effect = [None, {"aoi_name": "AOI 2"}]
            session = MagicMock()
            results = run_weather_threshold_checks([aoi1, aoi2], session)

        assert len(results) == 1
        assert results[0]["aoi_name"] == "AOI 2"

    def test_no_completed_runs_returns_none(self) -> None:
        """No completed weather runs → returns None, notify NOT called."""
        aoi = _make_aoi()
        session = _make_weather_session(curr_metadata=None, has_curr_run=False)

        with patch("ingest.weather_threshold_watch.notify") as mock_notify:
            result = check_weather_thresholds(aoi, session)

        assert result is None
        mock_notify.assert_not_called()
