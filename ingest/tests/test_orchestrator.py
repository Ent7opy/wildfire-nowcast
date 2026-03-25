import argparse
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from ingest.orchestrator import (
    JOB_DENOISER_DRIFT,
    JOB_INDUSTRIAL,
    JOB_LFMC,
    JOB_LULC,
    JOB_ORDER,
    JOB_WEATHER,
    ScheduledJob,
    _build_industrial_argv,
    _build_weather_argv,
    _run_denoiser_drift,
    _run_lfmc,
    run_once,
    run_scheduler,
    validate_and_reset_watermarks,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class TestOrchestrator(unittest.TestCase):
    def test_build_weather_argv(self):
        args = argparse.Namespace(
            weather_run_time="2026-02-11T12:00:00Z",
            weather_bbox=[-10.0, 30.0, 10.0, 40.0],
            weather_horizon_hours=24,
            weather_step_hours=6,
            weather_include_precip=True,
            weather_patch_mode=True,
        )

        argv = _build_weather_argv(args)

        self.assertEqual(
            argv,
            [
                "--run-time",
                "2026-02-11T12:00:00Z",
                "--bbox",
                "-10.0",
                "30.0",
                "10.0",
                "40.0",
                "--horizon-hours",
                "24",
                "--step-hours",
                "6",
                "--include-precip",
                "--patch-mode",
            ],
        )

    def test_build_industrial_argv(self):
        args = argparse.Namespace(
            industrial_source_profile="eu_eprtr_gold",
            industrial_config="/tmp/industrial_authority_profiles.yaml",
            industrial_start="2025-01-01T00:00:00Z",
            industrial_end="2025-01-31T23:59:59Z",
            industrial_run_id="industrial_run_1",
            industrial_curated_file=["/tmp/eu_part1.csv", "/tmp/eu_part2.csv"],
            industrial_timeout_seconds=60.0,
            industrial_dry_run=True,
        )

        argv = _build_industrial_argv(args)

        self.assertEqual(
            argv,
            [
                "--source-profile",
                "eu_eprtr_gold",
                "--config",
                "/tmp/industrial_authority_profiles.yaml",
                "--start",
                "2025-01-01T00:00:00Z",
                "--end",
                "2025-01-31T23:59:59Z",
                "--run-id",
                "industrial_run_1",
                "--curated-file",
                "/tmp/eu_part1.csv",
                "--curated-file",
                "/tmp/eu_part2.csv",
                "--timeout-seconds",
                "60.0",
                "--dry-run",
            ],
        )

    def test_run_once_continues_when_stop_on_error_disabled(self):
        calls: list[str] = []

        def _fail() -> int:
            calls.append("firms")
            return 1

        def _ok() -> int:
            calls.append("weather")
            return 0

        jobs = [
            ScheduledJob(name="firms", interval_seconds=60.0, runner=_fail),
            ScheduledJob(name="weather", interval_seconds=60.0, runner=_ok),
        ]

        exit_code = run_once(jobs, stop_on_error=False)

        self.assertEqual(1, exit_code)
        self.assertEqual(["firms", "weather"], calls)

    def test_run_once_stops_on_first_error(self):
        calls: list[str] = []

        def _fail() -> int:
            calls.append("firms")
            return 1

        def _ok() -> int:
            calls.append("weather")
            return 0

        jobs = [
            ScheduledJob(name="firms", interval_seconds=60.0, runner=_fail),
            ScheduledJob(name="weather", interval_seconds=60.0, runner=_ok),
        ]

        exit_code = run_once(jobs, stop_on_error=True)

        self.assertEqual(1, exit_code)
        self.assertEqual(["firms"], calls)

    def test_run_scheduler_respects_intervals(self):
        clock = FakeClock()
        calls: list[str] = []

        def _job_a() -> int:
            calls.append("firms")
            return 0

        def _job_b() -> int:
            calls.append("weather")
            return 0

        jobs = [
            ScheduledJob(name="firms", interval_seconds=5.0, runner=_job_a),
            ScheduledJob(name="weather", interval_seconds=10.0, runner=_job_b),
        ]

        def _stop() -> bool:
            return clock.now >= 12.0

        exit_code = run_scheduler(
            jobs,
            poll_seconds=2.0,
            run_on_start=False,
            stop_on_error=False,
            stop_requested=_stop,
            now_fn=clock.monotonic,
            sleep_fn=clock.sleep,
        )

        self.assertEqual(0, exit_code)
        self.assertEqual(["firms", "firms", "weather"], calls)
        self.assertTrue(clock.sleeps)

    def test_run_once_retries_then_succeeds(self):
        calls = {"count": 0}
        sleeps: list[float] = []

        def _flaky() -> int:
            calls["count"] += 1
            return 0 if calls["count"] == 2 else 1

        jobs = [ScheduledJob(name="firms", interval_seconds=60.0, runner=_flaky)]

        exit_code = run_once(
            jobs,
            stop_on_error=True,
            max_retries=2,
            retry_backoff_seconds=3.0,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(0, exit_code)
        self.assertEqual(2, calls["count"])
        self.assertEqual([3.0], sleeps)

    def test_run_once_does_not_skip_firms_when_freshness_enforced(self):
        calls = {"count": 0}

        def _runner() -> int:
            calls["count"] += 1
            return 0

        jobs = [ScheduledJob(name="firms", interval_seconds=60.0, runner=_runner)]

        def _snapshot():
            return {
                "sources": {
                    "firms": {"state": "fresh"},
                }
            }

        exit_code = run_once(
            jobs,
            stop_on_error=True,
            enforce_freshness=True,
            status_snapshot_fn=_snapshot,
        )

        self.assertEqual(0, exit_code)
        self.assertEqual(1, calls["count"])

    def test_run_once_skips_non_firms_fresh_job_when_enforced(self):
        calls = {"count": 0}

        def _runner() -> int:
            calls["count"] += 1
            return 0

        jobs = [ScheduledJob(name="weather", interval_seconds=60.0, runner=_runner)]

        def _snapshot():
            return {
                "sources": {
                    "weather": {"state": "fresh"},
                }
            }

        exit_code = run_once(
            jobs,
            stop_on_error=True,
            enforce_freshness=True,
            status_snapshot_fn=_snapshot,
        )

        self.assertEqual(0, exit_code)
        self.assertEqual(0, calls["count"])


class TestDriftJob(unittest.TestCase):
    def test_denoiser_drift_in_job_order(self):
        self.assertIn(JOB_DENOISER_DRIFT, JOB_ORDER)
        # Must come after industrial (ingest jobs run before monitoring)
        self.assertGreater(JOB_ORDER.index(JOB_DENOISER_DRIFT), JOB_ORDER.index(JOB_INDUSTRIAL))

    def test_run_denoiser_drift_ok_returns_zero(self):
        summary = {
            "metrics": {
                "psi_score": {"value": 0.05, "severity": "ok"},
                "score_mean_delta": {"value": 0.02, "severity": "ok"},
            }
        }
        args = argparse.Namespace()
        with patch("ingest.denoiser_drift_monitor.monitor_denoiser_drift", return_value=summary):
            code = _run_denoiser_drift(args)
        self.assertEqual(0, code)

    def test_run_denoiser_drift_warn_returns_zero(self):
        summary = {
            "metrics": {
                "psi_score": {"value": 0.25, "severity": "warn"},
                "score_mean_delta": {"value": 0.05, "severity": "ok"},
            }
        }
        args = argparse.Namespace()
        with patch("ingest.denoiser_drift_monitor.monitor_denoiser_drift", return_value=summary):
            code = _run_denoiser_drift(args)
        self.assertEqual(0, code)

    def test_run_denoiser_drift_hard_violation_returns_one_and_logs_blocker(self):
        summary = {
            "metrics": {
                "psi_score": {"value": 0.42, "severity": "hard"},
                "score_mean_delta": {"value": 0.22, "severity": "hard"},
            }
        }
        args = argparse.Namespace()
        with patch("ingest.denoiser_drift_monitor.monitor_denoiser_drift", return_value=summary):
            with self.assertLogs("ingest_orchestrator", level="ERROR") as log_ctx:
                code = _run_denoiser_drift(args)
        self.assertEqual(1, code)
        self.assertTrue(
            any("BLOCKER" in line for line in log_ctx.output),
            f"Expected BLOCKER in logs, got: {log_ctx.output}",
        )

    def test_run_denoiser_drift_hard_violation_does_not_rollback(self):
        """allow_rollback must be False when called from orchestrator."""
        summary = {
            "metrics": {
                "psi_score": {"value": 0.50, "severity": "hard"},
                "score_mean_delta": {"value": 0.25, "severity": "hard"},
            }
        }
        args = argparse.Namespace()
        with patch("ingest.denoiser_drift_monitor.monitor_denoiser_drift", return_value=summary) as mock_monitor:
            with self.assertLogs("ingest_orchestrator", level="ERROR"):
                _run_denoiser_drift(args)
        _call_kwargs = mock_monitor.call_args.kwargs
        self.assertFalse(_call_kwargs.get("allow_rollback"), "orchestrator must never auto-rollback")

    def test_drift_job_failure_surfaces_in_run_once_metrics(self):
        """A hard-violation (exit=1) is tracked as a failure in orchestrator metrics."""
        calls: list[str] = []

        def _drift_hard() -> int:
            calls.append("drift")
            return 1

        jobs = [ScheduledJob(name="denoiser_drift", interval_seconds=60.0, runner=_drift_hard)]
        exit_code = run_once(jobs, stop_on_error=False)
        self.assertEqual(1, exit_code)
        self.assertEqual(["drift"], calls)


class TestWatermarkValidation(unittest.TestCase):
    _NOW = datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc)

    def _wm(self, source: str, area_key: str, ts: datetime | None) -> dict:
        return {"source": source, "area_key": area_key, "last_acq_time_utc": ts}

    def _run(self, watermarks: list) -> tuple[int, list]:
        resets: list[tuple[str, str]] = []
        count = validate_and_reset_watermarks(
            now_utc=self._NOW,
            list_fn=lambda: watermarks,
            reset_fn=lambda s, a: resets.append((s, a)),
        )
        return count, resets

    # --- Corrupt: future timestamp ---

    def test_future_watermark_is_reset(self):
        future_ts = self._NOW + timedelta(hours=2)
        count, resets = self._run([self._wm("VIIRS_SNPP_NRT", "world", future_ts)])
        self.assertEqual(1, count)
        self.assertEqual([("VIIRS_SNPP_NRT", "world")], resets)

    def test_future_watermark_non_nrt_is_also_reset(self):
        """Future timestamps are invalid regardless of source type."""
        future_ts = self._NOW + timedelta(hours=1)
        count, resets = self._run([self._wm("VIIRS_SNPP_SP", "world", future_ts)])
        self.assertEqual(1, count)
        self.assertEqual([("VIIRS_SNPP_SP", "world")], resets)

    def test_within_clock_skew_tolerance_is_not_reset(self):
        """A timestamp just inside the 5-minute grace window must not be reset."""
        almost_future = self._NOW + timedelta(seconds=299)
        count, resets = self._run([self._wm("VIIRS_SNPP_NRT", "world", almost_future)])
        self.assertEqual(0, count)
        self.assertEqual([], resets)

    # --- Corrupt: NRT staleness ---

    def test_nrt_watermark_older_than_30_days_is_reset(self):
        stale_ts = self._NOW - timedelta(days=31)
        count, resets = self._run([self._wm("VIIRS_SNPP_NRT", "world", stale_ts)])
        self.assertEqual(1, count)
        self.assertEqual([("VIIRS_SNPP_NRT", "world")], resets)

    def test_viirs_noaa20_nrt_staleness_is_detected(self):
        stale_ts = self._NOW - timedelta(days=45)
        count, resets = self._run([self._wm("VIIRS_NOAA20_NRT", "-130,30,-60,60", stale_ts)])
        self.assertEqual(1, count)
        self.assertEqual([("VIIRS_NOAA20_NRT", "-130,30,-60,60")], resets)

    def test_non_nrt_watermark_older_than_30_days_is_not_reset(self):
        """Batch/archive sources are not subject to the NRT staleness limit."""
        stale_ts = self._NOW - timedelta(days=31)
        count, resets = self._run([self._wm("VIIRS_SNPP_SP", "world", stale_ts)])
        self.assertEqual(0, count)
        self.assertEqual([], resets)

    def test_nrt_watermark_exactly_30_days_old_is_not_reset(self):
        """Boundary: exactly 30 days is still within the limit."""
        boundary_ts = self._NOW - timedelta(days=30)
        count, resets = self._run([self._wm("VIIRS_SNPP_NRT", "world", boundary_ts)])
        self.assertEqual(0, count)
        self.assertEqual([], resets)

    # --- Valid: should not be touched ---

    def test_null_watermark_is_not_reset(self):
        """NULL is valid — already in bootstrap mode."""
        count, resets = self._run([self._wm("VIIRS_SNPP_NRT", "world", None)])
        self.assertEqual(0, count)
        self.assertEqual([], resets)

    def test_sane_recent_watermark_is_not_reset(self):
        good_ts = self._NOW - timedelta(hours=1)
        count, resets = self._run([self._wm("VIIRS_SNPP_NRT", "world", good_ts)])
        self.assertEqual(0, count)
        self.assertEqual([], resets)

    # --- Mixed batch ---

    def test_multiple_watermarks_only_corrupt_ones_reset(self):
        watermarks = [
            self._wm("VIIRS_SNPP_NRT", "world", self._NOW - timedelta(hours=1)),      # valid
            self._wm("VIIRS_NOAA20_NRT", "world", self._NOW + timedelta(hours=2)),    # future
            self._wm("VIIRS_SNPP_NRT", "-130,30,-60,60", self._NOW - timedelta(days=45)),  # stale NRT
            self._wm("VIIRS_SNPP_SP", "world", self._NOW - timedelta(days=60)),        # old but not NRT
        ]
        count, resets = self._run(watermarks)
        self.assertEqual(2, count)
        self.assertIn(("VIIRS_NOAA20_NRT", "world"), resets)
        self.assertIn(("VIIRS_SNPP_NRT", "-130,30,-60,60"), resets)
        self.assertNotIn(("VIIRS_SNPP_NRT", "world"), resets)
        self.assertNotIn(("VIIRS_SNPP_SP", "world"), resets)

    # --- Error resilience ---

    def test_db_error_during_list_returns_zero_and_does_not_raise(self):
        def _fail():
            raise RuntimeError("db connection refused")

        count = validate_and_reset_watermarks(
            now_utc=self._NOW,
            list_fn=_fail,
            reset_fn=lambda s, a: None,
        )
        self.assertEqual(0, count)

    def test_reset_failure_is_not_counted_but_does_not_abort_remaining(self):
        """If the DB reset for one watermark fails, the others still get processed."""
        future_ts = self._NOW + timedelta(hours=2)
        also_future_ts = self._NOW + timedelta(hours=3)
        succeeded: list[str] = []

        def _reset(s: str, a: str) -> None:
            if s == "VIIRS_SNPP_NRT":
                raise RuntimeError("transient error")
            succeeded.append(s)

        count = validate_and_reset_watermarks(
            now_utc=self._NOW,
            list_fn=lambda: [
                self._wm("VIIRS_SNPP_NRT", "world", future_ts),
                self._wm("VIIRS_NOAA20_NRT", "world", also_future_ts),
            ],
            reset_fn=_reset,
        )
        # Only NOAA20 reset succeeded
        self.assertEqual(1, count)
        self.assertEqual(["VIIRS_NOAA20_NRT"], succeeded)

    def test_empty_watermarks_returns_zero(self):
        count, resets = self._run([])
        self.assertEqual(0, count)
        self.assertEqual([], resets)


class TestFuelJobs(unittest.TestCase):
    """Tests for LFMC and LULC orchestrator integration."""

    def test_lfmc_in_job_order(self):
        self.assertIn(JOB_LFMC, JOB_ORDER)

    def test_lulc_in_job_order(self):
        self.assertIn(JOB_LULC, JOB_ORDER)

    def test_lfmc_after_weather(self):
        self.assertGreater(JOB_ORDER.index(JOB_LFMC), JOB_ORDER.index(JOB_WEATHER))

    def test_lulc_after_weather(self):
        self.assertGreater(JOB_ORDER.index(JOB_LULC), JOB_ORDER.index(JOB_WEATHER))

    def test_lfmc_before_lulc(self):
        self.assertLess(JOB_ORDER.index(JOB_LFMC), JOB_ORDER.index(JOB_LULC))

    def test_run_lfmc_timeout_returns_one_and_logs_warning(self):
        """A TimeoutError from the LFMC API must return exit_code=1 with a warning log."""
        args = argparse.Namespace(
            lfmc_bbox=None,
            lfmc_timeout_seconds=30.0,
        )
        with patch(
            "ingest.lfmc_ecland_ingest.ingest_lfmc_ecland_for_bbox",
            side_effect=TimeoutError("LFMC ecLand job timed out"),
        ):
            with self.assertLogs("ingest_orchestrator", level="WARNING") as log_ctx:
                code = _run_lfmc(args)

        self.assertEqual(1, code)
        self.assertTrue(
            any("pipeline continues" in line for line in log_ctx.output),
            f"Expected 'pipeline continues' in warning log, got: {log_ctx.output}",
        )

    def test_run_lfmc_api_error_returns_one(self):
        """Any LFMC exception (e.g. missing API URL) returns exit_code=1."""
        args = argparse.Namespace(
            lfmc_bbox=None,
            lfmc_timeout_seconds=30.0,
        )
        with patch(
            "ingest.lfmc_ecland_ingest.ingest_lfmc_ecland_for_bbox",
            side_effect=RuntimeError("LFMC_ECLAND_API_URL is not set"),
        ):
            with self.assertLogs("ingest_orchestrator", level="WARNING"):
                code = _run_lfmc(args)

        self.assertEqual(1, code)

    def test_lfmc_failure_does_not_stop_subsequent_jobs(self):
        """LFMC timeout/failure must not block downstream jobs in run_once."""
        calls: list[str] = []

        def _lfmc_timeout() -> int:
            calls.append("lfmc")
            return 1  # simulates _run_lfmc returning 1 after timeout

        def _downstream() -> int:
            calls.append("downstream")
            return 0

        jobs = [
            ScheduledJob(name="lfmc", interval_seconds=21600.0, runner=_lfmc_timeout),
            ScheduledJob(name="terrain", interval_seconds=86400.0, runner=_downstream),
        ]
        exit_code = run_once(jobs, stop_on_error=False)

        # Pipeline completes; overall exit is 1 (has failure) but downstream ran
        self.assertEqual(1, exit_code)
        self.assertEqual(["lfmc", "downstream"], calls)

    def test_lfmc_failure_with_stop_on_error_halts_pipeline(self):
        """stop_on_error=True must stop after LFMC failure (existing semantics, not LFMC-specific)."""
        calls: list[str] = []

        def _lfmc_timeout() -> int:
            calls.append("lfmc")
            return 1

        def _downstream() -> int:  # pragma: no cover
            calls.append("downstream")
            return 0

        jobs = [
            ScheduledJob(name="lfmc", interval_seconds=21600.0, runner=_lfmc_timeout),
            ScheduledJob(name="terrain", interval_seconds=86400.0, runner=_downstream),
        ]
        exit_code = run_once(jobs, stop_on_error=True)

        self.assertEqual(1, exit_code)
        self.assertEqual(["lfmc"], calls)  # downstream did not run


if __name__ == "__main__":
    unittest.main()
