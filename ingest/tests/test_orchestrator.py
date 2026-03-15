import argparse
import unittest

from ingest.orchestrator import (
    ScheduledJob,
    _build_industrial_argv,
    _build_weather_argv,
    run_once,
    run_scheduler,
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


if __name__ == "__main__":
    unittest.main()
