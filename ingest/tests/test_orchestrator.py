import argparse
import unittest

from ingest.orchestrator import ScheduledJob, _build_weather_argv, run_once, run_scheduler


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


if __name__ == "__main__":
    unittest.main()
