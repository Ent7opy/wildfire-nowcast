from state import ForecastDisplayState, ForecastJobState


def test_forecast_display_radius_defaults_to_20km() -> None:
    state = ForecastDisplayState()
    assert state.forecast_radius_km == 20


def test_forecast_job_complete_copies_request_context() -> None:
    job = ForecastJobState()
    ctx = {
        "event_key": "event_id:evt-9",
        "event_id": "evt-9",
        "lat": 31.0,
        "lon": 46.0,
    }
    job.start("job-1", request_context=ctx)
    job.complete("run-1", "job-1")

    assert job.last_forecast is not None
    assert job.last_forecast["run"]["id"] == "run-1"
    assert job.last_forecast["event_key"] == "event_id:evt-9"
    assert job.job_id is None
    assert job.active_request is None


def test_forecast_job_clear_resets_inflight_request() -> None:
    job = ForecastJobState(job_id="job-x", poll_count=4, active_request={"event_key": "abc"})
    job.clear()

    assert job.job_id is None
    assert job.poll_count == 0
    assert job.active_request is None
