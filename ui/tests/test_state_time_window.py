from state import AppState


def test_time_window_is_exact_for_last_hour() -> None:
    state = AppState()
    state.filters.hours_start = 1
    state.filters.hours_end = 0
    assert state.time_window == "Last 1 hour"


def test_time_window_is_exact_for_last_six_hours() -> None:
    state = AppState()
    state.filters.hours_start = 6
    state.filters.hours_end = 0
    assert state.time_window == "Last 6 hours"


def test_time_window_describes_offset_windows() -> None:
    state = AppState()
    state.filters.hours_start = 7
    state.filters.hours_end = 2
    assert state.time_window == "5h window (7h ago to 2h ago)"


def test_time_range_last_hour_is_60_minutes() -> None:
    state = AppState()
    state.filters.hours_start = 1
    state.filters.hours_end = 0
    start, end = state.time_range
    assert (end - start).total_seconds() == 3600
