import pytest

from ml.denoiser.eventize import EventizeParams, _is_static_like


def test_eventize_params_defaults_are_valid() -> None:
    params = EventizeParams()
    assert params.front_link_radius_m > 0
    assert params.front_max_gap_minutes > 0
    assert params.event_link_radius_m > 0
    assert params.event_max_gap_days > 0
    assert 0.0 <= params.static_persistence_threshold <= 1.0
    assert params.strict_static_split is True
    assert 0.0 <= params.perimeter_match_overlap_min <= 1.0
    assert params.perimeter_match_time_pad_hours >= 0
    assert 0.0 < params.estimated_concave_percent <= 1.0
    assert params.estimated_point_buffer_m > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("front_link_radius_m", 0),
        ("front_max_gap_minutes", 0),
        ("event_link_radius_m", 0),
        ("event_max_gap_days", 0),
        ("static_persistence_threshold", 1.1),
        ("perimeter_match_overlap_min", -0.1),
        ("perimeter_match_time_pad_hours", -1),
        ("estimated_concave_percent", 0),
        ("estimated_point_buffer_m", 0),
    ],
)
def test_eventize_params_validation(field: str, value: float) -> None:
    kwargs = {
        "front_link_radius_m": 2500.0,
        "front_max_gap_minutes": 45,
        "event_link_radius_m": 10000.0,
        "event_max_gap_days": 11,
        "static_persistence_threshold": 0.85,
        "strict_static_split": True,
        "perimeter_match_overlap_min": 0.2,
        "perimeter_match_time_pad_hours": 24,
        "estimated_concave_percent": 0.92,
        "estimated_point_buffer_m": 375.0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        EventizeParams(**kwargs)


def test_is_static_like_logic() -> None:
    assert _is_static_like(True, 0.1, 0.85) is True
    assert _is_static_like(False, 0.9, 0.85) is True
    assert _is_static_like(False, 0.2, 0.85) is False
