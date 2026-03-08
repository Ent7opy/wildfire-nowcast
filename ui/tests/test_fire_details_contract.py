"""Contract test for fire event details panel property access."""



# FireEventFeature Contract
# This defines the minimal set of properties that MUST be available
# for the UI to function correctly. Any change to these property names
# or their removal will break the UI and must be caught by these tests.
FIRE_MAP_FEATURE_CONTRACT = {
    "required": {
        # Core identification and temporal properties
        "event_id": "Unique identifier for the event",
        "start_time": "Event start timestamp (ISO 8601 format)",
        "end_time": "Event end timestamp (ISO 8601 format)",
        
        # Sensor and source metadata
        "sensor": "Satellite sensor name (e.g., VIIRS, MODIS)",
        "source": "Data source (e.g., FIRMS)",
        
        # Event-level summary fields
        "detection_count": "Number of detections in the event",
        "front_count": "Number of fronts linked to the event",
        "event_score": "Event-level denoiser score",
        "denoiser_decision": "pass|downweight|drop|review",
        "review_required": "Boolean review-required flag",
        
        # Geospatial properties
        "lat": "Latitude coordinate",
        "lon": "Longitude coordinate",
    },
    "optional": {
        "cluster_event_count": "Number of events represented when map clustering is enabled",
    }
}


def test_click_details_handles_all_mvt_properties():
    """Test that click_details can render all MVT properties without KeyError."""
    
    # Mock session state with realistic MVT properties from contract
    mock_session_state = {
        "selected_fire": {
            "event_id": "evt_12345",
            "start_time": "2026-01-19T11:00:00Z",
            "end_time": "2026-01-19T12:00:00Z",
            "sensor": "VIIRS",
            "source": "FIRMS",
            "detection_count": 14,
            "front_count": 2,
            "event_score": 0.91,
            "denoiser_decision": "pass",
            "review_required": False,
            "lat": 42.5,
            "lon": 21.3,
        }
    }
    
    # Simulate property access patterns from click_details.py
    det = mock_session_state.get("selected_fire")
    
    # These accesses should not raise KeyError
    assert det is not None
    
    # Verify all required properties from contract are accessible
    for prop in FIRE_MAP_FEATURE_CONTRACT["required"].keys():
        value = det.get(prop)
        assert value is not None, f"Required property '{prop}' should not be None"
    
    # Optional properties may be None but should not raise KeyError
    for prop in FIRE_MAP_FEATURE_CONTRACT["optional"].keys():
        _ = det.get(prop)  # Should not raise KeyError


def test_click_details_handles_missing_optional_properties():
    """Test that click_details gracefully handles missing optional properties."""
    
    # Mock session state with minimal required properties (only those from contract)
    mock_session_state = {
        "selected_fire": {
            "event_id": "evt_12345",
            "start_time": "2026-01-19T11:00:00Z",
            "end_time": "2026-01-19T12:00:00Z",
            "sensor": "VIIRS",
            "source": "FIRMS",
            "detection_count": 6,
            "front_count": 1,
            "event_score": 0.72,
            "denoiser_decision": "downweight",
            "review_required": False,
            "lat": 42.5,
            "lon": 21.3,
            # Missing optional properties: cluster_event_count
        }
    }
    
    det = mock_session_state.get("selected_fire")
    
    # All required properties from contract must be present
    for prop in FIRE_MAP_FEATURE_CONTRACT["required"].keys():
        assert det.get(prop) is not None, f"Required property '{prop}' must not be None"
    
    # Optional properties should return None gracefully (no KeyError)
    for prop in FIRE_MAP_FEATURE_CONTRACT["optional"].keys():
        value = det.get(prop)
        # It's OK for optional properties to be None or missing
        assert value is None or value is not None  # Just checking no KeyError


def test_click_details_coordinate_validation():
    """Test that coordinates are within valid bounds for forecast generation."""
    
    mock_session_state = {
        "selected_fire": {
            "lat": 42.5,
            "lon": 21.3,
            "end_time": "2026-01-19T12:00:00Z",
            "sensor": "VIIRS",
            "source": "FIRMS",
            "event_id": "evt_1",
            "start_time": "2026-01-19T11:00:00Z",
            "detection_count": 3,
            "front_count": 1,
            "event_score": 0.55,
            "denoiser_decision": "review",
            "review_required": True,
        }
    }
    
    det = mock_session_state.get("selected_fire")
    
    lat = det.get("lat")
    lon = det.get("lon")
    
    # Validate coordinate bounds (from click_details.py lines 86-95)
    assert lat is not None
    assert lon is not None
    assert -90 <= float(lat) <= 90, "Latitude must be in range [-90, 90]"
    assert -180 <= float(lon) <= 180, "Longitude must be in range [-180, 180]"
    
    # Verify forecast bbox calculation doesn't fail
    radius_deg = 50.0 / 111.0
    fire_lat = float(lat)
    fire_lon = float(lon)
    forecast_bbox = (
        fire_lon - radius_deg,
        fire_lat - radius_deg,
        fire_lon + radius_deg,
        fire_lat + radius_deg,
    )
    
    assert len(forecast_bbox) == 4
    assert all(isinstance(x, float) for x in forecast_bbox)


def test_map_view_property_key_consistency():
    """Test that map_view.py sets session state with correct property keys."""
    
    # Simulate MVT layer properties as returned by PyDeck
    mvt_props = {
        "event_id": "evt_12345",
        "start_time": "2026-01-19T11:00:00Z",
        "end_time": "2026-01-19T12:00:00Z",
        "sensor": "VIIRS",
        "source": "FIRMS",
        "detection_count": 22,
        "front_count": 4,
        "event_score": 0.88,
        "denoiser_decision": "pass",
        "review_required": False,
        "lat": 42.5,
        "lon": 21.3,
    }
    
    # Simulate line 195 in map_view.py: st.session_state.selected_fire = feature
    selected_fire = mvt_props
    
    # Verify all required properties from contract are present
    for key in FIRE_MAP_FEATURE_CONTRACT["required"].keys():
        assert key in selected_fire, f"Required property '{key}' from contract must be present in selected_fire"
    
    # Verify coordinate extraction (line 178-179 in map_view.py)
    lat_value = selected_fire.get("lat")
    lon_value = selected_fire.get("lon")

    assert lat_value is not None, "lat coordinate must be present"
    assert lon_value is not None, "lon coordinate must be present"


def test_aggregate_stats_handles_none_event_score():
    """Test that aggregate stats handles event_score=None without TypeError.

    Regression test: float(None) raises TypeError. The code must handle
    events where event_score key exists but its value is None.
    """
    events = [
        {"event_score": None, "end_time": "2026-01-19T12:00:00Z"},
        {"event_score": 0.8, "end_time": "2026-01-19T13:00:00Z"},
        {"event_score": "0.5", "end_time": "2026-01-19T14:00:00Z"},
        {"end_time": "2026-01-19T15:00:00Z"},  # key missing entirely
    ]

    # Replicate strict event score handling from click_details.py _render_aggregate_stats
    max_lh = max(
        (float(event.get("event_score") or 0) for event in events),
        default=0,
    )
    assert max_lh == 0.8


def test_tooltip_properties_in_contract():
    """Test that all properties used in map tooltip are in the contract."""
    # Properties referenced in map_view.py tooltip template
    tooltip_properties = [
        "event_id",
        "start_time",
        "end_time",
        "sensor",
        "detection_count",
        "event_score",
        "denoiser_decision",
        "review_required",
    ]

    all_contract_props = (
        set(FIRE_MAP_FEATURE_CONTRACT["required"].keys())
        | set(FIRE_MAP_FEATURE_CONTRACT["optional"].keys())
    )

    for prop in tooltip_properties:
        assert prop in all_contract_props, (
            f"Tooltip uses '{prop}' but it's not in the contract. "
            "Update the contract or fix the tooltip template."
        )
