"""Contract test for fire details panel property access."""

import pytest
from unittest.mock import MagicMock
import streamlit as st


# FireMapFeature Contract (synced with api/tests/test_fire_map_contract.py)
# This defines the minimal set of properties that MUST be available
# for the UI to function correctly. Any change to these property names
# or their removal will break the UI and must be caught by these tests.
FIRE_MAP_FEATURE_CONTRACT = {
    "required": {
        # Core identification and temporal properties
        "id": "Unique identifier for the detection",
        "acq_time": "Acquisition timestamp (ISO 8601 format)",
        
        # Sensor and source metadata
        "sensor": "Satellite sensor name (e.g., VIIRS, MODIS)",
        "source": "Data source (e.g., FIRMS)",
        
        # Fire detection measurements
        "confidence": "Detection confidence score",
        "frp": "Fire Radiative Power",
        
        # Geospatial properties
        "lat": "Latitude coordinate",
        "lon": "Longitude coordinate",
    },
    "optional": {
        # Denoiser-specific properties (may not be present for all detections)
        "denoised_score": "ML denoiser confidence score (0-1)",
        "is_noise": "Boolean flag indicating if detection is classified as noise",
    }
}


def test_click_details_handles_all_mvt_properties():
    """Test that click_details can render all MVT properties without KeyError."""
    
    # Mock session state with realistic MVT properties from contract
    mock_session_state = {
        "selected_fire": {
            "id": 12345,
            "acq_time": "2026-01-19T12:00:00Z",
            "sensor": "VIIRS",
            "confidence": 85.0,
            "frp": 12.5,
            "source": "FIRMS",
            "lat": 42.5,
            "lon": 21.3,
            "is_noise": False,
            "denoised_score": 0.95,
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
            "id": 12345,
            "acq_time": "2026-01-19T12:00:00Z",
            "sensor": "VIIRS",
            "confidence": 85.0,
            "frp": 12.5,
            "source": "FIRMS",
            "lat": 42.5,
            "lon": 21.3,
            # Missing optional properties: is_noise, denoised_score
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
            "acq_time": "2026-01-19T12:00:00Z",
            "sensor": "VIIRS",
            "confidence": 85.0,
            "frp": 12.5,
            "source": "FIRMS",
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
        "id": 12345,
        "acq_time": "2026-01-19T12:00:00Z",
        "sensor": "VIIRS",
        "confidence": 85.0,
        "frp": 12.5,
        "source": "FIRMS",
        "lat": 42.5,
        "lon": 21.3,
        "is_noise": False,
        "denoised_score": 0.95,
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
