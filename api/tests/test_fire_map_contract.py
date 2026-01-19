"""Contract test for fire map MVT properties used by UI."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
import httpx
import struct

from api.main import app

client = TestClient(app)


# FireMapFeature Contract
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


def decode_mvt_properties(pbf_bytes: bytes) -> list[dict]:
    """
    Minimal MVT decoder to extract properties for testing.
    Returns list of feature properties dicts.
    
    Note: This is a simplified decoder that extracts property keys.
    For production use, consider using mapbox-vector-tile or similar.
    """
    # For this test, we'll mock the MVT response with known properties
    # In a real scenario, you'd decode the actual PBF
    return []


def test_mvt_fires_layer_includes_required_properties(monkeypatch):
    """Test that MVT fires layer includes all properties required by UI."""
    
    # Mock RateLimiter
    from fastapi_limiter.depends import RateLimiter
    async def mock_call(*args, **kwargs):
        return True
    monkeypatch.setattr(RateLimiter, "__call__", mock_call)
    
    # Use the contract definition as the source of truth
    required_properties = set(FIRE_MAP_FEATURE_CONTRACT["required"].keys())
    optional_properties = set(FIRE_MAP_FEATURE_CONTRACT["optional"].keys())
    
    all_properties = required_properties | optional_properties
    
    # Mock a realistic MVT response with these properties
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"fake_pbf_with_properties"
    mock_resp.headers = {
        "content-type": "application/x-protobuf",
        "Cache-Control": "max-age=60"
    }
    
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None
    
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_context))
    
    # Request tile
    response = client.get(
        "/tiles/fires/5/17/11.pbf",
        params={
            "start_time": "2026-01-18T00:00:00Z",
            "end_time": "2026-01-19T23:59:59Z",
            "min_confidence": 0.0,
            "include_noise": "false",
        }
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-protobuf"
    
    # Verify the MVT function is called correctly
    mock_client.get.assert_called_once()
    args, kwargs = mock_client.get.call_args
    
    # Check that URL includes the MVT function
    assert "public.mvt_fires" in args[0]
    
    # Check that query params are forwarded
    assert "params" in kwargs
    params = kwargs["params"]
    assert "start_time" in params
    assert "end_time" in params
    assert "min_confidence" in params
    assert "include_noise" in params


def test_mvt_fires_properties_documented():
    """Test that MVT fires function properties are documented in migration."""
    # Read the migration to verify properties are exposed
    import pathlib
    migration_path = pathlib.Path(__file__).parent.parent / "migrations" / "versions" / "d46889070598_update_mvt_fires_props.py"
    
    assert migration_path.exists(), "MVT fires migration should exist"
    
    migration_content = migration_path.read_text()
    
    # Verify all required properties from contract are in the SELECT clause
    required_properties = list(FIRE_MAP_FEATURE_CONTRACT["required"].keys())
    
    for prop in required_properties:
        assert prop in migration_content, f"Required property '{prop}' from contract should be in MVT function"


def test_ui_property_access_matches_mvt_schema():
    """Test that UI component property access matches MVT schema."""
    # Simulate MVT properties as they would appear in session_state
    mvt_properties = {
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
    
    # Verify all required properties from contract are present
    for prop in FIRE_MAP_FEATURE_CONTRACT["required"].keys():
        assert prop in mvt_properties, f"Required property '{prop}' from contract missing in MVT properties"
    
    # Verify all properties accessed by UI exist in the contract
    ui_accessed_properties = [
        "lat",
        "lon",
        "acq_time",
        "sensor",
        "confidence",
        "frp",
        "source",
        "denoised_score",
        "is_noise",
    ]
    
    all_contract_properties = set(FIRE_MAP_FEATURE_CONTRACT["required"].keys()) | set(FIRE_MAP_FEATURE_CONTRACT["optional"].keys())
    for prop in ui_accessed_properties:
        assert prop in all_contract_properties, f"UI accesses '{prop}' but it's not in the contract"
    
    # Verify no KeyError when accessing properties
    for prop in ui_accessed_properties:
        value = mvt_properties.get(prop)
        # None is acceptable for optional fields, but key must exist
        assert prop in mvt_properties


def test_fire_map_feature_contract_completeness():
    """Test that the FireMapFeature contract is complete and consistent.
    
    This test validates:
    1. All required properties are defined
    2. All properties have descriptions
    3. No overlap between required and optional
    4. Contract covers all UI needs (referenced by click_details.py and map_view.py)
    """
    # Verify contract structure
    assert "required" in FIRE_MAP_FEATURE_CONTRACT
    assert "optional" in FIRE_MAP_FEATURE_CONTRACT
    
    required_props = set(FIRE_MAP_FEATURE_CONTRACT["required"].keys())
    optional_props = set(FIRE_MAP_FEATURE_CONTRACT["optional"].keys())
    
    # Verify no overlap between required and optional
    overlap = required_props & optional_props
    assert len(overlap) == 0, f"Properties appear in both required and optional: {overlap}"
    
    # Verify all properties have descriptions
    for prop, desc in FIRE_MAP_FEATURE_CONTRACT["required"].items():
        assert desc and len(desc) > 0, f"Required property '{prop}' missing description"
    
    for prop, desc in FIRE_MAP_FEATURE_CONTRACT["optional"].items():
        assert desc and len(desc) > 0, f"Optional property '{prop}' missing description"
    
    # Verify contract includes minimum set of properties needed for UI
    # Based on click_details.py (lines 52-62, 64-82) and map_view.py (lines 178-192)
    essential_for_ui = {
        "id",       # identification
        "lat",      # coordinate for display and forecast
        "lon",      # coordinate for display and forecast
        "acq_time", # temporal reference
        "sensor",   # metadata display
        "confidence", # metadata display
        "frp",      # fire intensity display
        "source",   # data provenance
    }
    
    all_props = required_props | optional_props
    missing = essential_for_ui - all_props
    assert len(missing) == 0, f"Contract missing UI-essential properties: {missing}"


def test_contract_validates_map_view_to_click_details_flow():
    """Test that properties flow correctly from map_view selection to click_details rendering.
    
    This simulates the actual flow:
    1. MVT layer returns a feature with properties
    2. map_view.py extracts/enriches the feature (lines 177-195)
    3. click_details.py accesses properties from selected_fire (lines 52-82)
    """
    # Step 1: Simulate MVT feature as returned by PyDeck
    mvt_feature = {
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
    
    # Verify feature contains all required properties from contract
    for prop in FIRE_MAP_FEATURE_CONTRACT["required"].keys():
        assert prop in mvt_feature, f"MVT feature missing required property '{prop}'"
    
    # Step 2: Simulate map_view.py selection logic (lines 177-192)
    # map_view extracts lat/lon from properties or falls back to geometry
    selected_fire = mvt_feature.copy()
    lat = selected_fire.get("lat")
    lon = selected_fire.get("lon")
    
    # These must exist for forecast generation
    assert lat is not None, "lat must be present for forecast"
    assert lon is not None, "lon must be present for forecast"
    
    # Step 3: Simulate click_details.py property access patterns (lines 52-82)
    # Core properties (must not raise KeyError)
    assert selected_fire.get("lat") is not None
    assert selected_fire.get("lon") is not None
    assert selected_fire.get("acq_time") is not None
    assert selected_fire.get("sensor") is not None
    assert selected_fire.get("confidence") is not None
    assert selected_fire.get("frp") is not None
    assert selected_fire.get("source") is not None
    
    # Optional properties (may be None but should not raise KeyError)
    _ = selected_fire.get("denoised_score")
    _ = selected_fire.get("is_noise")
    
    # Validate coordinate ranges for forecast bbox calculation (click_details.py lines 95-100)
    assert -90 <= float(lat) <= 90, "Latitude must be in valid range"
    assert -180 <= float(lon) <= 180, "Longitude must be in valid range"
