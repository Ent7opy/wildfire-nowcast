import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.mark.integration
def test_get_risk_endpoint_returns_geojson_grid(db_available):
    """Test that the /risk endpoint returns GeoJSON grid with multiple cells."""
    response = client.get(
        "/risk",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 43.0,
            "cell_size_km": 50.0,  # Large cells for small test grid
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 1  # Should have multiple grid cells
    
    # Check first feature structure
    feature = data["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert "risk_score" in feature["properties"]
    assert "risk_level" in feature["properties"]
    assert "components" in feature["properties"]
    
    # Risk score should be in valid range
    risk_score = feature["properties"]["risk_score"]
    assert 0.0 <= risk_score <= 1.0
    
    # Risk level should be categorical
    risk_level = feature["properties"]["risk_level"]
    assert risk_level in ["low", "medium", "high"]


@pytest.mark.integration
def test_get_risk_endpoint_grid_cells_cover_bbox(db_available):
    """Test that risk grid cells cover the requested bbox."""
    response = client.get(
        "/risk",
        params={
            "min_lon": 10.0,
            "min_lat": 30.0,
            "max_lon": 15.0,
            "max_lat": 35.0,
            "cell_size_km": 100.0,  # Large cells
        },
    )

    assert response.status_code == 200
    data = response.json()
    
    # Collect all polygon coordinates
    all_coords = []
    for feature in data["features"]:
        coords = feature["geometry"]["coordinates"][0]
        all_coords.extend(coords[:-1])  # Skip duplicate last point
    
    # Extract min/max from all cells
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    
    # Grid should cover the requested bbox (approximately)
    assert min(lons) <= 10.1  # Allow small margin
    assert max(lons) >= 14.9
    assert min(lats) <= 30.1
    assert max(lats) >= 34.9


@pytest.mark.integration
def test_get_risk_endpoint_respects_cell_size(db_available):
    """Test that cell_size_km parameter controls grid resolution."""
    # Test with large cells (should return fewer features)
    response_large = client.get(
        "/risk",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 42.0,
            "cell_size_km": 50.0,
        },
    )
    
    # Test with small cells (should return more features)
    response_small = client.get(
        "/risk",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 42.0,
            "cell_size_km": 10.0,
        },
    )
    
    assert response_large.status_code == 200
    assert response_small.status_code == 200
    
    large_count = len(response_large.json()["features"])
    small_count = len(response_small.json()["features"])
    
    # Smaller cells should produce more features
    assert small_count > large_count
