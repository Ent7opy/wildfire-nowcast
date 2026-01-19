from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_get_risk_endpoint_returns_geojson():
    """Test that the /risk endpoint returns GeoJSON with baseline risk."""
    response = client.get(
        "/risk",
        params={
            "min_lon": 20.0,
            "min_lat": 40.0,
            "max_lon": 22.0,
            "max_lat": 43.0,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1
    
    feature = data["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert "risk_score" in feature["properties"]
    assert feature["properties"]["risk_level"] == "low"


def test_get_risk_endpoint_bbox_in_response():
    """Test that the risk endpoint uses the provided bbox."""
    response = client.get(
        "/risk",
        params={
            "min_lon": 10.0,
            "min_lat": 30.0,
            "max_lon": 15.0,
            "max_lat": 35.0,
        },
    )

    assert response.status_code == 200
    data = response.json()
    coords = data["features"][0]["geometry"]["coordinates"][0]
    
    # Check that bbox corners are in the polygon
    assert [10.0, 30.0] in coords
    assert [15.0, 35.0] in coords
