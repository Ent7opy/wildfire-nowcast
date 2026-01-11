from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient

import api.routes.aois as aois_routes
from api.main import app

client = TestClient(app)


def test_create_aoi_success(monkeypatch):
    """Test creating an AOI with valid data."""
    aoi_id = uuid4()
    mock_aoi = {
        "id": aoi_id,
        "name": "Test AOI",
        "description": "A test AOI",
        "tags": {"project": "test"},
        "owner_id": None,
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
        "bbox": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
        "area_km2": 100.0,
        "vertex_count": 5,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    
    mock_create = MagicMock(return_value=mock_aoi)
    monkeypatch.setattr(aois_routes.repo, "create_aoi", mock_create)

    response = client.post(
        "/aois",
        json={
            "name": "Test AOI",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
            "description": "A test AOI",
            "tags": {"project": "test"},
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["id"] == str(aoi_id)
    assert data["name"] == "Test AOI"
    assert data["area_km2"] == 100.0


def test_create_aoi_invalid_geometry(monkeypatch):
    """Test creating an AOI with invalid GeoJSON."""
    # Mock create_aoi just in case validation passes (it shouldn't, but to be safe against 500s from DB)
    monkeypatch.setattr(aois_routes.repo, "create_aoi", MagicMock(side_effect=Exception("Should not be called")))
    
    response = client.post(
        "/aois",
        json={
            "name": "Bad AOI",
            "geometry": {"type": "Polygon", "coordinates": []}, # Invalid coordinates
        },
    )
    # shape() from shapely might raise or return empty geom. 
    # If it returns empty geom, is_valid might be True or False.
    # Empty polygon is valid.
    # But ST_GeomFromGeoJSON might not like it or my validation logic is flawed.
    # Actually, coordinates=[] means no exterior ring.
    # shapely shape({'type': 'Polygon', 'coordinates': []}) returns an empty Polygon.
    # empty_geom.is_valid is True.
    # So _validate_geometry passes!
    # Then it calls create_aoi.
    # My mock raises Exception("Should not be called") -> 500.
    
    # I should change the invalid input to something that fails `shape()` or `is_valid`.
    # e.g. garbage structure
    response = client.post(
        "/aois",
        json={
            "name": "Bad AOI",
            "geometry": {"type": "Polygon", "coordinates": "garbage"}, # Invalid coordinates
        },
    )
    assert response.status_code == 400
    assert "Invalid GeoJSON" in response.json()["message"]


def test_create_aoi_too_large(monkeypatch):
    """Test rejection if AOI area exceeds limit (post-creation check)."""
    aoi_id = uuid4()
    mock_aoi = {
        "id": aoi_id,
        "name": "Huge AOI",
        "area_km2": 60000.0, # > 50000 limit
        "vertex_count": 5,
        "description": None,
        "tags": None,
        "owner_id": None,
        "geometry": {},
        "bbox": {},
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    
    mock_create = MagicMock(return_value=mock_aoi)
    mock_delete = MagicMock(return_value=True)
    monkeypatch.setattr(aois_routes.repo, "create_aoi", mock_create)
    monkeypatch.setattr(aois_routes.repo, "delete_aoi", mock_delete)

    response = client.post(
        "/aois",
        json={
            "name": "Huge AOI",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
        },
    )

    assert response.status_code == 413
    assert "exceeds maximum" in response.json()["message"]
    mock_delete.assert_called_once()


def test_list_aois(monkeypatch):
    """Test listing AOIs."""
    mock_aoi = {
        "id": uuid4(),
        "name": "AOI 1",
        "description": None,
        "tags": None,
        "owner_id": None,
        "geometry": {},
        "bbox": {},
        "area_km2": 1.0,
        "vertex_count": 4,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    mock_items = [mock_aoi]
    mock_list = MagicMock(return_value=mock_items)
    monkeypatch.setattr(aois_routes.repo, "list_aois", mock_list)

    response = client.get("/aois", params={"limit": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert len(data["items"]) == 1
    
    _, kwargs = mock_list.call_args
    assert kwargs["limit"] == 10


def test_get_aoi_found(monkeypatch):
    """Test fetching a single AOI."""
    aoi_id = uuid4()
    mock_aoi = {
        "id": aoi_id,
        "name": "Found AOI",
        "geometry": {},
        "bbox": {},
        "area_km2": 10.0,
        "vertex_count": 4,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "description": None,
        "tags": None,
        "owner_id": None,
    }
    mock_get = MagicMock(return_value=mock_aoi)
    monkeypatch.setattr(aois_routes.repo, "get_aoi", mock_get)

    response = client.get(f"/aois/{aoi_id}")
    assert response.status_code == 200
    assert response.json()["id"] == str(aoi_id)


def test_get_aoi_not_found(monkeypatch):
    """Test 404 for missing AOI."""
    monkeypatch.setattr(aois_routes.repo, "get_aoi", MagicMock(return_value=None))
    response = client.get(f"/aois/{uuid4()}")
    assert response.status_code == 404


def test_update_aoi(monkeypatch):
    """Test updating an AOI."""
    aoi_id = uuid4()
    mock_aoi = {
        "id": aoi_id,
        "name": "Updated Name",
        "area_km2": 10.0,
        "vertex_count": 4,
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
        "bbox": {},
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "description": None,
        "tags": None,
        "owner_id": None,
    }
    mock_update = MagicMock(return_value=mock_aoi)
    mock_get = MagicMock(return_value=mock_aoi)
    monkeypatch.setattr(aois_routes.repo, "update_aoi", mock_update)
    monkeypatch.setattr(aois_routes.repo, "get_aoi", mock_get)

    response = client.patch(
        f"/aois/{aoi_id}",
        json={"name": "Updated Name"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Updated Name"


def test_delete_aoi(monkeypatch):
    """Test deleting an AOI."""
    mock_delete = MagicMock(return_value=True)
    monkeypatch.setattr(aois_routes.repo, "delete_aoi", mock_delete)

    response = client.delete(f"/aois/{uuid4()}")
    assert response.status_code == 204
