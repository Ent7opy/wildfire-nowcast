from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
import httpx
from api.main import app
import api.routes.tiles as tiles_routes

client = TestClient(app)

def test_proxy_tiles_success(monkeypatch):
    """Test successful tile proxying."""
    
    # Mock RateLimiter.__call__ to skip logic
    from fastapi_limiter.depends import RateLimiter
    async def mock_call(*args, **kwargs): return True
    monkeypatch.setattr(RateLimiter, "__call__", mock_call)
    
    # Mock httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"fake_pbf_data"
    mock_resp.headers = {"content-type": "application/x-protobuf", "Cache-Control": "max-age=300"}
    
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    
    # Mock httpx.AsyncClient context manager
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None
    
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_context))

    response = client.get("/tiles/fires/0/0/0.pbf")
    
    assert response.status_code == 200
    mock_resp.content = b"fake_pbf_data"
    mock_resp.headers = {"content-type": "application/x-protobuf", "Cache-Control": "max-age=300"}
    
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    
    # Mock httpx.AsyncClient context manager
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None
    
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_context))

    response = client.get("/tiles/fires/0/0/0.pbf")
    
    assert response.status_code == 200
    assert response.content == b"fake_pbf_data"
    assert response.headers["content-type"] == "application/x-protobuf"
    
    # Verify url call
    mock_client.get.assert_called_once()
    args, _ = mock_client.get.call_args
    assert "/public.mvt_fires/0/0/0.pbf" in args[0]

def test_proxy_tiles_invalid_layer(monkeypatch):
    """Test 404 for unknown layer."""
    from fastapi_limiter.depends import RateLimiter
    async def mock_call(*args, **kwargs): return True
    monkeypatch.setattr(RateLimiter, "__call__", mock_call)
    
    response = client.get("/tiles/unknown_layer/0/0/0.pbf")
    assert response.status_code == 404

def test_proxy_tiles_backend_error(monkeypatch):
    """Test 502 when tile server is down."""
    from fastapi_limiter.depends import RateLimiter
    async def mock_call(*args, **kwargs): return True
    monkeypatch.setattr(RateLimiter, "__call__", mock_call)
    
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.RequestError("Connection failed")
    
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None
    
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_context))

    response = client.get("/tiles/fires/0/0/0.pbf")
    assert response.status_code == 502
