"""Tests for forecast raster tile proxy and colormap."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.forecast import FIRE_PROBABILITY_COLORMAP, forecast_router

app = FastAPI()
app.include_router(forecast_router)
client = TestClient(app)

MOCK_RASTER = {
    "horizon_hours": 24,
    "file_format": "COG",
    "storage_path": "data/forecasts/test_region/run_42/spread_h024_cog.tif",
}

_PNG_BYTES = b"\x89PNG\r\n\x1a\n"  # minimal PNG magic bytes


def _mock_titiler_ok(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch httpx.AsyncClient to return a 200 PNG response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = _PNG_BYTES

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp

    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None

    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_context))
    return mock_client


# ---------------------------------------------------------------------------
# Tile endpoint – success and error paths
# ---------------------------------------------------------------------------

def test_get_forecast_tile_success(monkeypatch):
    """GET /forecast/{run_id}/tiles/{z}/{x}/{y}.png returns PNG from TiTiler."""
    from fastapi_limiter.depends import RateLimiter
    async def _noop(*a, **kw): ...
    monkeypatch.setattr(RateLimiter, "__call__", _noop)

    monkeypatch.setattr("api.forecast.repo.get_raster_for_run", lambda run_id, h: MOCK_RASTER)
    mock_client = _mock_titiler_ok(monkeypatch)

    response = client.get("/forecast/42/tiles/5/16/12.png?horizon_hours=24")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == _PNG_BYTES

    # TiTiler must be called with rescale and colormap
    args, kwargs = mock_client.get.call_args
    assert "WebMercatorQuad/5/16/12.png" in args[0]
    assert kwargs["params"]["rescale"] == "0,1"
    colormap = json.loads(kwargs["params"]["colormap"])
    assert colormap["0"][3] == 0  # transparent at 0


def test_get_forecast_tile_default_horizon(monkeypatch):
    """When horizon_hours is omitted, defaults to 24."""
    from fastapi_limiter.depends import RateLimiter
    async def _noop(*a, **kw): ...
    monkeypatch.setattr(RateLimiter, "__call__", _noop)

    captured = {}

    def _get_raster(run_id, horizon_hours):
        captured["horizon_hours"] = horizon_hours
        return MOCK_RASTER

    monkeypatch.setattr("api.forecast.repo.get_raster_for_run", _get_raster)
    _mock_titiler_ok(monkeypatch)

    response = client.get("/forecast/42/tiles/5/16/12.png")

    assert response.status_code == 200
    assert captured["horizon_hours"] == 24


def test_get_forecast_tile_run_not_found(monkeypatch):
    """Returns 404 when no raster exists for the given run/horizon."""
    from fastapi_limiter.depends import RateLimiter
    async def _noop(*a, **kw): ...
    monkeypatch.setattr(RateLimiter, "__call__", _noop)

    monkeypatch.setattr("api.forecast.repo.get_raster_for_run", lambda *a: None)

    response = client.get("/forecast/99/tiles/5/16/12.png")

    assert response.status_code == 404


def test_get_forecast_tile_titiler_unreachable(monkeypatch):
    """Returns 502 when TiTiler cannot be reached."""
    from fastapi_limiter.depends import RateLimiter
    async def _noop(*a, **kw): ...
    monkeypatch.setattr(RateLimiter, "__call__", _noop)

    monkeypatch.setattr("api.forecast.repo.get_raster_for_run", lambda *a: MOCK_RASTER)

    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.RequestError("connection refused")
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_context))

    response = client.get("/forecast/42/tiles/5/16/12.png")

    assert response.status_code == 502


def test_get_forecast_tile_titiler_error(monkeypatch):
    """Returns 502 when TiTiler returns a non-200 status."""
    from fastapi_limiter.depends import RateLimiter
    async def _noop(*a, **kw): ...
    monkeypatch.setattr(RateLimiter, "__call__", _noop)

    monkeypatch.setattr("api.forecast.repo.get_raster_for_run", lambda *a: MOCK_RASTER)

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_context))

    response = client.get("/forecast/42/tiles/5/16/12.png")

    assert response.status_code == 502


# ---------------------------------------------------------------------------
# Colormap contract
# ---------------------------------------------------------------------------

def test_colormap_zero_is_transparent():
    """Probability 0 must map to fully transparent (alpha = 0)."""
    assert FIRE_PROBABILITY_COLORMAP["0"][3] == 0


def test_colormap_max_is_red():
    """Max value (255) must be a strong red (R > G and R > B)."""
    rgba = FIRE_PROBABILITY_COLORMAP["255"]
    assert rgba[0] > rgba[1], "Red channel must dominate green at max probability"
    assert rgba[0] > rgba[2], "Red channel must dominate blue at max probability"


def test_colormap_gradient_alpha_increases():
    """Alpha must increase monotonically from low to high probability."""
    sorted_keys = sorted(FIRE_PROBABILITY_COLORMAP.keys(), key=int)
    alphas = [FIRE_PROBABILITY_COLORMAP[k][3] for k in sorted_keys]
    for i in range(1, len(alphas)):
        assert alphas[i] >= alphas[i - 1], (
            f"Alpha must not decrease: key {sorted_keys[i - 1]}→{sorted_keys[i]}"
        )


# ---------------------------------------------------------------------------
# tile_url field in /forecast response
# ---------------------------------------------------------------------------

def test_tile_url_in_get_forecast_response(monkeypatch):
    """GET /forecast must include tile_url in each raster entry."""
    import json as _json
    from unittest.mock import patch

    mock_run = {
        "id": 101,
        "region_name": "balkans",
        "status": "completed",
        "model_name": "TestModel",
        "model_version": "v1",
        "forecast_reference_time": "2025-01-01T00:00:00+00:00",
        "metadata": {},
        "bbox_geojson": _json.dumps({
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }),
    }
    mock_rasters = [
        {
            "horizon_hours": 24,
            "file_format": "COG",
            "storage_path": "data/forecasts/balkans/run_101/spread_h024_cog.tif",
        }
    ]

    with patch("api.forecast.repo.get_latest_forecast_run", return_value=mock_run), \
         patch("api.forecast.repo.list_rasters_for_run", return_value=mock_rasters), \
         patch("api.forecast.repo.list_contours_for_run", return_value=[]):
        response = client.get(
            "/forecast",
            params={"region_name": "balkans", "min_lon": 0, "min_lat": 0, "max_lon": 1, "max_lat": 1},
        )

    assert response.status_code == 200
    raster = response.json()["rasters"][0]
    assert "tile_url" in raster
    tile_url = raster["tile_url"]
    assert "/forecast/101/tiles/" in tile_url
    assert "{z}" in tile_url
    assert "{x}" in tile_url
    assert "{y}" in tile_url
    assert "horizon_hours=24" in tile_url
