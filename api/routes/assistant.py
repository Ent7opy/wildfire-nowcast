"""Assistant proxy router — forwards chat requests to Gemini server-side."""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.config import settings

assistant_router = APIRouter(prefix="/assistant", tags=["assistant"])


class AssistantConfigResponse(BaseModel):
    configured: bool
    model: str


@assistant_router.get("/config", response_model=AssistantConfigResponse)
async def get_assistant_config() -> AssistantConfigResponse:
    return AssistantConfigResponse(
        configured=bool(settings.gemini_api_key),
        model=settings.gemini_model,
    )


@assistant_router.post("/chat")
async def proxy_chat(body: dict) -> dict:
    """Proxy a Gemini generateContent request, injecting the server-side API key."""
    if not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="Assistant not configured")

    url = (
        f"{settings.gemini_api_base_url}/models/"
        f"{settings.gemini_model}:generateContent"
        f"?key={settings.gemini_api_key}"
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=body)

    if not resp.is_success:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return resp.json()
