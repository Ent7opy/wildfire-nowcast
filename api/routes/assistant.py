"""Assistant proxy router — forwards chat requests to Gemini server-side."""

import logging
import threading
import time
from enum import Enum

import httpx
from fastapi import APIRouter, Depends, HTTPException

from api.deps import no_cache
from pydantic import BaseModel

from api.config import settings

LOGGER = logging.getLogger(__name__)

assistant_router = APIRouter(prefix="/assistant", tags=["assistant"])


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _CircuitBreaker:
    """Thread-safe circuit breaker for Gemini API calls.

    States:
      CLOSED   — normal operation; failures accumulate toward threshold.
      OPEN     — short-circuit; requests fail immediately until cooldown expires.
      HALF_OPEN — one probe request allowed; success closes, failure re-opens.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None

    def _compute_state(self) -> CircuitState:
        """Must be called with _lock held."""
        if self._opened_at is None:
            return CircuitState.CLOSED
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= settings.gemini_circuit_breaker_cooldown_seconds:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._compute_state()

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failures

    @property
    def cooldown_remaining_seconds(self) -> float | None:
        with self._lock:
            if self._opened_at is None:
                return None
            state = self._compute_state()
            if state != CircuitState.OPEN:
                return None
            elapsed = time.monotonic() - self._opened_at
            return max(0.0, settings.gemini_circuit_breaker_cooldown_seconds - elapsed)

    def snapshot(self) -> tuple[CircuitState, int, float | None]:
        """Return (state, failure_count, cooldown_remaining) under a single lock acquisition."""
        with self._lock:
            state = self._compute_state()
            failures = self._failures
            if self._opened_at is None or state != CircuitState.OPEN:
                cooldown: float | None = None
            else:
                elapsed = time.monotonic() - self._opened_at
                cooldown = max(0.0, settings.gemini_circuit_breaker_cooldown_seconds - elapsed)
        return state, failures, cooldown

    def allow_request(self) -> bool:
        """Return True if a request should proceed (CLOSED or HALF_OPEN probe)."""
        with self._lock:
            return self._compute_state() != CircuitState.OPEN

    def record_success(self) -> None:
        with self._lock:
            was_open = self._opened_at is not None
            self._failures = 0
            self._opened_at = None
        if was_open:
            LOGGER.info("gemini circuit breaker: closed after successful probe")

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            failures = self._failures
            threshold = settings.gemini_circuit_breaker_threshold
            if self._opened_at is not None:
                # Already open/half-open — reset the cooldown timer on each failure.
                self._opened_at = time.monotonic()
                log_args: tuple = ("gemini circuit breaker: failure during open/half-open, cooldown reset",)
            elif failures >= threshold:
                self._opened_at = time.monotonic()
                log_args = ("gemini circuit breaker opened after %d consecutive failures", failures)
            else:
                log_args = ("gemini circuit breaker: failure %d/%d", failures, threshold)
        LOGGER.warning(*log_args)


# Module-level singleton — shared across all requests within a worker process.
_circuit = _CircuitBreaker()


class AssistantConfigResponse(BaseModel):
    configured: bool
    model: str


class AssistantHealthResponse(BaseModel):
    configured: bool
    circuit_state: CircuitState
    failure_count: int
    cooldown_remaining_seconds: float | None


@assistant_router.get("/config", response_model=AssistantConfigResponse)
async def get_assistant_config() -> AssistantConfigResponse:
    return AssistantConfigResponse(
        configured=bool(settings.gemini_api_key),
        model=settings.gemini_model,
    )


@assistant_router.get("/health", response_model=AssistantHealthResponse)
async def get_assistant_health() -> AssistantHealthResponse:
    state, failure_count, cooldown = _circuit.snapshot()
    return AssistantHealthResponse(
        configured=bool(settings.gemini_api_key),
        circuit_state=state,
        failure_count=failure_count,
        cooldown_remaining_seconds=cooldown,
    )


@assistant_router.post("/chat", dependencies=[Depends(no_cache)])
async def proxy_chat(body: dict) -> dict:
    """Proxy a Gemini generateContent request, injecting the server-side API key."""
    if not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="Assistant not configured")

    if not _circuit.allow_request():
        LOGGER.warning("gemini circuit breaker is open — short-circuiting request")
        raise HTTPException(
            status_code=503,
            detail="Assistant temporarily unavailable (circuit open)",
        )

    url = (
        f"{settings.gemini_api_base_url}/models/"
        f"{settings.gemini_model}:generateContent"
    )

    try:
        async with httpx.AsyncClient(timeout=settings.gemini_timeout_seconds) as client:
            resp = await client.post(
                url,
                json=body,
                headers={"x-goog-api-key": settings.gemini_api_key},
            )
    except Exception as exc:
        LOGGER.error("gemini request raised exception: %s", exc)
        _circuit.record_failure()
        raise HTTPException(status_code=503, detail="Assistant request failed") from exc

    if not resp.is_success:
        LOGGER.error(
            "gemini returned non-success status %d: %.200s", resp.status_code, resp.text
        )
        _circuit.record_failure()
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    _circuit.record_success()
    return resp.json()
