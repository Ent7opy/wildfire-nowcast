import logging
import uuid

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from redis.asyncio import Redis

from api.config import settings
from api.errors import ErrorResponse, WildfireError, wildfire_error_handler
from api.logging_config import setup_logging, request_id_ctx
from api.routes import archive_router, assistant_router, internal_router, fires_router, forecast_router, aois_router, tiles_router, exports_router, risk_router, ignition_router
from api.startup_check import StartupError, run_api_startup_checks

# Configure structured JSON logging before any log statements.
setup_logging()

LOGGER = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.version)

cors_allow_origins = [
    origin.strip()
    for origin in settings.cors_allow_origins.split(",")
    if origin.strip()
]

# Explicit allowlists for CORS methods and headers — wildcards would permit
# non-standard methods (TRACE, CONNECT) and arbitrary headers, broadening the
# attack surface.  Only list what the app actually needs.  (Fixes #297)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)


# Pre-computed once at startup — settings are static.
_CSP_HEADER = f"frame-ancestors {settings.frame_ancestors}"


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    """Attach a request_id to every request for log correlation.

    If the caller supplies an ``X-Request-ID`` header it is reused; otherwise a
    new UUID4 is generated.  The id is stored in a contextvar so the JSON log
    formatter can include it automatically, and echoed back in the response.
    """
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_ctx.set(rid)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        request_id_ctx.reset(token)


@app.middleware("http")
async def _add_csp_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = _CSP_HEADER
    return response

@app.on_event("startup")
async def startup():
    try:
        run_api_startup_checks(settings)
    except StartupError as exc:
        LOGGER.critical("STARTUP CONFIG ERROR: %s", exc)
        raise SystemExit(f"Startup check failed: {exc}") from exc

    try:
        redis = Redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        # Test connection with a simple ping
        await redis.ping()
        await FastAPILimiter.init(redis)
        LOGGER.info("Redis connection established and rate limiter initialized")
    except Exception as e:
        LOGGER.warning(
            "Redis connection failed; rate limiting disabled. Error: %s",
            e,
            extra={"redis_host": settings.redis_host, "redis_port": settings.redis_port},
        )
        # Graceful degradation: rate limiting is disabled, but app continues
        # Store None to indicate rate limiter is not available
        app.state.redis = None
        app.state.limiter_disabled = True
        # Patch RateLimiter so routes don't crash when Redis is unavailable
        async def _noop_rate_limiter(self, *args, **kwargs):
            return None
        RateLimiter.__call__ = _noop_rate_limiter


@app.on_event("shutdown")
async def shutdown():
    from api.db import dispose_async_engine

    await dispose_async_engine()


app.add_exception_handler(WildfireError, wildfire_error_handler)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=str(exc.status_code),
            message=exc.detail,
        ).model_dump(),
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            code="validation_error",
            message="Invalid request parameters",
            details=exc.errors(),
        ).model_dump(),
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Only include internal details in development environment
    # Use explicit check to prevent information leakage in production
    include_details = settings.environment.lower() in ("dev", "development", "local", "debug")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            code="internal_error",
            message="Internal Server Error",
            details=str(exc) if include_details else None,
        ).model_dump(),
    )

app.include_router(archive_router)
app.include_router(assistant_router)
app.include_router(internal_router)
app.include_router(fires_router)
app.include_router(forecast_router)
app.include_router(aois_router)
app.include_router(tiles_router)
app.include_router(exports_router)
app.include_router(risk_router)
app.include_router(ignition_router, prefix="/ignition", tags=["ignition"])

