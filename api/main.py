import logging
import os
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from redis.asyncio import Redis

from api.config import settings
from api.errors import ErrorResponse
from api.routes import archive_router, assistant_router, internal_router, fires_router, forecast_router, aois_router, tiles_router, exports_router, risk_router
from api.startup_check import StartupError, run_api_startup_checks

LOGGER = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.version)

cors_allow_origins = [
    origin.strip()
    for origin in settings.cors_allow_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pre-computed once at startup — settings are static.
_CSP_HEADER = f"frame-ancestors {settings.frame_ancestors}"


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
            f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}",
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
            extra={"redis_host": os.getenv("REDIS_HOST", "localhost"), "redis_port": os.getenv("REDIS_PORT", "6379")},
        )
        # Graceful degradation: rate limiting is disabled, but app continues
        # Store None to indicate rate limiter is not available
        app.state.redis = None
        app.state.limiter_disabled = True
        # Patch RateLimiter so routes don't crash when Redis is unavailable
        async def _noop_rate_limiter(self, *args, **kwargs):
            return None
        RateLimiter.__call__ = _noop_rate_limiter

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

