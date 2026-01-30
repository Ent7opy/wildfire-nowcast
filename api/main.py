import os
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from redis.asyncio import Redis

from api.config import settings
from api.errors import ErrorResponse
from api.routes import internal_router, fires_router, forecast_router, aois_router, tiles_router, exports_router, risk_router

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

@app.on_event("startup")
async def startup():
    redis = Redis.from_url(f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis)

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

app.include_router(internal_router)
app.include_router(fires_router)
app.include_router(forecast_router)
app.include_router(aois_router)
app.include_router(tiles_router)
app.include_router(exports_router)
app.include_router(risk_router)

