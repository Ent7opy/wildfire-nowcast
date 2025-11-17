from fastapi import FastAPI

from .config import settings
from .routes import internal_router

app = FastAPI(title=settings.app_name, version=settings.version)

app.include_router(internal_router)

