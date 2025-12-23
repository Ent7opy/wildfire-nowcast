from fastapi import FastAPI

from api.config import settings
from api.routes import internal_router

app = FastAPI(title=settings.app_name, version=settings.version)

app.include_router(internal_router)

