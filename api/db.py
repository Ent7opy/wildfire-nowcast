"""Database connection and SQLAlchemy engine setup."""

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker

from api.config import settings

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine, creating it on first call.

    Note: under Gunicorn with multiple worker *processes*, each process gets
    its own pool.  Set DB_POOL_SIZE small (e.g. 2-5) so that
    total_connections = workers × pool_size stays within your DB limit.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_pool_max_overflow,
            pool_recycle=settings.db_pool_recycle_seconds,
            pool_pre_ping=True,
            echo=settings.environment == "dev",
        )
    return _engine


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
