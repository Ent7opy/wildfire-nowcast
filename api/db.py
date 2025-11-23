"""Database connection and SQLAlchemy engine setup."""

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker

from .config import settings


def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine for the database."""
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,  # Verify connections before use
        echo=settings.environment == "dev",  # Log SQL in development
    )


# Create a session factory for ORM operations
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
