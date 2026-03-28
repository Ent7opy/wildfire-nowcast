"""Pytest configuration for api tests.

This configuration file:
1. Adds the workspace root to sys.path to enable cross-module imports
   (api tests need to import from ingest, ml, and other sibling packages)
2. Registers custom pytest marks to eliminate warnings
3. Provides fixtures for integration tests (database schema validation)
"""
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

# Add workspace root to Python path for cross-module imports
# This enables api tests to import from ingest, ml, and other sibling packages
workspace_root = Path(__file__).parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))


@pytest.fixture(autouse=True)
def disable_rate_limiter(monkeypatch):
    """Bypass FastAPI rate limiting in tests to avoid Redis dependency.

    The replacement must mirror the original ``RateLimiter.__call__`` signature
    (``request``, ``response``) so that FastAPI's dependency inspector does not
    treat the parameters as required query-string fields.  Using ``*args`` /
    ``**kwargs`` breaks when ``app.dependency_overrides`` is non-empty because
    FastAPI re-inspects all dependency signatures in that code path.
    """
    from fastapi_limiter.depends import RateLimiter
    from starlette.requests import Request
    from starlette.responses import Response

    async def _allow(self, request: Request, response: Response) -> None:
        return None

    monkeypatch.setattr(RateLimiter, "__call__", _allow)


@pytest.fixture(autouse=True)
def clear_spread_model_catalog_cache():
    """Ensure model catalog env changes are reflected per test."""
    from api.forecast.model_catalog import get_spread_model_catalog

    get_spread_model_catalog.cache_clear()
    yield
    get_spread_model_catalog.cache_clear()



def pytest_configure(config):
    """Register custom pytest marks."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test (runs full pipeline, may be slower)",
    )


@pytest.fixture(scope="session")
def db_available():
    """Check if database is available and skip tests if not.
    
    Usage:
        def test_db_feature(db_available):
            ...
    """
    from api.db import get_engine
    try:
        with get_engine().begin() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


@pytest.fixture(scope="session")
def check_likelihood_schema(db_available):
    """Check if fire likelihood schema columns exist in the database.
    
    Integration tests that use fire likelihood scoring require specific database
    columns: persistence_score, landcover_score, weather_score, false_source_masked,
    and fire_likelihood.
    
    This fixture checks if these columns exist and skips tests if they don't,
    providing a clear message about running migrations.
    
    Usage:
        @pytest.mark.integration
        def test_likelihood_feature(check_likelihood_schema):
            # Test will skip if schema not migrated
            ...
    """
    from api.db import get_engine
    
    required_columns = [
        "persistence_score",
        "landcover_score", 
        "weather_score",
        "false_source_masked",
        "fire_likelihood"
    ]
    
    check_stmt = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'fire_detections' 
        AND column_name = ANY(:columns)
    """)
    
    try:
        with get_engine().begin() as conn:
            result = conn.execute(check_stmt, {"columns": required_columns})
            existing_columns = {row[0] for row in result}
            
        missing_columns = set(required_columns) - existing_columns
        
        if missing_columns:
            pytest.skip(
                f"Database schema missing required columns: {', '.join(sorted(missing_columns))}. "
                f"Run 'make migrate' to apply migrations before running integration tests."
            )
    except Exception as e:
        pytest.skip(
            f"Could not verify database schema: {e}. "
            f"Ensure database is running and migrations are applied with 'make migrate'."
        )
