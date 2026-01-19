"""Pytest configuration for api tests.

This configuration file:
1. Adds the workspace root to sys.path to enable cross-module imports
   (api tests need to import from ingest, ml, and other sibling packages)
2. Registers custom pytest marks to eliminate warnings
"""
import sys
from pathlib import Path

# Add workspace root to Python path for cross-module imports
# This enables api tests to import from ingest, ml, and other sibling packages
workspace_root = Path(__file__).parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))


def pytest_configure(config):
    """Register custom pytest marks."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test (runs full pipeline, may be slower)",
    )
