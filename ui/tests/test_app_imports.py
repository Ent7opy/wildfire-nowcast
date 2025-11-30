import importlib


def test_main_entrypoint_is_callable() -> None:
    """Smoke test to ensure the Streamlit app module loads."""
    app_module = importlib.import_module("app")
    assert hasattr(app_module, "main")
    assert callable(app_module.main)

