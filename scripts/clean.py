"""Remove Python build artifacts and cache directories."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIR_PATTERNS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
FILE_EXTENSIONS = {".pyc", ".pyo", ".pyd"}
GLOB_PATTERNS = ["*.egg-info"]


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def main() -> None:
    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name in DIR_PATTERNS:
            remove_path(path)
        elif path.is_file() and path.suffix in FILE_EXTENSIONS:
            remove_path(path)

    for pattern in GLOB_PATTERNS:
        for match in ROOT.glob(pattern):
            if match.is_dir():
                remove_path(match)


if __name__ == "__main__":
    main()

