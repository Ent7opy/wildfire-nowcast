"""Remove Python build artifacts and cache directories."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIR_PATTERNS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
FILE_EXTENSIONS = {".pyc", ".pyo", ".pyd"}
GLOB_PATTERNS = ["*.egg-info"]
VENV_DIRS = [ROOT / "api" / ".venv", ROOT / "ui" / ".venv", ROOT / "ml" / ".venv", ROOT / "ingest" / ".venv"]


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove Python build artifacts and cache directories")
    parser.add_argument("--include-venv", action="store_true", help="Also remove .venv directories")
    args = parser.parse_args()

    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name in DIR_PATTERNS:
            remove_path(path)
        elif path.is_file() and path.suffix in FILE_EXTENSIONS:
            remove_path(path)

    for pattern in GLOB_PATTERNS:
        for match in ROOT.glob(pattern):
            if match.is_dir():
                remove_path(match)
    
    if args.include_venv:
        for venv_dir in VENV_DIRS:
            if venv_dir.exists():
                remove_path(venv_dir)


if __name__ == "__main__":
    main()

