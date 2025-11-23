from __future__ import annotations

import os
import subprocess
from importlib import metadata
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


def _get_project_version() -> str:
    try:
        return metadata.version("wildfire-nowcast-api")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _get_git_commit() -> str:
    if (value := os.getenv("GIT_COMMIT")):
        return value

    repo_root = Path(__file__).resolve().parent.parent
    if (git_dir := repo_root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            pass

    return "unknown"


class AppSettings(BaseSettings):
    app_name: str = "Wildfire Nowcast API"
    version: str = Field(default_factory=_get_project_version)
    environment: str = Field(default="dev", env="APP_ENV")
    git_commit: str = Field(default_factory=_get_git_commit, env="GIT_COMMIT")

    # Database settings
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="wildfire", env="POSTGRES_USER")
    postgres_password: str = Field(default="wildfire", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="wildfire", env="POSTGRES_DB")

    @property
    def database_url(self) -> str:
        """Construct database URL from individual components."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        case_sensitive = False


settings = AppSettings()

