"""Environment-backed settings.

Database connectivity is not implemented yet. `database_url` exists so
PostgreSQL can be wired in later without changing the configuration shape.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str | None = None
    data_dir: Path = Path("data")

    bgg_base_url: str = "https://boardgamegeek.com/xmlapi2"
    bgg_token: str | None = None
    bgg_timeout_seconds: float = Field(default=30.0, gt=0)
    bgg_rate_limit_seconds: float = Field(default=5.0, ge=0)
    bgg_max_retries: int = Field(default=4, ge=1)
    bgg_retry_backoff_seconds: float = Field(default=2.0, ge=0)
    http_user_agent: str = (
        "board-game-analysis/0.1.0 "
        "(research ingestion; https://github.com/rychen1/board-game-analysis)"
    )
