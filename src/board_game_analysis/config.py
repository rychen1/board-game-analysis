"""Environment-backed settings.

Database connectivity is not implemented yet. `database_url` exists so
PostgreSQL can be wired in later without changing the configuration shape.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str | None = None
