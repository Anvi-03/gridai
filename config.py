"""
GridPulse AI — Configuration
Environment-driven settings using Pydantic BaseSettings.
All sensitive values are loaded from a .env file or OS environment variables.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME: str = "GridPulse AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Database ──────────────────────────────────────────────────────────────
    # Full async DSN — asyncpg dialect required for non-blocking I/O.
    # Example: postgresql+asyncpg://user:password@localhost:5432/gridpulse
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/gridpulse"

    # Connection-pool tuning (tweak per hardware / load profile)
    DB_POOL_SIZE: int = 10           # persistent connections kept alive
    DB_MAX_OVERFLOW: int = 20        # extra connections allowed under spike load
    DB_POOL_RECYCLE: int = 3600      # recycle connections every 1 hour (secs)
    DB_POOL_TIMEOUT: int = 30        # seconds to wait for a free pool slot

    # ── API ───────────────────────────────────────────────────────────────────
    API_PREFIX: str = "/api/v1"
    TELEMETRY_ENDPOINT: str = "/telemetry"

    # Comma-separated list of allowed CORS origins.
    # Wildcard ("*") is acceptable for local dev; restrict in production.
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── GenAI Copilot ─────────────────────────────────────────────────────────
    # Gemini API key — obtain from https://aistudio.google.com/app/apikey
    GEMINI_API_KEY: str = ""
    # Model name — see https://ai.google.dev/models/gemini for options
    GEMINI_MODEL:   str = "gemini-2.5-flash"

    # ── Ingestion / Batching ──────────────────────────────────────────────────
    # Maximum records accepted in a single POST body.
    MAX_BATCH_SIZE: int = 500

    # ── Simulator (read by simulator.py) ─────────────────────────────────────
    SIMULATOR_TARGET_URL: str = "http://localhost:8000/api/v1/telemetry"
    SIMULATOR_NUM_METERS: int = 20      # concurrent virtual meters
    SIMULATOR_INTERVAL_S: float = 0.5   # seconds between each meter's burst
    SIMULATOR_BATCH_SIZE: int = 10      # readings per POST from each meter

    # ── Authentication / JWT ──────────────────────────────────────────────────
    # CRITICAL: Override JWT_SECRET_KEY in production with a long random string.
    # Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
    JWT_SECRET_KEY: str = "gridpulse-dev-secret-change-in-production"
    JWT_ALGORITHM:  str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60        # access token lifetime in minutes

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton — avoids re-parsing env on every request."""
    return Settings()


# Convenience alias used throughout the codebase.
settings: Settings = get_settings()
