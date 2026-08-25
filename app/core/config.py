from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    APP_NAME: str = "Speech Therapy AI API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Laravel integration
    LARAVEL_API_BASE_URL: str = "http://your-laravel-app/api"
    LARAVEL_API_TOKEN: str = ""
    LARAVEL_TIMEOUT: float = 10.0

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()