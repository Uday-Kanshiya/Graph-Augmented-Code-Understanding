from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "Context Optimization Engine"
    api_prefix: str = "/api"
    data_dir: Path = Field(default=ROOT_DIR / "data", alias="CONTEXT_ENGINE_DATA_DIR")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    graphify_timeout_seconds: int = Field(default=120, alias="GRAPHIFY_TIMEOUT_SECONDS")
    max_upload_mb: int = Field(default=200, alias="MAX_UPLOAD_MB")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        alias="CORS_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("data_dir", mode="after")
    @classmethod
    def resolve_data_dir(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return (ROOT_DIR / value).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
