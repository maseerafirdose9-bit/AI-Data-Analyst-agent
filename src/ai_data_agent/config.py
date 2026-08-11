# src/ai_data_agent/config.py

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Centralized application configuration.
    Values are loaded from environment variables, falling back to
    a local .env file if present, then to the defaults below.
    """

    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    llm_model_name: str = Field(default="gemini-2.5-flash", alias="LLM_MODEL_NAME")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        populate_by_name = True


# A single shared instance, imported everywhere else in the app
settings = Settings()