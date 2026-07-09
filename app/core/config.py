"""
app/core/config.py

Central configuration for the whole application.

HOW IT WORKS:
  - pydantic-settings reads every variable from .env
  - each becomes a typed attribute (a wrong type fails loudly at startup,
    not mysteriously at runtime)
  - any code that needs a setting calls get_settings()
  - @lru_cache means .env is read once and cached

WHY NOT just use os.getenv() everywhere:
  - type safety: a missing DATABASE_URL is caught immediately
  - one place to look: every setting is defined here
  - testability: tests can override settings without editing .env
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database
    database_url: str = "postgresql+asyncpg://shm_user:shm_pass@localhost:5432/shm_db"
    database_sync_url: str = "postgresql://shm_user:shm_pass@localhost:5432/shm_db"

    # AWS
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket_name: str = "shm-drone-images"
    sqs_queue_url: str = ""

    # ML models
    yolov8_weights_path: Path = Path("ml/weights/yolov8_codebrim.pt")
    yolov9_weights_path: Path = Path("ml/weights/yolov9_codebrim.pt")
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    mlflow_tracking_uri: str = "http://localhost:5000"

    # ACI 318 severity thresholds
    severity_low_max: float = 0.50
    severity_moderate_max: float = 0.75

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()