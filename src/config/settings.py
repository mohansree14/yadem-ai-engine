"""
YADEM Application Settings
Reads from environment variables / .env file using pydantic-settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import json


class Settings(BaseSettings):
    """Global application settings loaded from environment."""

    # --- Application ---
    app_name: str = "YADEM AI Credit Decisioning Engine"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: str = '["http://localhost:3000","http://localhost:8080"]'

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.cors_origins)

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///./yadem.db"
    database_sync_url: str = "sqlite:///./yadem.db"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- MLflow ---
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "yadem-credit-scoring"

    # --- Model ---
    model_registry_path: str = "./models"
    ensemble_weights_lr: float = 0.25
    ensemble_weights_rf: float = 0.35
    ensemble_weights_xgb: float = 0.40

    # --- Risk Bands ---
    risk_band_a_min: int = 800
    risk_band_b_min: int = 650
    risk_band_c_min: int = 500
    risk_band_d_min: int = 300

    # --- Security ---
    aes_encryption_key: str = "change-me-32-byte-key-12345678"
    jwt_secret_key: str = "change-me-jwt-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
settings = Settings()
