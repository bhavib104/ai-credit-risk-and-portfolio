from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    app_name: str = "credit-risk-decisioning"
    env: str = "dev"
    database_url: str = "sqlite:///./credit_risk.db"
    artifacts_dir: str = "./artifacts"
    data_dir: str = "./data"
    random_seed: int = 42
    log_level: str = "INFO"
    class Config:
        env_file = ".env"

settings = Settings()
ROOT = Path(__file__).resolve().parent.parent
