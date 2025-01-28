# File: /Users/deangladish/tikaPOC/src/services/config.py

from pydantic_settings import BaseSettings
from pydantic import Field, AnyHttpUrl
from typing import Optional  # <-- Import Optional here

class Settings(BaseSettings):
    db_host: str
    db_name: str
    db_user: str
    db_password: str
    db_port: int = 5432
    ssl_root_cert: str
    azure_openai_endpoint: AnyHttpUrl
    azure_openai_key: str
    azure_openai_deployment: str
    vault_url: AnyHttpUrl
    ieee_api_key: Optional[str] = None
    acm_api_key: Optional[str] = None
    ams_api_key: Optional[str] = None

    class Config:
        env_file = ".env"

settings = Settings()
