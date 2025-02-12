from pydantic_settings import BaseSettings
from typing import List
import semver

class Settings(BaseSettings):
    # API
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # OpenTelemetry
    OTEL_SERVICE_NAME: str = "i2-bridge"
    OTEL_COLLECTOR_URL: str = "http://otel-collector:4317"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/i2bridge"
    
    @property
    def semver(self) -> semver.VersionInfo:
        return semver.VersionInfo.parse(self.VERSION)
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 