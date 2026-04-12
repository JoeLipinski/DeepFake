from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "production"
    max_upload_mb: int = 20
    job_ttl_hours: int = 24
    workers: int = 1

    depth_model_id: str = "depth-anything/Depth-Anything-V2-Large-hf"
    realesrgan_model_url: str = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    )

    # Comma-separated origins for CORS
    cors_origins: str = "http://localhost:80,http://localhost:5173"

    # Local filesystem path for job artifacts
    jobs_dir: str = "/app/jobs"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
