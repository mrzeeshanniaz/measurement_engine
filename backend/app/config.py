from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEBUG: bool = False
    DEVICE: str = "cpu"
    MODEL_CACHE_DIR: str = "./models"

    DEFAULT_FOCAL_LENGTH_MM: float = 4.25
    DEFAULT_SENSOR_WIDTH_MM: float = 4.8

    class Config:
        env_file = ".env"


settings = Settings()
