from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    chroma_dir: str = "./data/chroma"
    collection_name: str = "pdf_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 150
    max_upload_mb: int = 15

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()