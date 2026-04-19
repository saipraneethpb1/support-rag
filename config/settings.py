"""Centralized configuration. Everything env-driven, validated at startup."""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "support_docs"
    qdrant_api_key: str = ""

    # Cache & registry
    redis_url: str = "redis://localhost:6379/0"
    registry_db_url: str = "sqlite+aiosqlite:///./data/registry/registry.db"

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_backend: str = "sentence-transformers"  # or "fastembed" for low-RAM
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_enabled: bool = True
    embedding_dim: int = 384

    # Chunking
    chunk_size_tokens: int = Field(default=500, ge=128, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=400)
    max_chunk_tokens: int = Field(default=800, ge=256, le=4000)

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # API
    api_key: str = "local-dev-key"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
