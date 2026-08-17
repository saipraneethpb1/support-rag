"""Centralized configuration. Everything env-driven, validated at startup."""
from functools import lru_cache
from pydantic import Field, field_validator
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
    qdrant_collection: str = "ask_docs"
    qdrant_api_key: str = ""

    # Cache & registry
    redis_url: str = "redis://localhost:6379/0"
    registry_db_url: str = "sqlite+aiosqlite:///./data/registry/registry.db"

    # Models — API backend defaults match Gemini embeddings used in Docker/Render.
    # For local sentence-transformers, set EMBEDDING_BACKEND=sentence-transformers,
    # EMBEDDING_MODEL=BAAI/bge-small-en-v1.5, EMBEDDING_DIM=384.
    embedding_model: str = "gemini-embedding-001"
    embedding_backend: str = "api"  # "api" | "sentence-transformers" | "fastembed"
    embedding_dim: int = 768
    hf_token: str = ""  # required only when api backend falls back to HuggingFace
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_enabled: bool = False

    # Chunking
    chunk_size_tokens: int = Field(default=500, ge=128, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=400)
    max_chunk_tokens: int = Field(default=800, ge=256, le=4000)

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # API / security
    api_key: str = "local-dev-key"
    webhook_secret: str = ""  # falls back to api_key when empty
    cors_origins: str = "*"  # comma-separated origins, or "*"
    log_level: str = "INFO"
    env: str = "development"  # "development" | "production"

    # Background ingest poller (off by default; prefer webhooks + POST /ingest/run)
    poller_enabled: bool = False
    poller_interval_seconds: int = Field(default=300, ge=30)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _strip_cors(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if not raw or raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    def resolved_webhook_secret(self) -> str:
        return self.webhook_secret or self.api_key

    def effective_embedding_model_id(self) -> str:
        """Model id used for cache keys and logging (matches the encoder that runs)."""
        if self.embedding_backend == "api" and self.google_api_key:
            return "gemini-embedding-001"
        return self.embedding_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
