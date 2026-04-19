# Ultra-light build for Render free tier (512MB RAM).
# No torch, no sentence-transformers, no fastembed.
# Embeddings come from HuggingFace Inference API (free, no key).
# Reranker disabled. Total RAM ~100MB.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_BACKEND=api \
    RERANKER_ENABLED=false

RUN pip install --upgrade pip && pip install \
    pydantic pydantic-settings python-dotenv structlog tenacity httpx \
    beautifulsoup4 lxml markdown-it-py \
    qdrant-client rank-bm25 \
    groq google-generativeai \
    fastapi "uvicorn[standard]" \
    redis sqlalchemy aiosqlite

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
