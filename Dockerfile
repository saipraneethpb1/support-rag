# Ultra-light build for Render free tier (512MB RAM).
# No torch, no sentence-transformers, no fastembed.
# Embeddings via Google Gemini API (EMBEDDING_BACKEND=api).
# Reranker disabled. Total RAM ~100MB.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_BACKEND=api \
    EMBEDDING_MODEL=gemini-embedding-001 \
    EMBEDDING_DIM=768 \
    RERANKER_ENABLED=false \
    POLLER_ENABLED=false

COPY pyproject.toml requirements.lock.txt README.md ./
COPY api ./api
COPY cache ./cache
COPY config ./config
COPY core ./core
COPY generation ./generation
COPY ingestion ./ingestion
COPY observability ./observability
COPY retrieval ./retrieval
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY data ./data

RUN pip install --upgrade pip \
    && pip install -r requirements.lock.txt \
    && pip install --no-deps .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
