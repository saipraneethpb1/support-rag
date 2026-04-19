FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 EMBEDDING_BACKEND=fastembed RERANKER_ENABLED=false
RUN pip install --upgrade pip && pip install \
    pydantic pydantic-settings python-dotenv structlog tenacity httpx \
    beautifulsoup4 lxml markdown-it-py fastembed \
    qdrant-client rank-bm25 groq google-generativeai \
    fastapi "uvicorn[standard]" redis sqlalchemy aiosqlite
COPY . .
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]