# Run it locally

You need Python 3.11+, Docker (for Qdrant and Redis), and at least one
LLM key (Groq recommended). Embeddings default to Google Gemini.

## Steps

1. `docker compose up -d qdrant redis`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -e ".[dev]"`
4. `cp .env.example .env` and set `GROQ_API_KEY` and `GOOGLE_API_KEY`
5. `python -m scripts.seed_demo_data` (optional; refreshes example files)
6. `python -m scripts.bootstrap_index`
7. `uvicorn api.main:app --reload`

Health check: `GET /health`. Chat UI: `GET /`.

## Free-tier deploy

The Docker image keeps the reranker off (`RERANKER_ENABLED=false`) so
it fits small hosts. Point `QDRANT_URL` and `REDIS_URL` at hosted
services if you deploy the web process alone.
