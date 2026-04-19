# Support RAG — Production-grade customer support chatbot

A full RAG system for customer support over product docs, help articles,
resolved tickets, changelog, and API reference. Built step-by-step as a
portfolio demonstration of production RAG engineering.

## What's in the box

- **Hybrid retrieval**: vector (Qdrant, bge-small) + BM25, fused via RRF, reranked by a cross-encoder
- **Real-time ingestion**: content-hash diffing in a doc registry means only changed docs are re-embedded. Push (webhooks) + pull (poller).
- **Multi-source connectors**: markdown docs, help-center HTML, tickets (JSONL), changelog, OpenAPI
- **Structure-aware chunking**: splits on H1/H2/H3, prepends title + heading path
- **LLM router**: Groq primary → Gemini fallback, retries, exponential backoff, circuit breaker
- **Citation audit**: flags when the LLM cites markers not present in context — strong hallucination signal, exposed in metrics
- **Streaming**: `/chat/stream` returns SSE tokens live, then a final meta event with citations
- **Semantic cache**: near-duplicate queries return cached answers; invalidated on corpus version bump
- **Observability**: Langfuse tracing (optional), structured logging, per-stage timings
- **Evaluation**:
  - Retrieval: Hit@K / MRR / nDCG / Recall across 3 retriever configs
  - Answer: faithfulness, relevancy, hallucination rate (LLM-as-judge)
- **35 unit + integration tests**

Everything runs on free tiers.

## Local setup

```bash
# 1. Start infra
docker compose up -d qdrant redis langfuse

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
# Edit .env: set GROQ_API_KEY (free at console.groq.com)

# 4. Seed demo data + bootstrap index
python -m scripts.seed_demo_data
python -m scripts.bootstrap_index

# 5. Run API
uvicorn api.main:app --reload
# Open http://localhost:8000 for the demo UI
```

## Endpoints

| Method | Path                         | Auth    | Description                         |
|--------|------------------------------|---------|-------------------------------------|
| GET    | `/`                          | none    | Minimal chat UI                     |
| GET    | `/health`                    | none    | Dependency health                   |
| POST   | `/chat`                      | API key | Blocking chat (uses semantic cache) |
| POST   | `/chat/stream`               | API key | SSE streaming chat                  |
| POST   | `/ingest/run`                | API key | Force a full ingest pass            |
| POST   | `/webhooks/tickets/resolved` | secret  | Single-ticket ingest on push        |
| POST   | `/webhooks/docs/updated`     | secret  | Trigger docs re-ingest on CI build  |
| GET    | `/docs`                      | none    | OpenAPI interactive docs            |

## Folder structure

```
support-rag/
├── api/                    # FastAPI app, routes, auth, rate limit
├── config/                 # Settings + editable prompt templates
├── ingestion/              # Connectors, chunker, embedder, registry, pipeline
├── retrieval/              # Vector, BM25, hybrid, rerank, transforms, facade
├── generation/             # LLM router, prompt, citation, generator
├── cache/                  # Embedding + semantic caches
├── observability/          # Langfuse + structlog
├── evaluation/             # Retrieval + answer eval harnesses, golden QA set
├── scripts/                # seed + bootstrap
├── tests/                  # unit (27) + integration (8)
├── data/                   # Sample Flowpoint corpus
├── Dockerfile
├── docker-compose.yml
└── render.yaml             # Free-tier deploy config
```

## Testing

```bash
pytest tests/                    # 35 tests
```

## Evaluation

```bash
python -m evaluation.retrieval_eval   # no LLM keys needed
python -m evaluation.answer_eval      # needs provider key
```

## Deployment (Render free tier)

1. Push to GitHub
2. Render → New → Blueprint → connect repo
3. Set secrets: `GROQ_API_KEY`, `QDRANT_URL` (Qdrant Cloud free), `REDIS_URL` (Redis Cloud free)
4. First deploy ~10 min (model prefetch), subsequent ~2 min

See `render.yaml` for trade-offs.

## Scaling path

- Qdrant → Pinecone / Weaviate / pgvector
- Groq → self-hosted vLLM / Bedrock / Vertex
- In-process rate limiter → Redis-backed distributed
- SQLite registry → Postgres
- BM25 in-process → OpenSearch / Tantivy (past ~1M chunks)
- In-memory semantic cache index → Redis Search / HNSW

## Status

- [x] Step 1 — Architecture
- [x] Step 2 — Ingestion pipeline
- [x] Step 3 — Retrieval (hybrid + RRF + reranker + transforms + cache)
- [x] Step 4 — Generation (LLM router, citations, streaming, answer eval)
- [x] Step 5 — API + deployment
