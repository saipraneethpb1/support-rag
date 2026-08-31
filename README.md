# Ask — questions over your files

**[Live Demo](https://support-rag-p177.onrender.com)**

A small RAG app you can run yourself. Put markdown, HTML, tickets, a
changelog, or an OpenAPI spec in `data/`, ingest, and ask. Answers cite
the passages they used. It is not a company support bot — the bundled
files explain how to use the app, and you replace them with yours.

## How to use it

1. Start Qdrant + Redis, install, and set provider keys (see Local setup).
2. Open http://localhost:8000 and **upload** `.md`, `.txt`, `.html`, `.pdf`, or `.docx` (composer + or drag-and-drop). Uploads are scoped to your browser session so two visitors can use the same filename.
3. Ask a question. You can still drop files under `data/` and run ingest from the CLI.

| You have… | Put it here |
|-----------|-------------|
| Uploads from the UI | `data/uploads/<session>/` (written automatically; ephemeral on Render unless you add a disk) |
| Markdown docs | `data/sample_docs/` |
| HTML articles | `data/sample_help_center/` |
| Resolved tickets (JSONL) | `data/sample_tickets/tickets.jsonl` |
| Changelog | `data/CHANGELOG.md` |
| OpenAPI 3 JSON | `data/openapi.json` |

Then ingest again. Unchanged files are skipped.

## What's in the box

- **Hybrid retrieval**: vector (Qdrant + Gemini embeddings) + BM25, fused via RRF; optional cross-encoder rerank when `RERANKER_ENABLED=true` and `[local]` extras are installed
- **Upload**: `POST /ingest/upload` from the UI (`.md`, `.txt`, `.html`, `.pdf`, `.docx`) or drop files on the page. Each visitor gets an isolated folder and retrieval only sees shared corpus plus their own files.
- **Chats**: New chat plus a sidebar list of past threads, stored on the server for that visitor (SQLite). Older browser-only history is migrated once.
- **Connectors**: markdown, HTML, tickets (JSONL), changelog, OpenAPI
- **Structure-aware chunking**: splits on headings, prepends title + heading path
- **LLM router**: Groq primary → Gemini fallback, retries, circuit breaker
- **Citation audit**: flags markers that were not in context
- **Streaming**: `/chat/stream` SSE tokens, then a meta event with citations
- **Semantic cache**: near-duplicate questions; invalidated when ingest changes the corpus
- **Observability**: optional Langfuse, structured logs, per-stage timings
- **Evaluation**: retrieval (Hit@K / MRR / nDCG / Recall) and answer faithfulness

Runs on free tiers. Docker keeps the reranker off so it fits 512MB RAM.

## Local setup

```bash
# 1. Start infra
docker compose up -d qdrant redis langfuse

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Optional local embeddings/rerank: pip install -e ".[dev,local]"

# 3. Configure
cp .env.example .env
# Set GROQ_API_KEY and GOOGLE_API_KEY (embeddings + Gemini fallback)
# API_KEY is used for curl and webhooks; the chat UI does not ask visitors for it

# 4. Example files + index
python -m scripts.seed_demo_data
python -m scripts.bootstrap_index

# 5. Run
uvicorn api.main:app --reload
# Open http://localhost:8000 — upload files and ask; no key field
```

Try asking: “How do I add my own files?” or “How do citations work?”

Defaults use Gemini embeddings (`gemini-embedding-001`, dim 768). For local
sentence-transformers instead:

```bash
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384
RERANKER_ENABLED=true   # requires [local] extras
```

## Endpoints

| Method | Path                         | Auth    | Description                         |
|--------|------------------------------|---------|-------------------------------------|
| GET    | `/`                          | none    | Chat UI                             |
| GET    | `/health`                    | none    | Dependency health                   |
| POST   | `/chat`                      | cookie or API key | Blocking chat (uses semantic cache) |
| POST   | `/chat/stream`               | cookie or API key | SSE streaming chat                  |
| POST   | `/ingest/run`                | cookie or API key | Full ingest pass                    |
| GET    | `/chats/`                    | cookie or API key | List this visitor's chats           |
| POST   | `/chats/`                    | cookie or API key | Create a chat                       |
| GET    | `/chats/{id}`                | cookie or API key | Load a chat                         |
| PUT    | `/chats/{id}`                | cookie or API key | Save messages                       |
| DELETE | `/chats/{id}`                | cookie or API key | Delete a chat                       |
| POST   | `/ingest/upload`             | cookie or API key | Upload and index documents          |
| GET    | `/ingest/uploads`            | cookie or API key | List this visitor's uploads         |
| DELETE | `/ingest/uploads/{filename}` | cookie or API key | Delete an uploaded file             |
| GET    | `/ingest/status`             | cookie or API key | Indexed document count              |
| POST   | `/webhooks/tickets/resolved` | secret  | Single-ticket ingest on push        |
| POST   | `/webhooks/docs/updated`     | secret  | Re-ingest on docs CI                |
| GET    | `/docs`                      | none    | OpenAPI interactive docs            |

Webhook auth uses `WEBHOOK_SECRET` when set, otherwise `API_KEY`.

## Folder structure

```
ask-docs/
├── api/                    # FastAPI app, routes, auth, rate limit, chat UI
├── config/                 # Settings + editable prompt templates
├── ingestion/              # Connectors, chunker, embedder, registry, pipeline
├── retrieval/              # Vector, BM25, hybrid, rerank, transforms, facade
├── generation/             # LLM router, prompt, citation, generator
├── cache/                  # Embedding + semantic caches
├── observability/          # Langfuse + structlog
├── evaluation/             # Retrieval + answer eval, golden QA set
├── scripts/                # seed example files + bootstrap index
├── tests/
├── data/                   # Files you ingest (examples ship here)
├── Dockerfile
├── docker-compose.yml
└── render.yaml
```

## Testing

```bash
pytest tests/ -v
ruff check .
```

## Evaluation

```bash
python -m evaluation.retrieval_eval   # no LLM keys needed
python -m evaluation.answer_eval      # needs provider key
```

Golden questions in `evaluation/datasets/golden_qa.jsonl` match the
bundled example files. If you replace `data/`, update that set.

## Deployment (Render free tier)

1. Push to GitHub
2. Render → New → Blueprint → connect repo
3. Set secrets: `GROQ_API_KEY`, `GOOGLE_API_KEY`, `QDRANT_URL`, `REDIS_URL`
4. Blueprint generates `API_KEY` / `WEBHOOK_SECRET`

See `render.yaml`. `RERANKER_ENABLED=false` is honored so cold starts
do not load a local cross-encoder.

## Scaling path

- Qdrant → Pinecone / Weaviate / pgvector
- Groq → self-hosted vLLM / Bedrock / Vertex
- In-process rate limiter → Redis-backed distributed
- SQLite registry → Postgres
- BM25 in-process → OpenSearch / Tantivy (past ~1M chunks)
- In-memory semantic cache index → Redis Search / HNSW
