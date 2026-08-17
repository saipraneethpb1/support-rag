"""Write the bundled example corpus.

The sample files document THIS app so anyone can open the UI and
understand what to do: add files, ingest, ask questions.

Usage:
    python -m scripts.seed_demo_data
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("data")


DOCS = {
    "what-this-is.md": """# What this is

This app answers questions using **only files you give it**. It is not
tied to a company, product, or support desk. Put your notes, docs, or
exports in the `data/` folder, ingest them, then ask.

## What happens when you ask

1. Your question is searched against the ingested files (keyword +
   meaning).
2. The best matching passages are sent to a language model.
3. The reply cites those passages. If nothing relevant was ingested,
   it should say so instead of guessing.

## What it is not

It does not browse the web. It does not know your files until you
ingest them. Changing a file on disk does nothing until you run ingest
again (`POST /ingest/run` or `python -m scripts.bootstrap_index`).
""",
    "add-your-files.md": """# Add your own files

Replace the example files under `data/` with yours, then ingest.

## Folders the app already watches

| Path | What to put there |
|------|-------------------|
| `data/sample_docs/` | Markdown (`.md`). Nested folders are fine. |
| `data/sample_help_center/` | HTML articles (`.html`). |
| `data/sample_tickets/tickets.jsonl` | One JSON object per line (id, subject, status, messages). |
| `data/CHANGELOG.md` | A Keep-a-Changelog file; each version becomes its own document. |
| `data/openapi.json` | OpenAPI 3 JSON; each endpoint becomes its own document. |

You can keep some of these empty. Connectors skip missing folders.

## After you add or edit files

```
python -m scripts.bootstrap_index
```

Or, with the API running and your API key:

```
curl -X POST http://localhost:8000/ingest/run -H "x-api-key: YOUR_KEY"
```

Unchanged files are skipped (content hash). Deleted files are removed
from the index on a full ingest pass.
""",
    "ask-questions.md": """# Ask questions

Open http://localhost:8000. Paste the same `API_KEY` you set in `.env`
into the key field (it is empty on purpose). Type a question.

## Chat API

Blocking JSON:

```
curl -X POST http://localhost:8000/chat \\
  -H "content-type: application/json" \\
  -H "x-api-key: YOUR_KEY" \\
  -d '{"question": "How do I add my own files?"}'
```

Streaming (server-sent events): `POST /chat/stream` with the same body.
Tokens arrive as `event: token`; citations arrive as a final `event: meta`.

## If you get 401

The `x-api-key` header (or the UI field) must match `API_KEY` in `.env`.
The demo UI does not pre-fill the key.

## Optional filters

You can limit retrieval to source types: `markdown_docs`, `help_center`,
`tickets`, `changelog`, `openapi` via the `source_types` field on the
chat request.
""",
    "how-answers-work.md": """# How answers work

Retrieval is **hybrid**: vector search (meaning) plus BM25 (exact words
like error codes and filenames), fused with Reciprocal Rank Fusion.

## Citations

Every factual sentence should include markers like [1] that map to
ingested files. The UI shows those as links under the answer. Invented
markers (numbers that were not in the retrieved context) are stripped
and counted as a hallucination signal.

## Cache

Near-duplicate questions can return a cached answer. After a successful
ingest that changed the corpus, that cache is invalidated.

## Honesty

The model is instructed to refuse when the ingested files do not
contain the answer. If it still guesses, treat citations as the
source of truth and re-ingest better docs.
""",
    "sources.md": """# Source types

The pipeline is the same for every source: clean → chunk → embed →
store. Only the connector differs.

## Markdown

Best for hand-written docs. Headings become chunk boundaries. The
first `#` heading is the title.

## HTML

For exported help-center pages. Nav, footer, and scripts are stripped
so the model sees the article body.

## Tickets

JSONL of resolved conversations. Useful for "we already answered this"
style questions. Unresolved tickets in a webhook payload are ignored.

## Changelog

One document per `##` version heading. Ask "what changed in 0.2.0?"
and you should retrieve that section, not the whole file.

## OpenAPI

One document per HTTP method + path. Ask "how do I call POST /chat?"
and you should retrieve that operation.
""",
    "run-locally.md": """# Run it locally

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
""",
}

HELP_HTML = {
    "add-your-files.html": """<!DOCTYPE html>
<html>
<head><title>Add your own files</title></head>
<body>
  <nav>Help</nav>
  <article>
    <h1>Add your own files</h1>
    <p>Drop markdown into <code>data/sample_docs</code>, HTML into
    <code>data/sample_help_center</code>, then run ingest. The app does
    not read files until ingest runs.</p>
    <h2>Ingest</h2>
    <p>Use <code>python -m scripts.bootstrap_index</code> or
    <code>POST /ingest/run</code> with your API key.</p>
    <p>Unchanged files are skipped. Deleted files drop out of the index
    on a full ingest pass.</p>
  </article>
  <footer>Ask</footer>
</body>
</html>
""",
    "api-key.html": """<!DOCTYPE html>
<html>
<head><title>API key</title></head>
<body>
  <nav>Help</nav>
  <main class="article">
    <h1>API key</h1>
    <p>Chat and ingest require the <code>x-api-key</code> header. Set
    <code>API_KEY</code> in <code>.env</code>. The chat page has a key
    field at the top; it starts empty.</p>
    <p>A 401 means the key does not match. Webhooks use
    <code>WEBHOOK_SECRET</code> if set, otherwise the same API key.</p>
  </main>
</body>
</html>
""",
}

CHANGELOG = """# Changelog

All notable changes to this app are listed here.

## [0.2.0] - 2026-08-17

### Added
- Generic document Q&A (not tied to a company support desk)
- Example corpus that explains how to add your own files

### Changed
- Semantic cache invalidates when ingest actually changes the corpus

## [0.1.0] - 2026-08-09

### Added
- Hybrid retrieval (vector + BM25 + RRF)
- Streaming chat and citation audit
- Markdown, HTML, tickets, changelog, and OpenAPI connectors
"""

TICKETS = [
    {
        "id": "T-1001",
        "subject": "Chat returns 401",
        "status": "resolved",
        "created_at": "2026-08-10T10:12:00Z",
        "updated_at": "2026-08-10T11:05:00Z",
        "tags": ["auth"],
        "messages": [
            {"author": "user", "body": "The UI says invalid API key when I send a question.", "ts": "2026-08-10T10:12:00Z"},
            {"author": "agent", "body": "Paste the same value as API_KEY from your .env into the key field. The field is empty on purpose and is sent as the x-api-key header.", "ts": "2026-08-10T10:44:00Z"},
            {"author": "user", "body": "That worked.", "ts": "2026-08-10T11:05:00Z"},
        ],
    },
    {
        "id": "T-1002",
        "subject": "Answers ignore a file I just added",
        "status": "resolved",
        "created_at": "2026-08-12T14:22:00Z",
        "updated_at": "2026-08-12T16:40:00Z",
        "tags": ["ingest"],
        "messages": [
            {"author": "user", "body": "I copied a markdown file into data/sample_docs but chat still says it is not in the files.", "ts": "2026-08-12T14:22:00Z"},
            {"author": "agent", "body": "Ingest is not automatic. Run python -m scripts.bootstrap_index or POST /ingest/run. Then ask again.", "ts": "2026-08-12T15:30:00Z"},
            {"author": "user", "body": "Got it, ingest fixed it.", "ts": "2026-08-12T16:40:00Z"},
        ],
    },
    {
        "id": "T-1003",
        "subject": "Want citations not a generic essay",
        "status": "resolved",
        "created_at": "2026-08-14T08:00:00Z",
        "updated_at": "2026-08-14T09:10:00Z",
        "tags": ["citations"],
        "messages": [
            {"author": "user", "body": "The model answered without pointing at a file.", "ts": "2026-08-14T08:00:00Z"},
            {"author": "agent", "body": "Answers should include [1] style markers from retrieved passages. If markers are missing, the files may not contain the fact — add a clearer doc and re-ingest. Invented markers are stripped.", "ts": "2026-08-14T09:10:00Z"},
        ],
    },
]

OPENAPI = {
    "openapi": "3.0.3",
    "info": {
        "title": "Ask HTTP API",
        "version": "0.1.0",
        "description": "Chat and ingest over files you indexed.",
    },
    "paths": {
        "/chat": {
            "post": {
                "operationId": "chat",
                "summary": "Ask a question (JSON)",
                "description": "Blocking chat. Requires x-api-key. Optional source_types filter: markdown_docs, help_center, tickets, changelog, openapi.",
                "parameters": [
                    {"name": "x-api-key", "in": "header", "required": True, "description": "Must match API_KEY."},
                ],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "source_types": {"type": "array", "items": {"type": "string"}},
                                    "use_cache": {"type": "boolean"},
                                },
                                "required": ["question"],
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Answer, citations, and timings."},
                    "401": {"description": "Invalid API key."},
                },
            }
        },
        "/chat/stream": {
            "post": {
                "operationId": "chatStream",
                "summary": "Ask a question (SSE)",
                "description": "Same body as POST /chat. Streams event: token then a final event: meta with citations.",
                "responses": {
                    "200": {"description": "text/event-stream"},
                    "401": {"description": "Invalid API key."},
                },
            }
        },
        "/ingest/run": {
            "post": {
                "operationId": "ingestRun",
                "summary": "Re-index files on disk",
                "description": "Full ingest pass. Unchanged files are skipped. Requires x-api-key.",
                "responses": {
                    "200": {"description": "Counts of new, updated, unchanged, deleted."},
                    "401": {"description": "Invalid API key."},
                },
            }
        },
        "/health": {
            "get": {
                "operationId": "health",
                "summary": "Dependency health",
                "description": "Reports Qdrant, Redis, and configured LLM providers. No auth.",
                "responses": {"200": {"description": "Health payload."}},
            }
        },
    },
}


def main() -> None:
    docs_root = ROOT / "sample_docs"
    if docs_root.exists():
        for path in docs_root.rglob("*"):
            if path.is_file():
                path.unlink()
    docs_root.mkdir(parents=True, exist_ok=True)
    for rel, body in DOCS.items():
        path = docs_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    help_root = ROOT / "sample_help_center"
    if help_root.exists():
        for path in help_root.glob("*.html"):
            path.unlink()
    help_root.mkdir(parents=True, exist_ok=True)
    for rel, body in HELP_HTML.items():
        (help_root / rel).write_text(body, encoding="utf-8")

    (ROOT / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")

    (ROOT / "sample_tickets").mkdir(parents=True, exist_ok=True)
    with (ROOT / "sample_tickets" / "tickets.jsonl").open("w", encoding="utf-8") as f:
        for t in TICKETS:
            f.write(json.dumps(t) + "\n")

    (ROOT / "openapi.json").write_text(json.dumps(OPENAPI, indent=2), encoding="utf-8")

    print(
        f"Seeded: {len(DOCS)} docs, {len(HELP_HTML)} HTML articles, "
        f"1 changelog, {len(TICKETS)} tickets, 1 openapi spec"
    )


if __name__ == "__main__":
    main()
