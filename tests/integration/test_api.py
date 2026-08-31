"""API integration tests.

We boot FastAPI with injected fake singletons — no Qdrant, Redis, or
LLM calls. The goal is to pin down request/response contracts: auth,
rate limiting, schema shape, streaming wire format.
"""
from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.ingest import router as ingest_router
from api.routes.chats import router as chats_router
from core.chat_store import ChatStore
from ingestion.workers.webhook_handler import router as webhooks_router
from api.schemas import ChatResponse
from generation.generator import GeneratedAnswer, StreamEvent
from generation.prompt_builder import Citation
from generation.citation import CitationAudit
from retrieval.retriever import RetrievalResult


# ---------- Fakes ----------

class _FakeGenerator:
    async def generate(self, question, *, history=None, source_types=None, owner_id=None, use_cache=True):
        cite = Citation(
            marker=1, chunk_id="c1", doc_id="d1", title="Cancel Subscription",
            url="https://ex.com/1", source_type="markdown_docs",
            snippet="How to cancel your subscription.",
        )
        audit = CitationAudit(
            used_markers={1}, invented_markers=set(), sentence_coverage=1.0,
            cleaned_answer="Go to Settings > Billing [1].", used_citations=[cite],
        )
        empty_retrieval = RetrievalResult(
            query=question, transformed=None, chunks=[], timings_ms={},  # type: ignore[arg-type]
            candidate_count_before_rerank=0,
        )
        return GeneratedAnswer(
            trace_id="trace123",
            answer="Go to Settings > Billing [1].",
            citations=[cite],
            audit=audit,
            retrieval=empty_retrieval,
            llm_provider="fake",
            cache_hit=False,
            timings_ms={"total": 10.0},
        )

    async def stream(self, question, *, history=None, source_types=None, owner_id=None, use_cache=True) -> AsyncIterator[StreamEvent]:
        for tok in ["Go ", "to ", "Settings", "."]:
            yield StreamEvent(type="token", data={"text": tok})
            await asyncio.sleep(0)
        yield StreamEvent(
            type="meta",
            data={
                "trace_id": "trace123",
                "citations": [{"marker": 1, "title": "T", "url": "u", "source_type": "markdown_docs", "snippet": "s"}],
                "invented_citations": [],
                "coverage": 1.0,
                "timings_ms": {"total": 5.0},
            },
        )


class _FakeTracer:
    def trace(self, **kw):
        class _T:
            def update(self, **kw): pass
            def score(self, **kw): pass
            def span(self, **kw): return self
            def end(self, **kw): pass
        return _T()
    def flush(self): pass


class _FakePipeline:
    async def run(self):
        @dataclass
        class _S:
            new: int = 0
            updated: int = 0
            unchanged: int = 0
            deleted: int = 0
            chunks_written: int = 0
            errors: list = field(default_factory=list)
        return _S()

    async def ingest_single(self, rec):
        return True

    async def delete_doc(self, doc_id):
        return None

    async def document_count(self):
        return 3


class _FakeVectorStore:
    def __init__(self):
        class _C:
            async def get_collections(self):
                class R:
                    collections = []
                return R()
        self._client = _C()


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    app.state.generator = _FakeGenerator()
    app.state.tracer = _FakeTracer()
    app.state.ingestion_pipeline = _FakePipeline()
    app.state.vector_store = _FakeVectorStore()
    app.state.chat_store = ChatStore(tmp_path / "chats.db")
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(chats_router)
    app.include_router(ingest_router)
    app.include_router(webhooks_router)
    return TestClient(app)


# ---------- Tests ----------

def test_health_returns_200_and_schema(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body and "qdrant" in body and "redis" in body


def test_chat_requires_api_key(client):
    r = client.post("/chat", json={"question": "hi"})
    assert r.status_code == 401


def test_chat_rejects_wrong_api_key(client):
    r = client.post("/chat", json={"question": "hi"}, headers={"x-api-key": "wrong"})
    assert r.status_code == 401


def test_chat_happy_path(client):
    r = client.post(
        "/chat",
        json={"question": "How do I cancel?"},
        headers={"x-api-key": "local-dev-key"},
    )
    assert r.status_code == 200
    body = ChatResponse(**r.json())
    assert "Settings" in body.answer
    assert body.citations[0].marker == 1
    assert body.invented_citations == []


def test_chat_validates_empty_question(client):
    r = client.post(
        "/chat",
        json={"question": ""},
        headers={"x-api-key": "local-dev-key"},
    )
    assert r.status_code == 422  # pydantic validation


def test_chat_stream_emits_sse_tokens_then_meta(client):
    with client.stream(
        "POST", "/chat/stream",
        json={"question": "How do I cancel?"},
        headers={"x-api-key": "local-dev-key"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = b"".join(r.iter_bytes()).decode()

    # Parse SSE frames
    frames = [f for f in raw.split("\n\n") if f.strip()]
    events = []
    for f in frames:
        data_line = next((line for line in f.split("\n") if line.startswith("data:")), None)
        if data_line:
            events.append(json.loads(data_line[5:].strip()))

    token_events = [e for e in events if e["type"] == "token"]
    meta_events = [e for e in events if e["type"] == "meta"]
    assert len(token_events) >= 1
    assert len(meta_events) == 1
    # Tokens should concatenate to the full answer
    combined = "".join(e["text"] for e in token_events)
    assert "Settings" in combined
    # Meta must carry citations + trace info
    assert meta_events[0]["trace_id"] == "trace123"
    assert len(meta_events[0]["citations"]) == 1


def test_ingest_requires_api_key(client):
    r = client.post("/ingest/run")
    assert r.status_code == 401


def test_ingest_accepts_access_cookie(client):
    client.cookies.set("ask_access", "local-dev-key")
    r = client.post("/ingest/run")
    assert r.status_code == 200
    r = client.post("/ingest/run", headers={"x-api-key": "local-dev-key"})
    assert r.status_code == 200
    body = r.json()
    assert "new" in body and "chunks_written" in body


def test_upload_requires_api_key(client):
    r = client.post("/ingest/upload", files={"files": ("note.md", b"# Hello", "text/markdown")})
    assert r.status_code == 401


def test_upload_markdown_indexes_file(client, tmp_path, monkeypatch):
    from ingestion import uploads as uploads_mod
    monkeypatch.setattr(uploads_mod, "UPLOAD_DIR", tmp_path)
    r = client.post(
        "/ingest/upload",
        headers={"x-api-key": "local-dev-key"},
        files={"files": ("guide.md", b"# Onboarding\nWelcome to the team.", "text/markdown")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ingested"] == 1
    assert body["files"][0]["ok"] is True
    assert body["files"][0]["filename"] == "guide.md"
    assert (tmp_path / "anon" / "guide.md").read_text() == "# Onboarding\nWelcome to the team."


def test_upload_rejects_unsupported_type(client, tmp_path, monkeypatch):
    from ingestion import uploads as uploads_mod
    monkeypatch.setattr(uploads_mod, "UPLOAD_DIR", tmp_path)
    r = client.post(
        "/ingest/upload",
        headers={"x-api-key": "local-dev-key"},
        files={"files": ("photo.png", b"not-an-image", "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ingested"] == 0
    assert body["files"][0]["ok"] is False


def test_list_uploads(client, tmp_path, monkeypatch):
    from ingestion import uploads as uploads_mod
    monkeypatch.setattr(uploads_mod, "UPLOAD_DIR", tmp_path)
    (tmp_path / "anon").mkdir()
    (tmp_path / "anon" / "note.md").write_text("# Note\n", encoding="utf-8")
    r = client.get("/ingest/uploads", headers={"x-api-key": "local-dev-key"})
    assert r.status_code == 200
    files = r.json()["files"]
    assert files[0]["filename"] == "note.md"
    assert files[0]["bytes"] > 0


def test_upload_isolation_by_visitor_header(client, tmp_path, monkeypatch):
    from ingestion import uploads as uploads_mod
    monkeypatch.setattr(uploads_mod, "UPLOAD_DIR", tmp_path)
    r = client.post(
        "/ingest/upload",
        headers={"x-api-key": "local-dev-key", "X-Visitor-Id": "alice"},
        files={"files": ("exercise.md", b"# Alice\nprivate", "text/markdown")},
    )
    assert r.status_code == 200
    assert (tmp_path / "alice" / "exercise.md").exists()
    listed = client.get(
        "/ingest/uploads",
        headers={"x-api-key": "local-dev-key", "X-Visitor-Id": "bob"},
    )
    assert listed.json()["files"] == []
    listed_alice = client.get(
        "/ingest/uploads",
        headers={"x-api-key": "local-dev-key", "X-Visitor-Id": "alice"},
    )
    assert listed_alice.json()["files"][0]["filename"] == "exercise.md"


def test_delete_upload(client, tmp_path, monkeypatch):
    from ingestion import uploads as uploads_mod
    monkeypatch.setattr(uploads_mod, "UPLOAD_DIR", tmp_path)
    client.post(
        "/ingest/upload",
        headers={"x-api-key": "local-dev-key", "X-Visitor-Id": "alice"},
        files={"files": ("gone.md", b"# Bye", "text/markdown")},
    )
    r = client.delete(
        "/ingest/uploads/gone.md",
        headers={"x-api-key": "local-dev-key", "X-Visitor-Id": "alice"},
    )
    assert r.status_code == 200
    assert not (tmp_path / "alice" / "gone.md").exists()


def test_chats_are_scoped_to_visitor(client):
    headers_a = {"x-api-key": "local-dev-key", "X-Visitor-Id": "alice"}
    headers_b = {"x-api-key": "local-dev-key", "X-Visitor-Id": "bob"}
    created = client.post("/chats/", json={"title": "Alice thread"}, headers=headers_a)
    assert created.status_code == 200
    chat_id = created.json()["id"]
    save = client.put(
        f"/chats/{chat_id}",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=headers_a,
    )
    assert save.status_code == 200
    bob_list = client.get("/chats/", headers=headers_b)
    assert bob_list.json()["chats"] == []
    bob_get = client.get(f"/chats/{chat_id}", headers=headers_b)
    assert bob_get.status_code == 404
    alice_get = client.get(f"/chats/{chat_id}", headers=headers_a)
    assert alice_get.status_code == 200
    assert alice_get.json()["messages"][0]["content"] == "hi"


def test_ingest_status(client):
    r = client.get("/ingest/status", headers={"x-api-key": "local-dev-key"})
    assert r.status_code == 200
    assert r.json()["documents"] == 3

