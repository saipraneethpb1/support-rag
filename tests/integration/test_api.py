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
from api.schemas import ChatResponse
from generation.generator import GeneratedAnswer, StreamEvent
from generation.prompt_builder import Citation
from generation.citation import CitationAudit
from retrieval.retriever import RetrievalResult


# ---------- Fakes ----------

class _FakeGenerator:
    async def generate(self, question, *, history=None, source_types=None, use_cache=True):
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

    async def stream(self, question, *, history=None, source_types=None) -> AsyncIterator[StreamEvent]:
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
            new: int = 0; updated: int = 0; unchanged: int = 0
            deleted: int = 0; chunks_written: int = 0
            errors: list = field(default_factory=list)
        return _S()


class _FakeVectorStore:
    def __init__(self):
        class _C:
            async def get_collections(self):
                class R: collections = []
                return R()
        self._client = _C()


@pytest.fixture
def client():
    app = FastAPI()
    app.state.generator = _FakeGenerator()
    app.state.tracer = _FakeTracer()
    app.state.ingestion_pipeline = _FakePipeline()
    app.state.vector_store = _FakeVectorStore()
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(ingest_router)
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
        data_line = next((l for l in f.split("\n") if l.startswith("data:")), None)
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


def test_ingest_happy_path(client):
    r = client.post("/ingest/run", headers={"x-api-key": "local-dev-key"})
    assert r.status_code == 200
    body = r.json()
    assert "new" in body and "chunks_written" in body
