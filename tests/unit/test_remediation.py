"""Tests for auth, BM25 filters, semantic cache versioning, and webhooks."""
from __future__ import annotations
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.middleware.auth import require_api_key
from cache.semantic_cache import SemanticCache
from retrieval.bm25_store import BM25Store
from retrieval.retriever import Retriever
from ingestion.workers.webhook_handler import router as webhooks_router


# ---------- Auth ----------

@pytest.mark.asyncio
async def test_require_api_key_accepts_matching_key(monkeypatch):
    from config import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    monkeypatch.setenv("API_KEY", "secret-key")
    settings_mod.get_settings.cache_clear()
    await require_api_key(x_api_key="secret-key")


@pytest.mark.asyncio
async def test_require_api_key_rejects_wrong_key(monkeypatch):
    from config import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    monkeypatch.setenv("API_KEY", "secret-key")
    settings_mod.get_settings.cache_clear()
    with pytest.raises(HTTPException) as ei:
        await require_api_key(x_api_key="wrong")
    assert ei.value.status_code == 401


# ---------- BM25 source filter ----------

def test_bm25_respects_source_types_filter():
    store = BM25Store.__new__(BM25Store)
    store._chunk_ids = []
    store._payloads = []
    store._bm25 = None
    store.rebuild([
        ("c1", "subscription cancellation policy for annual plans", {
            "chunk_id": "c1", "text": "subscription cancellation policy for annual plans",
            "source_type": "markdown_docs",
        }),
        ("c2", "zendesk ticket about refunding a billing charge", {
            "chunk_id": "c2", "text": "zendesk ticket about refunding a billing charge",
            "source_type": "tickets",
        }),
        ("c3", "saml identity provider configuration guide", {
            "chunk_id": "c3", "text": "saml identity provider configuration guide",
            "source_type": "help_center",
        }),
        ("c4", "unrelated changelog entry about UI polish", {
            "chunk_id": "c4", "text": "unrelated changelog entry about UI polish",
            "source_type": "changelog",
        }),
    ])
    hits = store.search("billing charge refund", top_k=10, source_types=["tickets"])
    assert hits
    assert all(h["source_type"] == "tickets" for h in hits)
    assert hits[0]["chunk_id"] == "c2"


def test_bm25_without_filter_returns_mixed_sources():
    store = BM25Store.__new__(BM25Store)
    store.rebuild([
        ("c1", "openapi rate limit headers for the public api", {
            "chunk_id": "c1", "text": "openapi rate limit headers for the public api",
            "source_type": "openapi",
        }),
        ("c2", "markdown docs describing api rate limiting behavior", {
            "chunk_id": "c2", "text": "markdown docs describing api rate limiting behavior",
            "source_type": "markdown_docs",
        }),
        ("c3", "gardening tips for tomatoes", {
            "chunk_id": "c3", "text": "gardening tips for tomatoes",
            "source_type": "help_center",
        }),
    ])
    hits = store.search("api rate limit", top_k=10)
    types = {h["source_type"] for h in hits}
    assert "openapi" in types
    assert "markdown_docs" in types
    assert "help_center" not in types


# ---------- Retriever respects settings default ----------

def test_retriever_defaults_rerank_from_settings(monkeypatch):
    from config import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    settings_mod.get_settings.cache_clear()
    r = Retriever.__new__(Retriever)
    # Call __init__ carefully with no heavy deps
    with patch("retrieval.retriever.QueryTransformer"), \
         patch("retrieval.retriever.HybridSearcher"), \
         patch("retrieval.retriever.Reranker"):
        r = Retriever(enable_rerank=None)
    assert r.enable_rerank is False


def test_retriever_explicit_enable_rerank_overrides(monkeypatch):
    from config import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    settings_mod.get_settings.cache_clear()
    with patch("retrieval.retriever.QueryTransformer"), \
         patch("retrieval.retriever.HybridSearcher"), \
         patch("retrieval.retriever.Reranker"):
        r = Retriever(enable_rerank=True)
    assert r.enable_rerank is True


# ---------- Semantic cache corpus bump ----------

@pytest.mark.asyncio
async def test_semcache_bump_changes_version_without_redis():
    cache = SemanticCache(corpus_version="v1")
    cache._available = False  # force no redis
    new_v = await cache.bump_corpus_version()
    assert new_v == "v2"
    assert cache.corpus_version == "v2"
    assert cache._loaded is False
    assert cache._index == []


# ---------- Webhooks ----------

class _FakePipeline:
    def __init__(self):
        self.single_calls = []
        self.run_calls = 0

    async def ingest_single(self, rec):
        self.single_calls.append(rec)
        return True

    async def run(self):
        self.run_calls += 1

        @dataclass
        class _S:
            new: int = 1
            updated: int = 0
            unchanged: int = 0
            deleted: int = 0
            chunks_written: int = 2
            errors: list = field(default_factory=list)

        return _S()


@pytest.fixture
def webhook_client(monkeypatch):
    from config import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    monkeypatch.setenv("API_KEY", "local-dev-key")
    monkeypatch.setenv("WEBHOOK_SECRET", "whsec")
    settings_mod.get_settings.cache_clear()

    app = FastAPI()
    pipe = _FakePipeline()
    app.state.ingestion_pipeline = pipe
    app.include_router(webhooks_router)
    client = TestClient(app)
    client.pipe = pipe  # type: ignore[attr-defined]
    return client


def test_webhook_rejects_bad_secret(webhook_client):
    r = webhook_client.post(
        "/webhooks/tickets/resolved",
        json={"id": "1", "status": "resolved"},
        headers={"x-webhook-secret": "nope"},
    )
    assert r.status_code == 401


def test_webhook_ticket_resolved_happy_path(webhook_client):
    r = webhook_client.post(
        "/webhooks/tickets/resolved",
        json={
            "id": "42",
            "status": "resolved",
            "subject": "Billing question",
            "messages": [{"author": "agent", "body": "Refunded"}],
            "tags": ["billing"],
        },
        headers={"x-webhook-secret": "whsec"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ingested"] is True
    assert body["changed"] is True
    assert len(webhook_client.pipe.single_calls) == 1


def test_webhook_ticket_ignores_non_resolved(webhook_client):
    r = webhook_client.post(
        "/webhooks/tickets/resolved",
        json={"id": "9", "status": "open"},
        headers={"x-webhook-secret": "whsec"},
    )
    assert r.status_code == 200
    assert r.json()["ignored"] is True
    assert webhook_client.pipe.single_calls == []


def test_webhook_ticket_validates_payload(webhook_client):
    r = webhook_client.post(
        "/webhooks/tickets/resolved",
        json={"status": "resolved"},  # missing id
        headers={"x-webhook-secret": "whsec"},
    )
    assert r.status_code == 422


def test_webhook_docs_updated(webhook_client):
    r = webhook_client.post(
        "/webhooks/docs/updated",
        json={"reason": "ci"},
        headers={"x-webhook-secret": "whsec"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert webhook_client.pipe.run_calls == 1
