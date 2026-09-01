"""UI-equivalent chat session tests: save, new chat, switch back."""
from __future__ import annotations

import json

from api.schemas import ChatResponse
from tests.integration.test_api import (
    _FakeGenerator,
    _FakePipeline,
    _FakeTracer,
    _FakeVectorStore,
)
from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.chats import router as chats_router
from api.routes.ingest import router as ingest_router
from ingestion.workers.webhook_handler import router as webhooks_router
from core.chat_store import ChatStore
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

AUTH = {"x-api-key": "local-dev-key", "X-Visitor-Id": "tester"}


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


def _turn(client, chat_id: str, question: str, title: str | None = None):
    """Mirror the UI: stream an answer, then PUT the thread."""
    with client.stream(
        "POST",
        "/chat/stream",
        json={"question": question, "history": []},
        headers=AUTH,
    ) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes()).decode()
    tokens = []
    citations = []
    for frame in raw.split("\n\n"):
        line = next((ln for ln in frame.split("\n") if ln.startswith("data:")), None)
        if not line:
            continue
        payload = json.loads(line[5:].strip())
        if payload.get("type") == "token":
            tokens.append(payload["text"])
        if payload.get("type") == "meta":
            citations = payload.get("citations") or []
    answer = "".join(tokens)
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "citations": citations},
    ]
    body = {"messages": messages}
    if title:
        body["title"] = title
    put = client.put(f"/chats/{chat_id}", json=body, headers=AUTH)
    assert put.status_code == 200
    return answer, messages


def test_new_chat_keeps_previous_thread(client):
    first = client.post("/chats/", json={"title": "New chat"}, headers=AUTH)
    assert first.status_code == 200
    chat_a = first.json()["id"]

    answer_a, _ = _turn(client, chat_a, "How do I cancel?", title="How do I cancel?")
    assert "Settings" in answer_a

    listed = client.get("/chats/", headers=AUTH).json()["chats"]
    assert any(c["id"] == chat_a for c in listed)
    assert next(c for c in listed if c["id"] == chat_a)["title"] == "How do I cancel?"

    second = client.post("/chats/", json={"title": "New chat"}, headers=AUTH)
    assert second.status_code == 200
    chat_b = second.json()["id"]
    assert chat_b != chat_a

    listed = client.get("/chats/", headers=AUTH).json()["chats"]
    ids = {c["id"] for c in listed}
    assert {chat_a, chat_b} <= ids

    reloaded_a = client.get(f"/chats/{chat_a}", headers=AUTH).json()
    assert reloaded_a["title"] == "How do I cancel?"
    assert [m["role"] for m in reloaded_a["messages"]] == ["user", "assistant"]
    assert reloaded_a["messages"][0]["content"] == "How do I cancel?"
    assert "Settings" in reloaded_a["messages"][1]["content"]

    empty_b = client.get(f"/chats/{chat_b}", headers=AUTH).json()
    assert empty_b["messages"] == []
    assert empty_b["title"] == "New chat"

    _turn(client, chat_b, "How do citations work?", title="How do citations work?")

    still_a = client.get(f"/chats/{chat_a}", headers=AUTH).json()
    now_b = client.get(f"/chats/{chat_b}", headers=AUTH).json()
    assert still_a["messages"][0]["content"] == "How do I cancel?"
    assert now_b["messages"][0]["content"] == "How do citations work?"
    assert still_a["messages"] != now_b["messages"]

    listed = client.get("/chats/", headers=AUTH).json()["chats"]
    titles = {c["id"]: c["title"] for c in listed}
    assert titles[chat_a] == "How do I cancel?"
    assert titles[chat_b] == "How do citations work?"

    deleted = client.delete(f"/chats/{chat_b}", headers=AUTH)
    assert deleted.status_code == 200
    listed = client.get("/chats/", headers=AUTH).json()["chats"]
    assert chat_a in {c["id"] for c in listed}
    assert chat_b not in {c["id"] for c in listed}
    assert client.get(f"/chats/{chat_a}", headers=AUTH).status_code == 200
    assert client.get(f"/chats/{chat_b}", headers=AUTH).status_code == 404


def test_blocking_chat_and_source_types(client):
    r = client.post(
        "/chat",
        json={"question": "How do I cancel?", "source_types": ["markdown_docs"]},
        headers=AUTH,
    )
    assert r.status_code == 200
    body = ChatResponse(**r.json())
    assert "Settings" in body.answer


def test_health_and_ingest_status(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert "qdrant" in health.json()
    status = client.get("/ingest/status", headers=AUTH)
    assert status.status_code == 200
    assert status.json()["documents"] == 3
