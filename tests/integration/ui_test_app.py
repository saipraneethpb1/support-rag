"""Local UI test server with fake LLM/ingest — not for production."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from api.middleware.auth import attach_access_cookie
from api.middleware.visitor import attach_visitor_cookie
from api.routes.chat import router as chat_router
from api.routes.chats import router as chats_router
from api.routes.health import router as health_router
from api.routes.ingest import router as ingest_router
from core.chat_store import ChatStore
from core.session import new_visitor_id
from tests.integration.test_api import (
    _FakeGenerator,
    _FakePipeline,
    _FakeTracer,
    _FakeVectorStore,
)

_UI = (Path(__file__).resolve().parents[2] / "api" / "chat_ui.html").read_text(encoding="utf-8")

app = FastAPI()
app.state.generator = _FakeGenerator()
app.state.tracer = _FakeTracer()
app.state.ingestion_pipeline = _FakePipeline()
app.state.vector_store = _FakeVectorStore()
app.state.chat_store = ChatStore("data/registry/chats-ui-test.db")
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(chats_router)
app.include_router(ingest_router)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    response = HTMLResponse(_UI)
    attach_access_cookie(response)
    attach_visitor_cookie(response, new_visitor_id())
    return response
