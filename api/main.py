"""FastAPI app.

Uses lifespan to construct singletons (embedder, vector store, retriever,
generator, ingestion pipeline, Langfuse tracer) once per process, not per
request. Heavy objects like the embedding model load exactly once.

Routes:
  GET  /               - minimal chat UI (no auth, local demo only)
  GET  /health         - dependency health
  POST /chat           - blocking JSON chat (auth + rate-limited)
  POST /chat/stream    - SSE streaming chat (auth + rate-limited)
  POST /ingest/run     - force an ingest pass (auth)
  POST /ingest/upload  - upload documents and index them (auth)
  GET  /ingest/uploads - list uploaded documents (auth)
  POST /webhooks/*     - source push handlers (shared-secret)
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.ingest import router as ingest_router
from ingestion.workers.webhook_handler import router as webhooks_router
from ingestion.workers.poller import Poller
from ingestion.sources import default_connectors
from ingestion.pipeline import IngestionPipeline
from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store
from retrieval.retriever import Retriever
from retrieval.hybrid import HybridSearcher
from retrieval.reranker import Reranker
from retrieval.query_transform import QueryTransformer
from generation.generator import Generator
from generation.llm_router import LLMRouter
from generation.prompt_builder import PromptBuilder
from cache.semantic_cache import SemanticCache
from config.settings import get_settings
from observability.langfuse_client import get_tracer
from observability.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


def _build_connectors() -> list:
    return default_connectors()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info(
        "app_starting",
        embedding_backend=settings.embedding_backend,
        embedding_dim=settings.embedding_dim,
        reranker_enabled=settings.reranker_enabled,
        poller_enabled=settings.poller_enabled,
    )

    # Singletons
    embedder = Embedder()
    vector_store = VectorStore()
    bm25_store = BM25Store()
    llm_router = LLMRouter()

    # Retrieval stack — honor RERANKER_ENABLED (off on free-tier Docker)
    hybrid = HybridSearcher(embedder=embedder, vector_store=vector_store, bm25_store=bm25_store)
    reranker = Reranker()
    query_transformer = QueryTransformer(
        llm=llm_router if llm_router.providers else None,
        rewrite=bool(llm_router.providers),
        expansions=2 if llm_router.providers else 0,
    )
    retriever = Retriever(
        query_transformer=query_transformer,
        hybrid=hybrid,
        reranker=reranker,
        enable_rerank=settings.reranker_enabled,
    )

    # Generation
    prompt_builder = PromptBuilder()
    semantic_cache = SemanticCache()
    await semantic_cache.sync_corpus_version()
    generator = Generator(
        retriever=retriever,
        llm_router=llm_router,
        prompt_builder=prompt_builder,
        semantic_cache=semantic_cache,
        embedder=embedder,
    )

    # Ingestion (shares semantic cache so successful ingest bumps corpus version)
    pipeline = IngestionPipeline(
        connectors=_build_connectors(),
        embedder=embedder,
        vector_store=vector_store,
        bm25_store=bm25_store,
        semantic_cache=semantic_cache,
    )

    # Attach to app.state
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.bm25_store = bm25_store
    app.state.retriever = retriever
    app.state.generator = generator
    app.state.ingestion_pipeline = pipeline
    app.state.semantic_cache = semantic_cache
    app.state.tracer = get_tracer()

    # Warm critical paths (collection exists, BM25 loaded)
    try:
        await vector_store.ensure_collection()
        bm25_store.load()
    except Exception as e:
        log.warning("startup_warmup_failed", error=str(e))

    poller_task: asyncio.Task | None = None
    if settings.poller_enabled:
        poller = Poller(pipeline, interval_seconds=settings.poller_interval_seconds)
        poller_task = asyncio.create_task(poller.run_forever(), name="ingest-poller")
        app.state.poller = poller
        log.info("poller_task_started", interval_s=settings.poller_interval_seconds)

    log.info("app_ready")
    yield

    log.info("app_shutting_down")
    if poller_task is not None:
        poller = getattr(app.state, "poller", None)
        if poller is not None:
            poller._stop.set()
        poller_task.cancel()
        try:
            await poller_task
        except asyncio.CancelledError:
            pass
    try:
        app.state.tracer.flush()
    except Exception:
        pass


_settings = get_settings()

app = FastAPI(
    title="Ask",
    version="0.1.0",
    description="Ask questions over files you ingest: markdown, HTML, tickets, changelogs, and OpenAPI specs.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(webhooks_router)


# ---------- Chat UI ----------

_UI_HTML = (Path(__file__).resolve().parent / "chat_ui.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_UI_HTML)
