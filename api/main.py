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
  POST /webhooks/*     - source push handlers (shared-secret)
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.ingest import router as ingest_router
from ingestion.workers.webhook_handler import router as webhooks_router
from ingestion.pipeline import IngestionPipeline
from ingestion.connectors.markdown_docs import MarkdownDocsConnector
from ingestion.connectors.help_center_html import HelpCenterHTMLConnector
from ingestion.connectors.tickets import TicketsConnector
from ingestion.connectors.changelog import ChangelogConnector
from ingestion.connectors.openapi import OpenAPIConnector
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
from observability.langfuse_client import get_tracer
from observability.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


def _build_connectors() -> list:
    data = Path("data")
    connectors = []
    if (data / "sample_docs").exists():
        connectors.append(MarkdownDocsConnector(data / "sample_docs"))
    if (data / "sample_help_center").exists():
        connectors.append(HelpCenterHTMLConnector(data / "sample_help_center"))
    if (data / "sample_tickets" / "tickets.jsonl").exists():
        connectors.append(TicketsConnector(data / "sample_tickets" / "tickets.jsonl"))
    if (data / "CHANGELOG.md").exists():
        connectors.append(ChangelogConnector(data / "CHANGELOG.md"))
    if (data / "openapi.json").exists():
        connectors.append(OpenAPIConnector(data / "openapi.json"))
    return connectors


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("app_starting")

    # Singletons
    embedder = Embedder()
    vector_store = VectorStore()
    bm25_store = BM25Store()
    llm_router = LLMRouter()

    # Retrieval stack
    hybrid = HybridSearcher(embedder=embedder, vector_store=vector_store, bm25_store=bm25_store)
    reranker = Reranker()
    # Use the LLM router itself as the transformer's LLM — it satisfies the LLMAdapter protocol
    query_transformer = QueryTransformer(
        llm=llm_router if llm_router.providers else None,
        rewrite=bool(llm_router.providers),
        expansions=2 if llm_router.providers else 0,
    )
    retriever = Retriever(
        query_transformer=query_transformer,
        hybrid=hybrid,
        reranker=reranker,
        enable_rerank=True,
    )

    # Generation
    prompt_builder = PromptBuilder()
    semantic_cache = SemanticCache()
    generator = Generator(
        retriever=retriever,
        llm_router=llm_router,
        prompt_builder=prompt_builder,
        semantic_cache=semantic_cache,
        embedder=embedder,
    )

    # Ingestion
    pipeline = IngestionPipeline(
        connectors=_build_connectors(),
        embedder=embedder,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    # Attach to app.state
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.bm25_store = bm25_store
    app.state.retriever = retriever
    app.state.generator = generator
    app.state.ingestion_pipeline = pipeline
    app.state.tracer = get_tracer()

    # Warm critical paths (collection exists, BM25 loaded)
    try:
        await vector_store.ensure_collection()
        bm25_store.load()
    except Exception as e:
        log.warning("startup_warmup_failed", error=str(e))

    log.info("app_ready")
    yield

    log.info("app_shutting_down")
    try:
        app.state.tracer.flush()
    except Exception:
        pass


app = FastAPI(
    title="Support RAG",
    version="1.0.0",
    description="Production-grade RAG chatbot for customer support over Flowpoint docs + tickets.",
    lifespan=lifespan,
)

# CORS — tight by default; widen for your deployed frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in prod
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(webhooks_router)


# ---------- Minimal demo UI ----------

_UI_HTML = """<!doctype html>
<html lang=en><meta charset=utf-8>
<title>Flowpoint Support</title>
<style>
  body { font: 15px/1.5 system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #111; }
  h1 { font-size: 1.3rem; }
  #log { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; min-height: 240px; margin-bottom: 1rem; white-space: pre-wrap; }
  .user { color: #0b5; font-weight: 600; }
  .bot { color: #14a; }
  .meta { color: #888; font-size: 0.85em; margin-top: 0.5em; }
  .cite { display: inline-block; background: #eef; color: #224; padding: 1px 6px; border-radius: 10px; margin: 0 2px; font-size: 0.8em; text-decoration: none; }
  input[type=text] { width: 100%; padding: 0.6rem; font-size: 1rem; border-radius: 6px; border: 1px solid #bbb; }
  input[type=password] { padding: 0.4rem; font-size: 0.9rem; border-radius: 6px; border: 1px solid #bbb; width: 220px; }
  button { padding: 0.6rem 1rem; margin-left: 0.5rem; }
</style>
<h1>Flowpoint Support</h1>
<p><label>API Key: <input id=key type=password value=local-dev-key></label></p>
<div id=log></div>
<form id=f>
  <input id=q type=text placeholder="Ask anything about Flowpoint..." autofocus>
</form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const qEl = document.getElementById('q');
const keyEl = document.getElementById('key');

function append(html) { log.innerHTML += html; log.scrollTop = log.scrollHeight; }

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = qEl.value.trim();
  if (!question) return;
  qEl.value = '';
  append(`<div><span class="user">You:</span> ${escape(question)}</div>`);
  const botId = 'b' + Date.now();
  append(`<div><span class="bot">Bot:</span> <span id="${botId}"></span></div>`);
  const botEl = document.getElementById(botId);

  const resp = await fetch('/chat/stream', {
    method: 'POST',
    headers: {'content-type':'application/json', 'x-api-key': keyEl.value},
    body: JSON.stringify({question, history: []}),
  });
  if (!resp.ok) { botEl.textContent = `Error ${resp.status}`; return; }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream:true});
    const events = buf.split('\\n\\n');
    buf = events.pop();
    for (const ev of events) {
      const dataLine = ev.split('\\n').find(l => l.startsWith('data:'));
      if (!dataLine) continue;
      const payload = JSON.parse(dataLine.slice(5).trim());
      if (payload.type === 'token') { botEl.textContent += payload.text; }
      else if (payload.type === 'meta') { renderMeta(payload); }
      else if (payload.type === 'error') { botEl.textContent += ` [error: ${payload.message}]`; }
    }
    log.scrollTop = log.scrollHeight;
  }
});

function renderMeta(m) {
  if (!m.citations || !m.citations.length) return;
  let html = '<div class="meta">Sources: ';
  m.citations.forEach(c => {
    html += `<a class="cite" href="${c.url}" target=_blank title="${escape(c.snippet)}">[${c.marker}] ${escape(c.title)}</a>`;
  });
  html += '</div>';
  append(html);
}
function escape(s) { return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
</script>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_UI_HTML)
