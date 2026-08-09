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
from config.settings import get_settings
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
    title="Support RAG",
    version="0.1.0",
    description="Production-grade RAG chatbot for customer support over Flowpoint docs + tickets.",
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

_UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flowpoint Support</title>
<!-- Apply saved theme before render to avoid flash -->
<script>
  (function(){
    var t = localStorage.getItem('fp-theme');
    if (t === 'dark' || (!t && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
    }
  })();
</script>
<script>window.tailwind={config:{darkMode:'class'}}</script>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { font-family: 'Inter', system-ui, sans-serif; }

  #log::-webkit-scrollbar { width: 5px; }
  #log::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
  .dark #log::-webkit-scrollbar-thumb { background: #475569; }

  .dot { width:7px;height:7px;border-radius:50%;display:inline-block;animation:blink 1.2s infinite ease-in-out;background:#94a3b8; }
  .dark .dot { background: #64748b; }
  .dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
  @keyframes blink{0%,80%,100%{opacity:.2;transform:scale(.75)}40%{opacity:1;transform:scale(1)}}

  .appear{animation:up .2s ease}
  @keyframes up{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}

  /* Named classes for JS-created elements so dark: variants work without dynamic Tailwind scanning */
  .bot-bubble {
    background:#fff; border:1px solid #e2e8f0; color:#334155;
    border-radius:0 1rem 1rem 1rem; padding:.75rem 1rem;
    box-shadow:0 1px 3px rgba(0,0,0,.06); font-size:.875rem; max-width:32rem;
  }
  .dark .bot-bubble { background:#1e293b; border-color:#334155; color:#cbd5e1; }

  .typing-bubble {
    background:#fff; border:1px solid #e2e8f0;
    border-radius:0 1rem 1rem 1rem; padding:.875rem 1rem;
    box-shadow:0 1px 3px rgba(0,0,0,.06); display:flex; gap:6px; align-items:center;
  }
  .dark .typing-bubble { background:#1e293b; border-color:#334155; }

  .cite-bar { display:flex; flex-wrap:wrap; gap:6px; margin-top:.75rem; padding-top:.75rem; border-top:1px solid #e2e8f0; }
  .dark .cite-bar { border-top-color:#334155; }

  .cite-pill {
    display:inline-flex; align-items:center; gap:4px; font-size:.75rem;
    background:#eef2ff; color:#4338ca; border:1px solid #c7d2fe;
    padding:2px 8px; border-radius:9999px; text-decoration:none; transition:background .15s;
  }
  .cite-pill:hover { background:#e0e7ff; }
  .dark .cite-pill { background:rgba(67,56,202,.18); color:#a5b4fc; border-color:rgba(99,102,241,.3); }
  .dark .cite-pill:hover { background:rgba(67,56,202,.3); }

  .error-icon {
    width:2rem; height:2rem; border-radius:9999px; background:#fee2e2; color:#ef4444;
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
    font-size:.75rem; font-weight:700;
  }
  .dark .error-icon { background:rgba(127,29,29,.5); color:#f87171; }

  .error-bubble {
    background:#fef2f2; border:1px solid #fecaca; color:#b91c1c;
    border-radius:0 1rem 1rem 1rem; padding:.75rem 1rem;
    box-shadow:0 1px 3px rgba(0,0,0,.06); font-size:.875rem; max-width:32rem;
  }
  .dark .error-bubble { background:rgba(69,10,10,.5); border-color:rgba(127,29,29,.5); color:#fca5a5; }

  /* Markdown prose */
  .prose p{margin:0 0 .45em}.prose p:last-child{margin:0}
  .prose ol{padding-left:1.3em;list-style:decimal;margin-bottom:.4em}
  .prose ul{padding-left:1.3em;list-style:disc;margin-bottom:.4em}
  .prose li{margin-bottom:.15em}
  .prose strong{font-weight:600}
  .prose code{background:#f1f5f9;border-radius:4px;padding:1px 5px;font-size:.82em;font-family:ui-monospace,monospace}
  .dark .prose { color:#cbd5e1; }
  .dark .prose strong { color:#e2e8f0; }
  .dark .prose code { background:#0f172a; color:#a5b4fc; }

  /* Theme toggle icon visibility */
  .icon-sun  { display:none; }
  .icon-moon { display:block; }
  .dark .icon-sun  { display:block; }
  .dark .icon-moon { display:none; }
</style>
</head>
<body class="h-screen flex flex-col bg-slate-50 dark:bg-gray-950 overflow-hidden transition-colors duration-200">

<!-- Header -->
<header class="flex-shrink-0 bg-gradient-to-r from-indigo-600 via-blue-600 to-blue-500 shadow-lg">
  <div class="max-w-2xl mx-auto px-4 py-3 flex items-center gap-3">
    <div class="w-9 h-9 rounded-xl bg-white/20 flex items-center justify-center shadow-inner">
      <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
          d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/>
      </svg>
    </div>
    <div>
      <h1 class="text-white font-bold text-base leading-tight">Flowpoint Support</h1>
      <p class="text-blue-200 text-xs">AI assistant &middot; Docs &amp; tickets</p>
    </div>
    <div class="ml-auto flex items-center gap-3">
      <div class="flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
        <span class="text-blue-100 text-xs font-medium">Online</span>
      </div>
      <!-- Dark/Light toggle -->
      <button id="theme-btn" title="Toggle dark mode"
        class="w-8 h-8 rounded-lg bg-white/15 hover:bg-white/25 flex items-center justify-center transition-colors">
        <!-- Moon (shown in light mode) -->
        <svg class="icon-moon w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>
        </svg>
        <!-- Sun (shown in dark mode) -->
        <svg class="icon-sun w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707M17.657 17.657l-.707-.707M6.343 6.343l-.707-.707M12 8a4 4 0 100 8 4 4 0 000-8z"/>
        </svg>
      </button>
    </div>
  </div>
</header>

<!-- API key bar -->
<div class="flex-shrink-0 bg-white dark:bg-gray-900 border-b border-slate-200 dark:border-gray-700 transition-colors duration-200">
  <div class="max-w-2xl mx-auto px-4 py-2 flex items-center gap-2">
    <svg class="w-3.5 h-3.5 text-slate-400 dark:text-gray-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
        d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/>
    </svg>
    <input id="key" type="password" value=""
      class="flex-1 text-xs text-slate-500 dark:text-gray-400 bg-transparent outline-none" placeholder="API key">
  </div>
</div>

<!-- Messages -->
<div id="log" class="flex-1 overflow-y-auto px-4 py-5 transition-colors duration-200">
  <div id="msgs" class="max-w-2xl mx-auto flex flex-col gap-5">
    <!-- Welcome -->
    <div class="flex gap-3 appear">
      <div class="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow text-white text-xs font-bold">F</div>
      <div class="bot-bubble">
        Hi! I'm Flowpoint's AI support assistant. Ask me about billing, SSO setup, automations, the API, or anything else in the docs.
      </div>
    </div>
  </div>
</div>

<!-- Input -->
<footer class="flex-shrink-0 bg-white dark:bg-gray-900 border-t border-slate-200 dark:border-gray-700 shadow-[0_-4px_16px_rgba(0,0,0,0.06)] px-4 py-3 transition-colors duration-200">
  <div class="max-w-2xl mx-auto">
    <form id="f" class="flex items-end gap-2">
      <div class="flex-1 bg-slate-50 dark:bg-gray-800 border border-slate-200 dark:border-gray-600 rounded-2xl px-4 py-2.5
        focus-within:border-indigo-400 dark:focus-within:border-indigo-500 focus-within:ring-2 focus-within:ring-indigo-100 dark:focus-within:ring-indigo-900 transition-all">
        <textarea id="q" rows="1" autofocus placeholder="Ask anything about Flowpoint…"
          class="w-full bg-transparent text-sm text-slate-800 dark:text-gray-200 placeholder-slate-400 dark:placeholder-gray-500 outline-none resize-none leading-6"
          style="max-height:120px;overflow-y:auto"></textarea>
      </div>
      <button type="submit"
        class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-blue-600
          hover:from-indigo-700 hover:to-blue-700 flex items-center justify-center
          text-white shadow transition-all active:scale-95 hover:shadow-md flex-shrink-0">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
        </svg>
      </button>
    </form>
    <p class="text-center text-xs text-slate-400 dark:text-gray-500 mt-2">Answers cite Flowpoint docs and resolved support tickets</p>
  </div>
</footer>

<script>
const logEl = document.getElementById('log');
const msgs  = document.getElementById('msgs');
const form  = document.getElementById('f');
const qEl   = document.getElementById('q');

// ── Theme toggle ──────────────────────────────────────────────────────────────
document.getElementById('theme-btn').addEventListener('click', () => {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('fp-theme', isDark ? 'dark' : 'light');
});

// ── Textarea auto-grow ────────────────────────────────────────────────────────
qEl.addEventListener('input', () => {
  qEl.style.height = 'auto';
  qEl.style.height = Math.min(qEl.scrollHeight, 120) + 'px';
});
qEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function safeUrl(u) {
  try {
    const url = new URL(String(u || ''), window.location.origin);
    if (url.protocol === 'http:' || url.protocol === 'https:') return url.href;
  } catch (_) {}
  return '#';
}
function renderMarkdown(md) {
  const raw = marked.parse(String(md || ''), { async: false });
  return DOMPurify.sanitize(raw, { USE_PROFILES: { html: true } });
}
function scroll() { logEl.scrollTop = logEl.scrollHeight; }

function avatar() {
  const d = document.createElement('div');
  d.className = 'w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow text-white text-xs font-bold';
  d.textContent = 'F';
  return d;
}

// ── Message builders ──────────────────────────────────────────────────────────
function addUser(text) {
  const row = document.createElement('div');
  row.className = 'flex justify-end appear';
  const bubble = document.createElement('div');
  bubble.className = 'bg-gradient-to-br from-indigo-600 to-blue-600 text-white rounded-2xl rounded-tr-none px-4 py-3 shadow-sm text-sm max-w-sm';
  bubble.textContent = text;
  row.appendChild(bubble);
  msgs.appendChild(row); scroll();
}

function addTyping() {
  const row = document.createElement('div');
  row.id = 'typing'; row.className = 'flex gap-3 appear';
  const tb = document.createElement('div');
  tb.className = 'typing-bubble';
  tb.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  row.appendChild(avatar());
  row.appendChild(tb);
  msgs.appendChild(row); scroll();
}

function removeTyping() { const t = document.getElementById('typing'); if (t) t.remove(); }

function addBot() {
  const row = document.createElement('div');
  row.className = 'flex gap-3 appear';
  const bubble = document.createElement('div');
  bubble.className = 'bot-bubble';
  const prose = document.createElement('div');
  prose.className = 'prose';
  bubble.appendChild(prose);
  row.appendChild(avatar());
  row.appendChild(bubble);
  msgs.appendChild(row);
  return { prose, bubble };
}

function addError(msg) {
  removeTyping();
  const row = document.createElement('div');
  row.className = 'flex gap-3 appear';
  const icon = document.createElement('div');
  icon.className = 'error-icon';
  icon.textContent = '!';
  const bubble = document.createElement('div');
  bubble.className = 'error-bubble';
  bubble.textContent = msg;
  row.appendChild(icon);
  row.appendChild(bubble);
  msgs.appendChild(row); scroll();
}

function addCitations(citations, bubble) {
  if (!citations || !citations.length) return;
  const bar = document.createElement('div');
  bar.className = 'cite-bar';
  citations.forEach(c => {
    const a = document.createElement('a');
    a.href = safeUrl(c.url); a.target = '_blank'; a.rel = 'noopener noreferrer';
    a.className = 'cite-pill';
    a.innerHTML = '<span style="font-weight:600">[' + esc(c.marker) + ']</span> ' + esc(c.title);
    bar.appendChild(a);
  });
  bubble.appendChild(bar);
}

// ── Chat submit ───────────────────────────────────────────────────────────────
form.addEventListener('submit', async e => {
  e.preventDefault();
  const question = qEl.value.trim();
  if (!question) return;
  qEl.value = ''; qEl.style.height = 'auto';

  addUser(question);
  addTyping();

  let tokens = [], prose, bubble, botCreated = false;

  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: {'content-type': 'application/json', 'x-api-key': document.getElementById('key').value},
      body: JSON.stringify({question, history: []}),
    });
    if (!resp.ok) {
      addError(resp.status === 401 ? 'Invalid API key.' : 'Request failed (' + resp.status + '). Try again.');
      return;
    }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const parts = buf.split('\\n\\n');
      buf = parts.pop();
      for (const part of parts) {
        const line = part.split('\\n').find(l => l.startsWith('data:'));
        if (!line) continue;
        let payload;
        try { payload = JSON.parse(line.slice(5).trim()); } catch { continue; }

        if (payload.type === 'token') {
          if (!botCreated) { removeTyping(); ({prose, bubble} = addBot()); botCreated = true; }
          tokens.push(payload.text);
          prose.innerHTML = renderMarkdown(tokens.join(''));
          scroll();
        } else if (payload.type === 'meta' && bubble) {
          addCitations(payload.citations, bubble);
          scroll();
        } else if (payload.type === 'error') {
          addError(payload.message || 'Unknown error');
        }
      }
    }
  } catch (err) {
    addError('Connection error — is the server running?');
  }
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_UI_HTML)
