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

_UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ask</title>
<script>
  (function () {
    var t = localStorage.getItem('fp-theme');
    if (t === 'dark' || (!t && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
    }
  })();
</script>
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
  :root {
    --bg0: #f4f5f7;
    --bg1: #eceef2;
    --surface: rgba(255, 255, 255, 0.72);
    --ink: #16181d;
    --muted: #6b7280;
    --line: rgba(22, 24, 29, 0.08);
    --accent: #1f4f46;
    --accent-soft: rgba(31, 79, 70, 0.08);
    --user: #16181d;
    --user-ink: #f7f8fa;
    --danger: #8f2f2f;
    --danger-bg: rgba(143, 47, 47, 0.08);
    --shadow: 0 1px 0 rgba(22, 24, 29, 0.04);
    --radius: 14px;
    --font: "Instrument Sans", "Segoe UI", sans-serif;
    --display: "Instrument Serif", Georgia, serif;
  }
  .dark {
    --bg0: #0e1014;
    --bg1: #151821;
    --surface: rgba(24, 27, 34, 0.78);
    --ink: #e8eaef;
    --muted: #9aa1ad;
    --line: rgba(232, 234, 239, 0.1);
    --accent: #8fb9ad;
    --accent-soft: rgba(143, 185, 173, 0.12);
    --user: #e8eaef;
    --user-ink: #12141a;
    --danger: #f0a0a0;
    --danger-bg: rgba(143, 47, 47, 0.22);
    --shadow: 0 1px 0 rgba(0, 0, 0, 0.35);
  }

  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    color: var(--ink);
    font-family: var(--font);
    background:
      radial-gradient(1200px 600px at 12% -10%, rgba(31, 79, 70, 0.09), transparent 55%),
      radial-gradient(900px 500px at 100% 0%, rgba(22, 24, 29, 0.05), transparent 50%),
      linear-gradient(180deg, var(--bg0), var(--bg1));
    overflow: hidden;
  }
  .dark body,
  body {
    transition: background 0.3s ease, color 0.3s ease;
  }

  .shell {
    height: 100%;
    display: flex;
    flex-direction: column;
    max-width: 720px;
    margin: 0 auto;
  }

  header {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 1.25rem 1.25rem 0.85rem;
    animation: fade-in 0.5s ease both;
  }
  .brand {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;
  }
  .brand strong {
    font-family: var(--display);
    font-weight: 400;
    font-size: 1.65rem;
    letter-spacing: -0.02em;
    line-height: 1;
  }
  .brand span {
    color: var(--muted);
    font-size: 0.8rem;
    letter-spacing: 0.01em;
  }
  .header-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .ghost-btn {
    appearance: none;
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--muted);
    width: 2.25rem;
    height: 2.25rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    backdrop-filter: blur(10px);
    transition: color 0.2s ease, border-color 0.2s ease, transform 0.15s ease;
  }
  .ghost-btn:hover { color: var(--ink); border-color: rgba(22, 24, 29, 0.18); }
  .dark .ghost-btn:hover { border-color: rgba(232, 234, 239, 0.22); }
  .ghost-btn:active { transform: scale(0.96); }
  .ghost-btn svg { width: 1rem; height: 1rem; }
  .icon-sun { display: none; }
  .dark .icon-sun { display: block; }
  .dark .icon-moon { display: none; }

  .key-row {
    flex-shrink: 0;
    margin: 0 1.25rem 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.85rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--surface);
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow);
    animation: fade-in 0.55s ease 0.05s both;
  }
  .key-row svg {
    width: 0.9rem;
    height: 0.9rem;
    color: var(--muted);
    flex-shrink: 0;
  }
  .key-row input {
    flex: 1;
    border: 0;
    outline: none;
    background: transparent;
    color: var(--muted);
    font: inherit;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
  }
  .key-row input::placeholder { color: var(--muted); opacity: 0.7; }

  #log {
    flex: 1;
    overflow-y: auto;
    padding: 1rem 1.25rem 1.5rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(107, 114, 128, 0.35) transparent;
  }
  #msgs {
    display: flex;
    flex-direction: column;
    gap: 1.35rem;
    min-height: 100%;
  }

  .row { display: flex; gap: 0.85rem; align-items: flex-start; }
  .row.user { justify-content: flex-end; }
  .appear { animation: rise 0.35s cubic-bezier(0.22, 1, 0.36, 1) both; }

  .mark {
    width: 1.75rem;
    height: 1.75rem;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--accent);
    font-family: var(--display);
    font-size: 0.95rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 0.15rem;
  }

  .bot-msg {
    max-width: 38rem;
    color: var(--ink);
    font-size: 0.95rem;
    line-height: 1.55;
    padding-top: 0.15rem;
  }
  .user-msg {
    max-width: 28rem;
    background: var(--user);
    color: var(--user-ink);
    padding: 0.75rem 1rem;
    border-radius: var(--radius) var(--radius) 4px var(--radius);
    font-size: 0.92rem;
    line-height: 1.45;
  }

  .typing {
    display: inline-flex;
    gap: 0.35rem;
    align-items: center;
    padding: 0.55rem 0.1rem;
    color: var(--muted);
  }
  .typing i {
    width: 0.35rem;
    height: 0.35rem;
    border-radius: 999px;
    background: currentColor;
    display: block;
    animation: pulse 1.2s ease-in-out infinite;
  }
  .typing i:nth-child(2) { animation-delay: 0.15s; }
  .typing i:nth-child(3) { animation-delay: 0.3s; }

  .cite-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem 0.85rem;
    margin-top: 0.9rem;
    padding-top: 0.85rem;
    border-top: 1px solid var(--line);
  }
  .cite-link {
    color: var(--accent);
    text-decoration: none;
    font-size: 0.78rem;
    letter-spacing: 0.01em;
    border-bottom: 1px solid transparent;
    transition: border-color 0.15s ease, opacity 0.15s ease;
  }
  .cite-link:hover { border-bottom-color: currentColor; }
  .cite-link b { font-weight: 500; margin-right: 0.25rem; opacity: 0.7; }

  .error-row { display: flex; gap: 0.75rem; align-items: flex-start; }
  .error-msg {
    color: var(--danger);
    background: var(--danger-bg);
    border: 1px solid rgba(143, 47, 47, 0.18);
    border-radius: var(--radius);
    padding: 0.75rem 0.95rem;
    font-size: 0.88rem;
    max-width: 32rem;
  }

  .prose p { margin: 0 0 0.55em; }
  .prose p:last-child { margin: 0; }
  .prose ol, .prose ul { margin: 0 0 0.55em; padding-left: 1.2em; }
  .prose li { margin: 0.15em 0; }
  .prose strong { font-weight: 600; }
  .prose code {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.84em;
    background: var(--accent-soft);
    padding: 0.1em 0.35em;
    border-radius: 4px;
  }
  .prose a { color: var(--accent); }

  footer {
    flex-shrink: 0;
    padding: 0.75rem 1.25rem 1.15rem;
    animation: fade-in 0.6s ease 0.08s both;
  }
  form {
    display: flex;
    align-items: flex-end;
    gap: 0.65rem;
    padding: 0.55rem 0.55rem 0.55rem 1rem;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: var(--surface);
    backdrop-filter: blur(14px);
    box-shadow: var(--shadow);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  form:focus-within {
    border-color: rgba(31, 79, 70, 0.35);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  .dark form:focus-within { border-color: rgba(143, 185, 173, 0.4); }
  textarea {
    flex: 1;
    border: 0;
    outline: none;
    resize: none;
    background: transparent;
    color: var(--ink);
    font: inherit;
    font-size: 0.95rem;
    line-height: 1.45;
    max-height: 120px;
    padding: 0.45rem 0;
  }
  textarea::placeholder { color: var(--muted); }
  .send {
    appearance: none;
    border: 0;
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 999px;
    background: var(--accent);
    color: #f7faf9;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    flex-shrink: 0;
    transition: transform 0.15s ease, opacity 0.15s ease;
  }
  .dark .send { color: #0e1014; }
  .send:hover { opacity: 0.92; }
  .send:active { transform: scale(0.96); }
  .send svg { width: 1rem; height: 1rem; }
  .hint {
    margin: 0.65rem 0 0;
    text-align: center;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.02em;
  }

  @keyframes rise {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: none; }
  }
  @keyframes fade-in {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: none; }
  }
  @keyframes pulse {
    0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
    40% { opacity: 1; transform: translateY(-1px); }
  }

  @media (max-width: 640px) {
    header, #log, footer { padding-left: 1rem; padding-right: 1rem; }
    .key-row { margin-left: 1rem; margin-right: 1rem; }
    .brand strong { font-size: 1.45rem; }
  }
</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand">
      <strong>Ask</strong>
      <span>Your files</span>
    </div>
    <div class="header-actions">
      <button id="theme-btn" class="ghost-btn" type="button" title="Toggle theme" aria-label="Toggle theme">
        <svg class="icon-moon" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75"
            d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>
        </svg>
        <svg class="icon-sun" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75"
            d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707M17.657 17.657l-.707-.707M6.343 6.343l-.707-.707M12 8a4 4 0 100 8 4 4 0 000-8z"/>
        </svg>
      </button>
    </div>
  </header>

  <div class="key-row">
    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75"
        d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/>
    </svg>
    <input id="key" type="password" value="" placeholder="API key" autocomplete="off" spellcheck="false">
  </div>

  <div id="log">
    <div id="msgs">
      <div class="row appear">
        <div class="mark" aria-hidden="true">A</div>
        <div class="bot-msg">
          This answers from files you ingest — markdown, HTML, tickets, changelogs, or an OpenAPI spec.
          Put your own files in the <code>data/</code> folder, run ingest, then ask. I cite the source for each claim.
        </div>
      </div>
    </div>
  </div>

  <footer>
    <form id="f">
      <textarea id="q" rows="1" autofocus placeholder="Ask about your files…"></textarea>
      <button class="send" type="submit" aria-label="Send">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.9" d="M5 12h14M13 5l7 7-7 7"/>
        </svg>
      </button>
    </form>
    <p class="hint">Answers come only from ingested files, with citations</p>
  </footer>
</div>

<script>
const logEl = document.getElementById('log');
const msgs  = document.getElementById('msgs');
const form  = document.getElementById('f');
const qEl   = document.getElementById('q');

document.getElementById('theme-btn').addEventListener('click', () => {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('fp-theme', isDark ? 'dark' : 'light');
});

qEl.addEventListener('input', () => {
  qEl.style.height = 'auto';
  qEl.style.height = Math.min(qEl.scrollHeight, 120) + 'px';
});
qEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});

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

function mark() {
  const d = document.createElement('div');
  d.className = 'mark';
  d.setAttribute('aria-hidden', 'true');
  d.textContent = 'A';
  return d;
}

function addUser(text) {
  const row = document.createElement('div');
  row.className = 'row user appear';
  const bubble = document.createElement('div');
  bubble.className = 'user-msg';
  bubble.textContent = text;
  row.appendChild(bubble);
  msgs.appendChild(row); scroll();
}

function addTyping() {
  const row = document.createElement('div');
  row.id = 'typing';
  row.className = 'row appear';
  const tb = document.createElement('div');
  tb.className = 'typing';
  tb.innerHTML = '<i></i><i></i><i></i>';
  row.appendChild(mark());
  row.appendChild(tb);
  msgs.appendChild(row); scroll();
}

function removeTyping() { const t = document.getElementById('typing'); if (t) t.remove(); }

function addBot() {
  const row = document.createElement('div');
  row.className = 'row appear';
  const bubble = document.createElement('div');
  bubble.className = 'bot-msg';
  const prose = document.createElement('div');
  prose.className = 'prose';
  bubble.appendChild(prose);
  row.appendChild(mark());
  row.appendChild(bubble);
  msgs.appendChild(row);
  return { prose, bubble };
}

function addError(msg) {
  removeTyping();
  const row = document.createElement('div');
  row.className = 'error-row appear';
  const bubble = document.createElement('div');
  bubble.className = 'error-msg';
  bubble.textContent = msg;
  row.appendChild(bubble);
  msgs.appendChild(row); scroll();
}

function addCitations(citations, bubble) {
  if (!citations || !citations.length) return;
  const bar = document.createElement('div');
  bar.className = 'cite-bar';
  citations.forEach(c => {
    const a = document.createElement('a');
    a.href = safeUrl(c.url);
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.className = 'cite-link';
    a.innerHTML = '<b>[' + esc(c.marker) + ']</b>' + esc(c.title);
    bar.appendChild(a);
  });
  bubble.appendChild(bar);
}

form.addEventListener('submit', async e => {
  e.preventDefault();
  const question = qEl.value.trim();
  if (!question) return;
  qEl.value = '';
  qEl.style.height = 'auto';

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
