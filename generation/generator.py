"""Answer generator.

The public entry point for end-to-end RAG:

    retriever -> prompt_builder -> llm_router -> citation audit
                      ^
                      └── semantic cache checked first

Two modes:
  - generate()       returns a full GeneratedAnswer (blocking)
  - stream()         yields StreamEvents (tokens + final meta event)

Streaming design: we send TOKEN events as they arrive from the LLM, then a
single final META event carrying citations + trace info. The client UI can
render the answer live and only fill in citation pills at the end.

Observability: every call produces a trace record (query, retrieval
stats, prompt size, latency per stage, LLM provider used, cache hit or
miss, citation audit).
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

from generation.llm_router import LLMRouter
from generation.prompt_builder import PromptBuilder, Citation
from generation.citation import audit_citations, CitationAudit
from retrieval.retriever import Retriever, RetrievalResult
from cache.semantic_cache import SemanticCache
from ingestion.embedder import Embedder
from observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class GeneratedAnswer:
    trace_id: str
    answer: str
    citations: list[Citation]
    audit: CitationAudit
    retrieval: RetrievalResult
    llm_provider: str | None
    cache_hit: bool
    timings_ms: dict[str, float]


@dataclass
class StreamEvent:
    type: Literal["token", "meta", "error"]
    data: dict = field(default_factory=dict)


class Generator:
    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        llm_router: LLMRouter | None = None,
        prompt_builder: PromptBuilder | None = None,
        semantic_cache: SemanticCache | None = None,
        embedder: Embedder | None = None,
        max_answer_tokens: int = 512,
        temperature: float = 0.2,
    ):
        self.retriever = retriever or Retriever(llm_for_transforms=llm_router)
        self.llm = llm_router or LLMRouter()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.cache = semantic_cache or SemanticCache()
        self.embedder = embedder or Embedder()
        self.max_answer_tokens = max_answer_tokens
        self.temperature = temperature

    async def generate(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
        source_types: list[str] | None = None,
        owner_id: str | None = None,
        use_cache: bool = True,
    ) -> GeneratedAnswer:
        trace_id = uuid.uuid4().hex[:12]
        timings: dict[str, float] = {}
        overall_start = time.perf_counter()

        # ---- Semantic cache lookup ----
        cache_hit = False
        q_vec_for_cache: list[float] | None = None
        if use_cache:
            t0 = time.perf_counter()
            q_vec_for_cache = await self.embedder.embed_query(question)
            cached = await self.cache.lookup(q_vec_for_cache)
            timings["cache_lookup"] = (time.perf_counter() - t0) * 1000
            if cached:
                log.info(
                    "cache_hit",
                    trace_id=trace_id,
                    similarity=round(cached.similarity, 4),
                    original_query=cached.query,
                )
                citations = [Citation(**c) for c in cached.citations]
                audit = audit_citations(cached.answer, citations)
                empty = RetrievalResult(
                    query=question, transformed=None, chunks=[],  # type: ignore[arg-type]
                    timings_ms={}, candidate_count_before_rerank=0,
                )
                timings["total"] = (time.perf_counter() - overall_start) * 1000
                return GeneratedAnswer(
                    trace_id=trace_id,
                    answer=cached.answer,
                    citations=citations,
                    audit=audit,
                    retrieval=empty,
                    llm_provider=None,
                    cache_hit=True,
                    timings_ms=timings,
                )

        # ---- Retrieval ----
        t0 = time.perf_counter()
        retrieval = await self.retriever.retrieve(
            question,
            top_k=5,
            candidate_k=20,
            source_types=source_types,
            owner_id=owner_id,
        )
        timings["retrieval"] = (time.perf_counter() - t0) * 1000

        # ---- Prompt ----
        t0 = time.perf_counter()
        prompt = self.prompt_builder.build(question, retrieval.chunks, history=history)
        timings["prompt_build"] = (time.perf_counter() - t0) * 1000

        # ---- LLM ----
        t0 = time.perf_counter()
        raw_answer = await self.llm.complete(
            prompt.text,
            max_tokens=self.max_answer_tokens,
            temperature=self.temperature,
        )
        timings["llm"] = (time.perf_counter() - t0) * 1000
        provider = self.llm.last_provider

        # ---- Citation audit ----
        audit = audit_citations(raw_answer, prompt.citations)
        if audit.invented_markers:
            log.warning(
                "invented_citations",
                trace_id=trace_id,
                markers=sorted(audit.invented_markers),
            )

        # ---- Cache write ----
        if use_cache and not cache_hit and retrieval.chunks:
            try:
                if q_vec_for_cache is None:
                    q_vec_for_cache = await self.embedder.embed_query(question)
                await self.cache.store(
                    question,
                    q_vec_for_cache,
                    audit.cleaned_answer,
                    [c.__dict__ for c in audit.used_citations],
                )
            except Exception as e:
                log.warning("cache_store_failed", error=str(e))

        timings["total"] = (time.perf_counter() - overall_start) * 1000

        log.info(
            "generate_complete",
            trace_id=trace_id,
            provider=provider,
            chunks_retrieved=len(retrieval.chunks),
            chunks_used=prompt.used_chunks,
            invented_citations=len(audit.invented_markers),
            coverage=round(audit.sentence_coverage, 3),
            **{f"t_{k}_ms": round(v, 1) for k, v in timings.items()},
        )

        return GeneratedAnswer(
            trace_id=trace_id,
            answer=audit.cleaned_answer,
            citations=prompt.citations,
            audit=audit,
            retrieval=retrieval,
            llm_provider=provider,
            cache_hit=False,
            timings_ms=timings,
        )

    async def stream(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
        source_types: list[str] | None = None,
        owner_id: str | None = None,
        use_cache: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Stream tokens, then emit a final 'meta' event with citations.

        Semantic cache: on hit we emit the cached answer as tokens (chunked)
        then meta — same wire format as a live stream.
        """
        trace_id = uuid.uuid4().hex[:12]
        timings: dict[str, float] = {}
        overall_start = time.perf_counter()

        if use_cache:
            t0 = time.perf_counter()
            try:
                q_vec = await self.embedder.embed_query(question)
                cached = await self.cache.lookup(q_vec)
            except Exception as e:
                log.warning("stream_cache_lookup_failed", error=str(e))
                cached = None
            timings["cache_lookup"] = (time.perf_counter() - t0) * 1000
            if cached:
                # Emit as token chunks so the UI path stays identical
                chunk_size = 24
                text = cached.answer
                for i in range(0, len(text), chunk_size):
                    yield StreamEvent(type="token", data={"text": text[i : i + chunk_size]})
                timings["total"] = (time.perf_counter() - overall_start) * 1000
                yield StreamEvent(
                    type="meta",
                    data={
                        "trace_id": trace_id,
                        "citations": cached.citations,
                        "invented_citations": [],
                        "coverage": 1.0,
                        "cache_hit": True,
                        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
                    },
                )
                return

        # Retrieval
        t0 = time.perf_counter()
        retrieval = await self.retriever.retrieve(
            question,
            top_k=5,
            candidate_k=20,
            source_types=source_types,
            owner_id=owner_id,
        )
        timings["retrieval"] = (time.perf_counter() - t0) * 1000

        prompt = self.prompt_builder.build(question, retrieval.chunks, history=history)

        collected: list[str] = []
        t0 = time.perf_counter()
        try:
            async for delta in self.llm.stream(
                prompt.text,
                max_tokens=self.max_answer_tokens,
                temperature=self.temperature,
            ):
                collected.append(delta)
                yield StreamEvent(type="token", data={"text": delta})
        except Exception as e:
            log.exception("stream_failed", trace_id=trace_id, error=str(e))
            yield StreamEvent(type="error", data={"message": "Stream failed. Please try again."})
            return
        timings["llm"] = (time.perf_counter() - t0) * 1000
        provider = self.llm.last_provider

        full_answer = "".join(collected)
        audit = audit_citations(full_answer, prompt.citations)
        timings["total"] = (time.perf_counter() - overall_start) * 1000

        if use_cache and retrieval.chunks:
            try:
                q_vec = await self.embedder.embed_query(question)
                await self.cache.store(
                    question,
                    q_vec,
                    audit.cleaned_answer,
                    [c.__dict__ for c in audit.used_citations],
                )
            except Exception as e:
                log.warning("stream_cache_store_failed", error=str(e))

        yield StreamEvent(
            type="meta",
            data={
                "trace_id": trace_id,
                "citations": [
                    {
                        "marker": c.marker,
                        "title": c.title,
                        "url": c.url,
                        "source_type": c.source_type,
                        "snippet": c.snippet,
                    }
                    for c in audit.used_citations
                ],
                "invented_citations": sorted(audit.invented_markers),
                "coverage": round(audit.sentence_coverage, 3),
                "cache_hit": False,
                "llm_provider": provider,
                "empty_index": not retrieval.chunks,
                "timings_ms": {k: round(v, 1) for k, v in timings.items()},
            },
        )
