"""Hybrid search: vector + BM25 + Reciprocal Rank Fusion.

Why RRF instead of linear weighted combination?
  - Vector scores (cosine, 0..1) and BM25 scores (unbounded, corpus-relative)
    live on totally different scales. Tuning weights per-corpus is painful.
  - RRF ignores scores entirely — it only uses *ranks*. Each doc's fused
    score is sum(1 / (k + rank_i)) across retrievers. k=60 is a well-
    established default from the original RRF paper.
  - Result: hybrid works out-of-the-box across domains. We can still layer
    a weighted combo later if an eval shows it helps.

Multi-query expansion is folded in here: expansions run as additional
retrievers and contribute their own ranks to the fusion.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field

from retrieval.bm25_store import BM25Store
from retrieval.vector_store import VectorStore
from retrieval.query_transform import TransformedQuery
from ingestion.embedder import Embedder
from observability.logger import get_logger

log = get_logger(__name__)

RRF_K = 60  # standard default; higher = less aggressive re-weighting


@dataclass
class Candidate:
    chunk_id: str
    text: str
    payload: dict
    # Ranks from each retriever (1-indexed). Missing = didn't appear.
    ranks: dict[str, int] = field(default_factory=dict)
    # Raw scores from each retriever for debugging / observability
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def rrf_score(self) -> float:
        return sum(1.0 / (RRF_K + r) for r in self.ranks.values())


class HybridSearcher:
    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self._bm25_loaded = False

    def _ensure_bm25_loaded(self) -> None:
        if not self._bm25_loaded:
            self.bm25_store.load()
            self._bm25_loaded = True

    async def search(
        self,
        tq: TransformedQuery,
        *,
        top_k_per_retriever: int = 20,
        final_k: int = 20,
        source_types: list[str] | None = None,
    ) -> list[Candidate]:
        """Return top `final_k` fused candidates."""
        self._ensure_bm25_loaded()

        # Build the list of (retriever_name, search_coroutine) pairs to run in parallel
        tasks: list[tuple[str, asyncio.Task]] = []

        # Vector search on the rewritten query
        tasks.append((
            "vec:primary",
            asyncio.create_task(self._vector_search(tq.rewritten, top_k_per_retriever, source_types)),
        ))
        # BM25 on the rewritten query
        tasks.append((
            "bm25:primary",
            asyncio.create_task(self._bm25_search(tq.rewritten, top_k_per_retriever)),
        ))

        # Multi-query expansions each go through vector search (not BM25 —
        # expansions share content keywords, adding BM25 here just amplifies noise)
        for i, exp in enumerate(tq.expansions):
            tasks.append((
                f"vec:exp{i}",
                asyncio.create_task(self._vector_search(exp, top_k_per_retriever, source_types)),
            ))

        # HyDE: embed the hallucinated doc and run as another vector retriever
        if tq.hyde_doc:
            tasks.append((
                "vec:hyde",
                asyncio.create_task(self._vector_search_text(tq.hyde_doc, top_k_per_retriever, source_types)),
            ))

        # Collect
        results: dict[str, list[dict]] = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                log.warning("retriever_failed", name=name, error=str(e))
                results[name] = []

        fused = self._rrf_fuse(results)
        return fused[:final_k]

    async def _vector_search(
        self, query_text: str, top_k: int, source_types: list[str] | None
    ) -> list[dict]:
        vec = await self.embedder.embed_query(query_text)
        return await self.vector_store.search(vec, top_k=top_k, source_types=source_types)

    async def _vector_search_text(
        self, doc_text: str, top_k: int, source_types: list[str] | None
    ) -> list[dict]:
        # HyDE: embed as document (no query prefix) rather than query
        vecs = await self.embedder.embed_documents([doc_text])
        return await self.vector_store.search(vecs[0], top_k=top_k, source_types=source_types)

    async def _bm25_search(self, query_text: str, top_k: int) -> list[dict]:
        # BM25 is CPU-bound and fast; offload to thread to not block the loop
        return await asyncio.to_thread(self.bm25_store.search, query_text, top_k)

    def _rrf_fuse(self, results: dict[str, list[dict]]) -> list[Candidate]:
        merged: dict[str, Candidate] = {}
        for retriever_name, hits in results.items():
            for rank, hit in enumerate(hits, start=1):
                chunk_id = hit.get("chunk_id") or hit.get("doc_id") or f"unknown:{rank}"
                c = merged.get(chunk_id)
                if c is None:
                    c = Candidate(
                        chunk_id=chunk_id,
                        text=hit.get("text", ""),
                        payload=hit,
                    )
                    merged[chunk_id] = c
                c.ranks[retriever_name] = rank
                if "score" in hit:
                    c.scores[retriever_name] = float(hit["score"])
        return sorted(merged.values(), key=lambda x: x.rrf_score, reverse=True)
