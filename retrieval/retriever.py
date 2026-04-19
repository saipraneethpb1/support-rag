"""Retriever facade.

This is the single entry point the generation layer (Step 4) will use.
It composes: query transform -> hybrid search -> rerank -> return.

Every stage is observable: we return a RetrievalResult that carries per-
stage timings, candidate counts, and the ranks/scores of the final set
so we can trace WHY a given chunk was chosen.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

from retrieval.query_transform import QueryTransformer, LLMAdapter, TransformedQuery
from retrieval.hybrid import HybridSearcher, Candidate
from retrieval.reranker import Reranker
from observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    url: str
    source_type: str
    rerank_score: float
    rrf_score: float
    retriever_hits: dict[str, int]  # which retrievers ranked this, and at what rank
    payload: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    transformed: TransformedQuery
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float]
    candidate_count_before_rerank: int


class Retriever:
    def __init__(
        self,
        *,
        query_transformer: QueryTransformer | None = None,
        hybrid: HybridSearcher | None = None,
        reranker: Reranker | None = None,
        llm_for_transforms: LLMAdapter | None = None,
        enable_rerank: bool = True,
    ):
        self.query_transformer = query_transformer or QueryTransformer(
            llm=llm_for_transforms, rewrite=llm_for_transforms is not None,
            expansions=2 if llm_for_transforms else 0,
            use_hyde=False,
        )
        self.hybrid = hybrid or HybridSearcher()
        self.reranker = reranker or Reranker()
        self.enable_rerank = enable_rerank

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
        source_types: list[str] | None = None,
    ) -> RetrievalResult:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        tq = await self.query_transformer.transform(query)
        timings["transform"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        candidates: list[Candidate] = await self.hybrid.search(
            tq,
            top_k_per_retriever=candidate_k,
            final_k=candidate_k,
            source_types=source_types,
        )
        timings["hybrid_search"] = (time.perf_counter() - t0) * 1000
        candidate_count = len(candidates)

        # If rerank disabled, just take top_k by RRF
        if not self.enable_rerank or not candidates:
            final = candidates[:top_k]
            for c in final:
                c.payload.setdefault("rerank_score", c.rrf_score)
            chunks = [_to_retrieved(c) for c in final]
            timings["rerank"] = 0.0
            return RetrievalResult(
                query=query, transformed=tq, chunks=chunks,
                timings_ms=timings, candidate_count_before_rerank=candidate_count,
            )

        t0 = time.perf_counter()
        # Rerank operates on plain dicts; wrap + unwrap
        dicts = [{"text": c.text, "_ref": c} for c in candidates]
        reranked_dicts = await self.reranker.rerank(tq.rewritten, dicts, top_k=top_k)
        timings["rerank"] = (time.perf_counter() - t0) * 1000

        chunks: list[RetrievedChunk] = []
        for d in reranked_dicts:
            c: Candidate = d["_ref"]
            c.payload["rerank_score"] = d["rerank_score"]
            chunks.append(_to_retrieved(c))

        log.info(
            "retrieval_complete",
            query=query,
            candidates=candidate_count,
            returned=len(chunks),
            **{f"t_{k}_ms": round(v, 1) for k, v in timings.items()},
        )

        return RetrievalResult(
            query=query, transformed=tq, chunks=chunks,
            timings_ms=timings, candidate_count_before_rerank=candidate_count,
        )


def _to_retrieved(c: Candidate) -> RetrievedChunk:
    p = c.payload
    return RetrievedChunk(
        chunk_id=c.chunk_id,
        doc_id=p.get("doc_id", ""),
        text=c.text,
        title=p.get("title", ""),
        url=p.get("url", ""),
        source_type=p.get("source_type", ""),
        rerank_score=float(p.get("rerank_score", 0.0)),
        rrf_score=c.rrf_score,
        retriever_hits=dict(c.ranks),
        payload=p,
    )
