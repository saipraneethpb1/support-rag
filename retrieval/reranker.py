"""Cross-encoder reranker.

A cross-encoder scores (query, document) pairs jointly — much higher
fidelity than bi-encoder cosine similarity, but O(N) per query instead of
O(1). That's why we only rerank the top ~20 candidates, not the whole
corpus.

Empirically this is the single biggest retrieval-quality lever after
hybrid search. We use `bge-reranker-base` (free, local, ~100MB model).

For production at real scale you'd swap to `bge-reranker-v2-m3` (better
but 2GB) or a hosted reranker (Cohere, Voyage).
"""
from __future__ import annotations
import asyncio
from typing import Iterable

from config.settings import get_settings
from observability.logger import get_logger

log = get_logger(__name__)


class Reranker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            log.info("loading_reranker_model", model=self._settings.reranker_model)
            self._model = CrossEncoder(self._settings.reranker_model, max_length=512)
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        *,
        top_k: int = 5,
        text_key: str = "text",
    ) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c.get(text_key, "")) for c in candidates]
        scores = await asyncio.to_thread(self._score_sync, pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _score_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load_model()
        # CrossEncoder.predict returns numpy array
        return model.predict(pairs, batch_size=16, show_progress_bar=False).tolist()
