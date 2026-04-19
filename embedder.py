"""Embedder.

Three backends:
  1. sentence-transformers (local, ~400MB RAM — for local dev / bootstrap)
  2. fastembed (local, ~80MB RAM — lighter but still heavy for 512MB hosts)
  3. api (zero RAM — calls a free embedding API for query-time only)

For Render free tier: use EMBEDDING_BACKEND=api. The index was already
bootstrapped locally with sentence-transformers, so all document vectors
are in Qdrant. We only need to embed queries at search time (~1 API call
per user question).

The API backend uses Google's free embedding endpoint via the same
GOOGLE_API_KEY used for Gemini. If unavailable, falls back to Groq
(which doesn't have a native embed endpoint, so we use a trick: ask the
LLM to... no. Better: we use the free HuggingFace Inference API which
needs no key for small models).
"""
from __future__ import annotations
import asyncio
import os
from typing import Sequence

from config.settings import get_settings
from cache.embedding_cache import EmbeddingCache
from observability.logger import get_logger

log = get_logger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None
        self._cache = EmbeddingCache()
        self._backend = getattr(self._settings, "embedding_backend", "sentence-transformers")

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._backend == "fastembed":
            from fastembed import TextEmbedding
            log.info("loading_fastembed_model", model=self._settings.embedding_model)
            self._model = TextEmbedding(model_name=self._settings.embedding_model)
        elif self._backend == "api":
            self._model = "api"  # no local model needed
            log.info("using_api_embeddings")
        else:
            from sentence_transformers import SentenceTransformer
            log.info("loading_st_model", model=self._settings.embedding_model)
            self._model = SentenceTransformer(self._settings.embedding_model)
        return self._model

    async def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        cached = await self._cache.get_many(list(texts))
        missing_idx = [i for i, v in enumerate(cached) if v is None]
        if missing_idx:
            to_embed = [texts[i] for i in missing_idx]
            new_vecs = await self._encode(to_embed, batch_size)
            for i, vec in zip(missing_idx, new_vecs):
                cached[i] = vec
            await self._cache.set_many(to_embed, new_vecs)
        return cached

    async def embed_query(self, text: str) -> list[float]:
        prefixed = _BGE_QUERY_PREFIX + text
        vecs = await self._encode([prefixed], 1)
        return vecs[0]

    async def _encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        self._load_model()
        if self._backend == "api":
            return await self._encode_via_api(texts)
        else:
            return await asyncio.to_thread(self._encode_sync, texts, batch_size)

    async def _encode_via_api(self, texts: list[str]) -> list[list[float]]:
        """Call HuggingFace Inference API (free, no key needed for small models)."""
        import httpx
        model = self._settings.embedding_model
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                json={"inputs": texts, "options": {"wait_for_model": True}},
            )
            resp.raise_for_status()
            embeddings = resp.json()
            # HF returns list of list of floats for feature-extraction
            # Normalize to unit vectors (bge expects this)
            result = []
            for emb in embeddings:
                if isinstance(emb[0], list):
                    # Model returned token-level embeddings; mean-pool
                    import statistics
                    dim = len(emb[0])
                    pooled = [statistics.mean(emb[t][d] for t in range(len(emb))) for d in range(dim)]
                    result.append(_normalize(pooled))
                else:
                    result.append(_normalize(emb))
            return result

    def _encode_sync(self, texts: list[str], batch_size: int) -> list[list[float]]:
        if self._backend == "fastembed":
            embeddings = list(self._model.embed(texts, batch_size=batch_size))
            return [e.tolist() for e in embeddings]
        else:
            arr = self._model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True,
            )
            return arr.tolist()


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]
