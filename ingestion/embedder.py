"""Embedder. Two backends: sentence-transformers (heavy) or fastembed (light)."""
from __future__ import annotations
import asyncio
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
            new_vecs = await asyncio.to_thread(self._encode_sync, to_embed, batch_size)
            for i, vec in zip(missing_idx, new_vecs):
                cached[i] = vec
            await self._cache.set_many(to_embed, new_vecs)
        return cached

    async def embed_query(self, text: str) -> list[float]:
        prefixed = _BGE_QUERY_PREFIX + text
        vecs = await asyncio.to_thread(self._encode_sync, [prefixed], 1)
        return vecs[0]

    def _encode_sync(self, texts: list[str], batch_size: int) -> list[list[float]]:
        model = self._load_model()
        if self._backend == "fastembed":
            embeddings = list(model.embed(texts, batch_size=batch_size))
            return [e.tolist() for e in embeddings]
        else:
            arr = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            return arr.tolist()