"""Embedding cache.

Hashes chunk text -> vector. Avoids recomputing embeddings on re-ingest
when content didn't actually change. Critical for incremental updates.

Falls back gracefully to no-op if Redis is unavailable so local dev
doesn't break.
"""
from __future__ import annotations
import hashlib
import json

import redis.asyncio as aioredis

from config.settings import get_settings
from observability.logger import get_logger

log = get_logger(__name__)


def _hash_text(text: str, *, model: str, backend: str, dim: int) -> str:
    h = hashlib.sha256()
    h.update(backend.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(dim).encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return f"emb:{h.hexdigest()}"


class EmbeddingCache:
    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.effective_embedding_model_id()
        self._backend = settings.embedding_backend
        self._dim = settings.embedding_dim
        self._client: aioredis.Redis | None = None
        self._url = settings.redis_url
        self._available = True

    def _key(self, text: str) -> str:
        return _hash_text(
            text, model=self._model, backend=self._backend, dim=self._dim
        )

    async def _get_client(self) -> aioredis.Redis | None:
        if not self._available:
            return None
        if self._client is None:
            try:
                self._client = aioredis.from_url(
                    self._url, encoding="utf-8", decode_responses=True
                )
                await self._client.ping()
            except Exception as e:
                log.warning("redis_unavailable", error=str(e))
                self._available = False
                return None
        return self._client

    async def get_many(self, texts: list[str]) -> list[list[float] | None]:
        client = await self._get_client()
        if client is None:
            return [None] * len(texts)
        keys = [self._key(t) for t in texts]
        raw = await client.mget(keys)
        return [json.loads(v) if v else None for v in raw]

    async def set_many(self, texts: list[str], vectors: list[list[float]]) -> None:
        client = await self._get_client()
        if client is None:
            return
        pipe = client.pipeline()
        for t, v in zip(texts, vectors):
            pipe.set(self._key(t), json.dumps(v), ex=60 * 60 * 24 * 30)
        await pipe.execute()
