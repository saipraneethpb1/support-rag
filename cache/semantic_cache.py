"""Semantic query cache.

If a new query is semantically near a previously-answered query, return
the cached answer instead of running the full RAG pipeline. Huge cost
savings on high-volume support bots where ~30% of questions are
rephrasings of FAQs.

Design:
  - Key-value: Redis hash -> serialized {query, answer, citations, ts}
  - Index: a small in-memory list of (vec, key) we scan for cosine > threshold
    (Redis Search / HNSW would be nicer at scale; in-memory is fine below 50K
     cached queries.)
  - Threshold is tunable; 0.97 is a conservative default. Below 0.95 we start
    seeing semantically-different queries collide.

Gotcha: this cache is correct only when answers are deterministic w.r.t.
the query *and* the corpus hasn't changed. We include a corpus version
tag in the cache key so that a re-ingest naturally invalidates.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass

import redis.asyncio as aioredis

from config.settings import get_settings
from observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class CachedAnswer:
    query: str
    answer: str
    citations: list[dict]
    created_at: float
    similarity: float = 0.0


class SemanticCache:
    def __init__(self, *, threshold: float = 0.97, corpus_version: str = "v1"):
        self._settings = get_settings()
        self._threshold = threshold
        self._corpus_version = corpus_version
        self._client: aioredis.Redis | None = None
        self._available = True

        # In-memory vector index mirror. Populated on first use from Redis.
        # Each entry: (vec: list[float], key: str)
        self._index: list[tuple[list[float], str]] = []
        self._loaded = False

    def _ns(self, suffix: str) -> str:
        return f"semcache:{self._corpus_version}:{suffix}"

    async def _get_client(self) -> aioredis.Redis | None:
        if not self._available:
            return None
        if self._client is None:
            try:
                self._client = aioredis.from_url(
                    self._settings.redis_url, encoding="utf-8", decode_responses=True
                )
                await self._client.ping()
            except Exception as e:
                log.warning("semcache_redis_unavailable", error=str(e))
                self._available = False
                return None
        return self._client

    async def _load_index(self) -> None:
        if self._loaded:
            return
        client = await self._get_client()
        if client is None:
            self._loaded = True
            return
        keys = await client.keys(self._ns("vec:*"))
        if keys:
            raw = await client.mget(keys)
            for key, v in zip(keys, raw):
                if v:
                    try:
                        self._index.append((json.loads(v), key.replace(":vec:", ":ans:")))
                    except Exception:
                        continue
        self._loaded = True
        log.info("semcache_index_loaded", entries=len(self._index))

    async def lookup(self, query_vec: list[float]) -> CachedAnswer | None:
        await self._load_index()
        if not self._index:
            return None
        # Since vectors are L2-normalized, dot product == cosine similarity
        best_key, best_sim = None, -1.0
        for vec, key in self._index:
            sim = sum(a * b for a, b in zip(vec, query_vec))
            if sim > best_sim:
                best_sim, best_key = sim, key
        if best_key is None or best_sim < self._threshold:
            return None

        client = await self._get_client()
        if client is None:
            return None
        raw = await client.get(best_key)
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return CachedAnswer(similarity=best_sim, **data)
        except Exception:
            return None

    async def store(
        self, query: str, query_vec: list[float], answer: str, citations: list[dict]
    ) -> None:
        client = await self._get_client()
        if client is None:
            return
        # Stable key per query text (rounded) to avoid duplicates on identical queries
        import hashlib
        key_suffix = hashlib.sha1(query.lower().strip().encode()).hexdigest()[:16]
        ans_key = self._ns(f"ans:{key_suffix}")
        vec_key = self._ns(f"vec:{key_suffix}")
        payload = {
            "query": query,
            "answer": answer,
            "citations": citations,
            "created_at": time.time(),
        }
        ttl = 60 * 60 * 24 * 7  # 7 days
        pipe = client.pipeline()
        pipe.set(ans_key, json.dumps(payload), ex=ttl)
        pipe.set(vec_key, json.dumps(query_vec), ex=ttl)
        await pipe.execute()
        self._index.append((query_vec, ans_key))
