"""Qdrant vector store wrapper.

Responsibilities:
  - Idempotent collection creation
  - Upsert chunks (replacing prior version for the same doc_id)
  - Delete by doc_id
  - Search with metadata filters

We key vector point IDs on chunk_id (UUID5 over chunk_id string) so
re-indexing is naturally idempotent.
"""
from __future__ import annotations
import uuid
from typing import Iterable

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from config.settings import get_settings
from ingestion.chunkers import Chunk
from observability.logger import get_logger
from retrieval.visibility import owner_visible

log = get_logger(__name__)

_NAMESPACE = uuid.UUID("6f1c2c4a-7e3f-4d3b-9b6a-21d8e2c0a1b2")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class VectorStore:
    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)
        self._collection = s.qdrant_collection
        self._dim = s.embedding_dim
        self.available = False

    async def ensure_collection(self) -> bool:
        try:
            existing = await self._client.get_collections()
            names = {c.name for c in existing.collections}
            if self._collection in names:
                info = await self._client.get_collection(self._collection)
                # qdrant-client shapes vary slightly by version; be defensive
                vectors = getattr(getattr(info, "config", None), "params", None)
                vectors = getattr(vectors, "vectors", None)
                size = getattr(vectors, "size", None)
                if size is not None and int(size) != int(self._dim):
                    raise RuntimeError(
                        f"Qdrant collection '{self._collection}' has dim={size}, "
                        f"but EMBEDDING_DIM={self._dim}. Recreate the collection or "
                        f"align EMBEDDING_DIM / embedding backend."
                    )
                self.available = True
                return True
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )
            # Indexes for filterable metadata fields
            for field in ("source_type", "doc_id", "url", "owner"):
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            log.info("qdrant_collection_created", name=self._collection, dim=self._dim)
            self.available = True
            return True
        except Exception as e:
            self.available = False
            log.warning("qdrant_unavailable", error=str(e))
            return False

    async def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        if not chunks or not self.available:
            return
        points = [
            qm.PointStruct(
                id=_point_id(c.chunk_id),
                vector=v,
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    **c.metadata,
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        await self._client.upsert(collection_name=self._collection, points=points, wait=True)

    async def delete_by_doc_ids(self, doc_ids: Iterable[str]) -> None:
        ids = list(doc_ids)
        if not ids or not self.available:
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=ids))]
                )
            ),
            wait=True,
        )

    async def search(
        self,
        vector: list[float],
        top_k: int = 20,
        source_types: list[str] | None = None,
        owner_id: str | None = None,
    ) -> list[dict]:
        flt = None
        if source_types:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="source_type", match=qm.MatchAny(any=source_types))]
            )
        if not self.available:
            return []
        fetch_k = top_k * 4 if owner_id else top_k
        try:
            results = await self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=fetch_k,
                query_filter=flt,
                with_payload=True,
            )
        except Exception as e:
            self.available = False
            log.warning("qdrant_search_failed", error=str(e))
            return []
        hits = [{"score": r.score, **(r.payload or {})} for r in results]
        return [h for h in hits if owner_visible(h, owner_id)][:top_k]
