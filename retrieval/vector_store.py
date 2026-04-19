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

    async def ensure_collection(self) -> None:
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self._collection in names:
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
        )
        # Indexes for filterable metadata fields
        for field in ("source_type", "doc_id", "url"):
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        log.info("qdrant_collection_created", name=self._collection, dim=self._dim)

    async def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        if not chunks:
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
        if not ids:
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
    ) -> list[dict]:
        flt = None
        if source_types:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="source_type", match=qm.MatchAny(any=source_types))]
            )
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        )
        return [{"score": r.score, **(r.payload or {})} for r in results]
