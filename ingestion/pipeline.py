"""Ingestion pipeline orchestrator.

Runs one full ingest pass across a list of connectors:

    For each connector:
      For each record:
        clean -> hash content
          if hash unchanged in registry: mark seen, skip
          else: chunk -> embed (with cache) -> upsert to vector store
                       -> upsert registry
      After all records: stale_doc_ids = registry entries not seen this run
                         -> delete from vector store + registry
    Rebuild BM25 index from the current vector store payloads.

This same function is called by:
  - bootstrap_index.py (initial bulk ingest)
  - workers/poller.py  (every N minutes for real-time freshness)
  - webhook_handler    (single-doc ingest on push events)
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ingestion.connectors.base import BaseConnector, SourceRecord
from ingestion.cleaners import clean
from ingestion.chunkers import chunk_record
from ingestion.embedder import Embedder
from ingestion.registry import Registry, hash_content
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store
from observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class IngestStats:
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks_written: int = 0
    errors: list[str] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        connectors: list[BaseConnector],
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        registry: Registry | None = None,
        bm25_store: BM25Store | None = None,
    ):
        self.connectors = connectors
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.registry = registry or Registry()
        self.bm25_store = bm25_store or BM25Store()

    async def run(self, *, rebuild_bm25: bool = True) -> IngestStats:
        await self.registry.init()
        await self.vector_store.ensure_collection()

        stats = IngestStats()
        run_started_at = datetime.now(timezone.utc)

        for connector in self.connectors:
            log.info("connector_start", source=connector.source_type)
            await self._ingest_one_connector(connector, run_started_at, stats)

            # Sweep deletions for this connector only (don't touch other sources)
            stale = await self.registry.stale_doc_ids(
                source_type=connector.source_type,
                run_started_at=run_started_at,
            )
            if stale:
                await self.vector_store.delete_by_doc_ids(stale)
                await self.registry.delete(stale)
                stats.deleted += len(stale)
                log.info("stale_deleted", source=connector.source_type, count=len(stale))

        if rebuild_bm25:
            await self._rebuild_bm25()

        log.info("ingest_complete", **stats.__dict__)
        return stats

    async def ingest_single(self, record: SourceRecord) -> bool:
        """Ingest one record (used by webhook handler). Returns True if changed."""
        await self.registry.init()
        await self.vector_store.ensure_collection()
        changed = await self._process_record(record, IngestStats())
        if changed:
            await self._rebuild_bm25()
        return changed

    async def _ingest_one_connector(
        self, connector: BaseConnector, run_started_at: datetime, stats: IngestStats
    ) -> None:
        # Bounded concurrency per connector. Keeps memory + CPU in check.
        sem = asyncio.Semaphore(8)
        tasks: list[asyncio.Task] = []

        async def _bounded(rec: SourceRecord):
            async with sem:
                try:
                    await self._process_record(rec, stats)
                except Exception as e:  # never let one bad doc kill the run
                    log.exception("record_failed", doc_id=rec.doc_id, error=str(e))
                    stats.errors.append(f"{rec.doc_id}: {e}")

        async for record in connector.list_records():
            tasks.append(asyncio.create_task(_bounded(record)))
            # Drain in batches of 64 to avoid runaway task creation on huge sources
            if len(tasks) >= 64:
                await asyncio.gather(*tasks)
                tasks.clear()

        if tasks:
            await asyncio.gather(*tasks)

    async def _process_record(self, record: SourceRecord, stats: IngestStats) -> bool:
        cleaned = clean(record)
        content_hash = hash_content(cleaned.content)

        existing = await self.registry.get(record.doc_id)
        if existing and existing.content_hash == content_hash:
            await self.registry.mark_seen(record.doc_id)
            stats.unchanged += 1
            return False

        chunks = chunk_record(cleaned)
        if not chunks:
            log.warning("empty_after_chunking", doc_id=record.doc_id)
            return False

        vectors = await self.embedder.embed_documents([c.text for c in chunks])

        # Replace prior chunks for this doc_id (handles version-N -> version-N+1)
        if existing:
            await self.vector_store.delete_by_doc_ids([record.doc_id])

        await self.vector_store.upsert_chunks(chunks, vectors)
        await self.registry.upsert(
            doc_id=record.doc_id,
            source_type=record.source_type,
            content_hash=content_hash,
            chunk_count=len(chunks),
            url=record.url,
            title=record.title,
        )

        stats.chunks_written += len(chunks)
        if existing:
            stats.updated += 1
        else:
            stats.new += 1
        return True

    async def _rebuild_bm25(self) -> None:
        """Pull all payloads from Qdrant and rebuild the BM25 index."""
        from qdrant_client.http import models as qm
        client = self.vector_store._client  # ok: same package
        collection = self.vector_store._collection

        items: list[tuple[str, str, dict]] = []
        next_offset = None
        while True:
            res, next_offset = await client.scroll(
                collection_name=collection,
                limit=512,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
            for point in res:
                p = point.payload or {}
                items.append((p.get("chunk_id", str(point.id)), p.get("text", ""), p))
            if next_offset is None:
                break

        self.bm25_store.rebuild(items)
        self.bm25_store.save()
