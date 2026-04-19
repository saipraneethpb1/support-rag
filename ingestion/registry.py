"""Document registry.

Tracks every doc we've ingested with its content hash and version.
On each ingest run, we compare incoming hash vs stored hash:
  - hash unchanged  -> skip (no work)
  - hash changed    -> re-chunk + re-embed + replace in vector store
  - new doc         -> chunk + embed + insert
  - missing doc     -> mark for deletion (it disappeared from source)

This is what makes "real-time, incremental, cheap" updates possible
instead of nightly full reindex.
"""
from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import String, DateTime, Integer, select, delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.settings import get_settings


class Base(DeclarativeBase):
    pass


class DocRecord(Base):
    __tablename__ = "doc_registry"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_type: Mapped[str] = mapped_column(String, index=True)
    content_hash: Mapped[str] = mapped_column(String)
    version: Mapped[int] = mapped_column(Integer, default=1)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    url: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Registry:
    def __init__(self) -> None:
        self._engine = create_async_engine(get_settings().registry_db_url, echo=False)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get(self, doc_id: str) -> DocRecord | None:
        async with self._sessionmaker() as s:
            return await s.get(DocRecord, doc_id)

    async def upsert(
        self,
        *,
        doc_id: str,
        source_type: str,
        content_hash: str,
        chunk_count: int,
        url: str,
        title: str,
    ) -> DocRecord:
        now = datetime.now(timezone.utc)
        async with self._sessionmaker() as s:
            existing = await s.get(DocRecord, doc_id)
            if existing is None:
                rec = DocRecord(
                    doc_id=doc_id,
                    source_type=source_type,
                    content_hash=content_hash,
                    version=1,
                    chunk_count=chunk_count,
                    url=url,
                    title=title,
                    last_seen_at=now,
                    last_indexed_at=now,
                )
                s.add(rec)
            else:
                existing.content_hash = content_hash
                existing.version += 1
                existing.chunk_count = chunk_count
                existing.url = url
                existing.title = title
                existing.last_seen_at = now
                existing.last_indexed_at = now
                rec = existing
            await s.commit()
            return rec

    async def mark_seen(self, doc_id: str) -> None:
        async with self._sessionmaker() as s:
            existing = await s.get(DocRecord, doc_id)
            if existing:
                existing.last_seen_at = datetime.now(timezone.utc)
                await s.commit()

    async def stale_doc_ids(self, source_type: str, run_started_at: datetime) -> list[str]:
        """Docs of this source whose last_seen_at predates this ingest run -> deleted at source."""
        async with self._sessionmaker() as s:
            result = await s.execute(
                select(DocRecord.doc_id).where(
                    DocRecord.source_type == source_type,
                    DocRecord.last_seen_at < run_started_at,
                )
            )
            return [row[0] for row in result.all()]

    async def delete(self, doc_ids: Iterable[str]) -> None:
        ids = list(doc_ids)
        if not ids:
            return
        async with self._sessionmaker() as s:
            await s.execute(delete(DocRecord).where(DocRecord.doc_id.in_(ids)))
            await s.commit()
