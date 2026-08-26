from datetime import datetime, timezone

import pytest

from ingestion.chunkers import chunk_record
from ingestion.connectors.base import SourceRecord
from ingestion.pipeline import IngestionPipeline
from ingestion.registry import Registry
from retrieval.bm25_store import BM25Store


def _record(name: str, body: str) -> SourceRecord:
    return SourceRecord(
        source_type="markdown_docs",
        source_id=name,
        title=name,
        content=body,
        url=f"uploaded://docs/{name}",
        updated_at=datetime.now(timezone.utc),
        extra_metadata={"uploaded": True},
    )


def test_short_markdown_still_chunks():
    rec = _record("exercise.md", "# Exercise\nDo 10 pushups.")
    chunks = chunk_record(rec)
    assert chunks
    assert any("pushups" in c.text.lower() for c in chunks)


def test_bm25_replace_doc_indexes_text(tmp_path):
    store = BM25Store(tmp_path / "bm25.pkl")
    store.replace_doc(
        "markdown_docs::exercise.md",
        [
            (
                "c1",
                "Exercise: do 10 pushups every morning.",
                {
                    "doc_id": "markdown_docs::exercise.md",
                    "source_type": "markdown_docs",
                    "text": "Exercise: do 10 pushups every morning.",
                },
            )
        ],
    )
    hits = store.search("pushups")
    assert hits
    assert hits[0]["doc_id"] == "markdown_docs::exercise.md"


class _DeadVectors:
    available = False

    async def ensure_collection(self) -> bool:
        self.available = False
        return False

    async def upsert_chunks(self, *args, **kwargs):
        raise AssertionError("Qdrant should not be required")

    async def delete_by_doc_ids(self, *args, **kwargs):
        raise AssertionError("Qdrant should not be required")


class _FailingEmbedder:
    async def embed_documents(self, texts):
        raise RuntimeError("embeddings down")


@pytest.mark.asyncio
async def test_ingest_single_works_without_qdrant(tmp_path, monkeypatch):
    monkeypatch.setenv("REGISTRY_DB_URL", f"sqlite+aiosqlite:///{tmp_path}/registry.db")
    from config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    store = BM25Store(tmp_path / "bm25.pkl")
    pipeline = IngestionPipeline(
        connectors=[],
        embedder=_FailingEmbedder(),
        vector_store=_DeadVectors(),
        registry=Registry(),
        bm25_store=store,
    )
    rec = _record("exercise.md", "# Exercise\nDo ten pushups every morning before breakfast.")
    changed = await pipeline.ingest_single(rec)
    assert changed is True
    hits = store.search("pushups")
    assert hits
    settings_mod.get_settings.cache_clear()
