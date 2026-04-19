"""Chunker tests.

These protect against silent regressions in the most important part of
the ingestion pipeline. Run: pytest tests/unit/test_chunkers.py -v
"""
from __future__ import annotations
from datetime import datetime, timezone

from ingestion.chunkers import chunk_record, _split_by_headings, _approx_tokens
from ingestion.connectors.base import SourceRecord


def _make_record(content: str, title: str = "Test Doc") -> SourceRecord:
    return SourceRecord(
        source_type="markdown_docs",
        source_id="test.md",
        title=title,
        content=content,
        url="https://docs.example.com/test",
        updated_at=datetime.now(timezone.utc),
    )


def test_splits_by_headings():
    text = "# H1\npara1\n\n## H2a\npara2\n\n## H2b\npara3"
    sections = _split_by_headings(text)
    paths = [p for p, _ in sections]
    assert "H1" in paths
    assert "H1 > H2a" in paths
    assert "H1 > H2b" in paths


def test_chunk_metadata_carries_url_and_source():
    rec = _make_record("# Title\n\nSome body text here for the chunk.")
    chunks = chunk_record(rec)
    assert len(chunks) >= 1
    meta = chunks[0].metadata
    assert meta["source_type"] == "markdown_docs"
    assert meta["url"] == "https://docs.example.com/test"
    assert meta["title"] == "Test Doc"


def test_chunk_text_includes_title_and_heading_path():
    rec = _make_record("# Top\n\n## Section A\n\nBody content.")
    chunks = chunk_record(rec)
    # Every chunk should be prefixed with doc title
    assert all("Test Doc" in c.text for c in chunks)


def test_oversized_section_gets_split():
    big_para = " ".join(["word"] * 2000)  # ~2600 approx tokens
    rec = _make_record(f"# Big\n\n{big_para}")
    chunks = chunk_record(rec)
    assert len(chunks) > 1
    # No chunk should wildly exceed the max
    for c in chunks:
        assert _approx_tokens(c.text) <= 1200  # some overlap slack


def test_empty_content_yields_no_chunks_or_very_few():
    rec = _make_record("")
    chunks = chunk_record(rec)
    assert chunks == []


def test_chunk_ids_are_unique_and_deterministic():
    rec = _make_record("# A\n\npara1\n\n## B\n\npara2\n\n## C\n\npara3")
    a = chunk_record(rec)
    b = chunk_record(rec)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
