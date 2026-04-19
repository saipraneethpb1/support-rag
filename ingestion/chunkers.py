"""Structure-aware chunker.

Strategy:
  1. Split by markdown headings (H1 -> H2 -> H3) to keep logical sections together.
  2. If a section is still larger than max_chunk_tokens, recursively split on
     paragraphs, then sentences.
  3. Apply a small sliding overlap so context isn't sliced at boundaries.

Tokens are approximated via word count * 1.3. Good enough for budgeting;
the embedding model has its own tokenizer that handles real truncation.
"""
from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable

from config.settings import get_settings
from ingestion.connectors.base import SourceRecord


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


def _approx_tokens(text: str) -> int:
    # Cheap approximation. Real tokenization happens at embed time.
    return int(len(text.split()) * 1.3)


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Return list of (heading_path, section_text)."""
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    cursor = 0
    last_heading_path = ""

    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    for m in matches:
        # Section before this heading belongs to the previous heading
        if m.start() > cursor:
            body = text[cursor:m.start()].strip()
            if body:
                sections.append((last_heading_path, body))

        level = len(m.group(1))
        title = m.group(2).strip()
        heading_stack = heading_stack[: level - 1] + [title]
        last_heading_path = " > ".join(heading_stack)
        cursor = m.end()

    # Trailing content after final heading
    tail = text[cursor:].strip()
    if tail:
        sections.append((last_heading_path, tail))

    return sections


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Recursively split text that exceeds max_tokens, on paragraph then sentence."""
    if _approx_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        out: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for p in paragraphs:
            pt = _approx_tokens(p)
            if buf_tokens + pt > max_tokens and buf:
                out.append("\n\n".join(buf))
                buf, buf_tokens = [], 0
            buf.append(p)
            buf_tokens += pt
        if buf:
            out.append("\n\n".join(buf))
        # Any individual paragraph still oversized? Recurse to sentence split.
        final: list[str] = []
        for piece in out:
            if _approx_tokens(piece) > max_tokens:
                final.extend(_split_sentences(piece, max_tokens))
            else:
                final.append(piece)
        return final

    return _split_sentences(text, max_tokens)


def _split_sentences(text: str, max_tokens: int) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text)
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for s in sentences:
        # A "sentence" with no punctuation may itself exceed max_tokens.
        # Fall back to a hard word-window split in that case.
        if _approx_tokens(s) > max_tokens:
            if buf:
                out.append(" ".join(buf))
                buf, buf_tokens = [], 0
            out.extend(_split_words(s, max_tokens))
            continue
        st = _approx_tokens(s)
        if buf_tokens + st > max_tokens and buf:
            out.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(s)
        buf_tokens += st
    if buf:
        out.append(" ".join(buf))
    return out


def _split_words(text: str, max_tokens: int) -> list[str]:
    """Last-resort hard split on word boundaries."""
    words = text.split()
    n = max(1, int(max_tokens / 1.3))
    return [" ".join(words[i : i + n]) for i in range(0, len(words), n)]


def _apply_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """Prepend a short tail of the previous chunk to each chunk after the first."""
    if overlap_tokens <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, curr in zip(chunks, chunks[1:]):
        words = prev.split()
        # Approx: overlap_tokens / 1.3 words
        n_words = max(1, int(overlap_tokens / 1.3))
        tail = " ".join(words[-n_words:])
        out.append(f"{tail} {curr}".strip())
    return out


def chunk_record(record: SourceRecord) -> list[Chunk]:
    settings = get_settings()
    sections = _split_by_headings(record.content)

    raw_chunks: list[tuple[str, str]] = []  # (heading_path, text)
    for heading_path, section_text in sections:
        for piece in _split_oversized(section_text, settings.max_chunk_tokens):
            if _approx_tokens(piece) >= 5:  # drop ultra-short fragments
                raw_chunks.append((heading_path, piece))

    # Apply overlap on text only (heading path stays per-chunk)
    texts = [t for _, t in raw_chunks]
    texts = _apply_overlap(texts, settings.chunk_overlap_tokens)

    chunks: list[Chunk] = []
    for idx, ((heading_path, _), text) in enumerate(zip(raw_chunks, texts)):
        # Prepend title + heading path to the chunk text. This dramatically
        # improves retrieval because embeddings see context, not just body.
        header = record.title
        if heading_path:
            header = f"{record.title} — {heading_path}"
        embed_text = f"{header}\n\n{text}"

        chunks.append(
            Chunk(
                chunk_id=f"{record.doc_id}::chunk::{idx}",
                doc_id=record.doc_id,
                text=embed_text,
                metadata={
                    "source_type": record.source_type,
                    "source_id": record.source_id,
                    "title": record.title,
                    "heading_path": heading_path,
                    "url": record.url,
                    "updated_at": record.updated_at.isoformat(),
                    "chunk_index": idx,
                    **{f"x_{k}": v for k, v in record.extra_metadata.items() if isinstance(v, (str, int, float, bool))},
                },
            )
        )

    return chunks
