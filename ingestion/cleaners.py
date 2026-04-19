"""Content cleaners.

Each source type has subtly different junk to remove. We normalize to
plaintext-with-structure-markers (markdown-ish) so the chunker only has
to know one format.
"""
from __future__ import annotations
import re
from bs4 import BeautifulSoup

from ingestion.connectors.base import SourceRecord


_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def _normalize_whitespace(text: str) -> str:
    text = _TRAILING_WS.sub("", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def clean_markdown(content: str) -> str:
    # Strip HTML comments and front-matter
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    content = re.sub(r"^---\n.*?\n---\n", "", content, count=1, flags=re.DOTALL)
    return _normalize_whitespace(content)


def clean_html(content: str) -> str:
    """Convert cleaned HTML to a markdown-ish plaintext that preserves headings."""
    soup = BeautifulSoup(content, "lxml")
    parts: list[str] = []
    for el in soup.descendants:
        name = getattr(el, "name", None)
        if name in {"h1", "h2", "h3", "h4"}:
            level = int(name[1])
            parts.append("\n" + "#" * level + " " + el.get_text(strip=True) + "\n")
        elif name == "li":
            parts.append("- " + el.get_text(" ", strip=True))
        elif name in {"p", "pre"}:
            parts.append(el.get_text(" ", strip=True))
    text = "\n".join(p for p in parts if p)
    return _normalize_whitespace(text)


def clean(record: SourceRecord) -> SourceRecord:
    """Return a new SourceRecord with cleaned content."""
    if record.source_type == "help_center":
        cleaned = clean_html(record.content)
    else:
        cleaned = clean_markdown(record.content)

    return SourceRecord(
        source_type=record.source_type,
        source_id=record.source_id,
        title=record.title,
        content=cleaned,
        url=record.url,
        updated_at=record.updated_at,
        extra_metadata=record.extra_metadata,
    )
