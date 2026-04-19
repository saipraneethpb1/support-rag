"""Connector base classes and shared data types.

A connector knows how to:
  1. List all source records currently available.
  2. Fetch the raw content + metadata for a record.

Connectors do NOT clean, chunk, or embed — those are pipeline stages.
This separation is critical: it lets us add new sources without touching
the rest of the pipeline.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Literal


SourceType = Literal["markdown_docs", "help_center", "tickets", "changelog", "openapi"]


@dataclass(frozen=True)
class SourceRecord:
    """A unit of content from a source, before parsing/chunking."""
    source_type: SourceType
    source_id: str                    # stable id within the source (e.g. file path, ticket id)
    title: str
    content: str                      # raw content (markdown, html, json string)
    url: str                          # canonical link back to source — used for citations
    updated_at: datetime              # last modification time at source
    extra_metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """Globally unique doc id used as primary key in registry & vector store."""
        return f"{self.source_type}::{self.source_id}"


class BaseConnector(ABC):
    """All source connectors implement this."""
    source_type: SourceType

    @abstractmethod
    async def list_records(self) -> AsyncIterator[SourceRecord]:
        """Yield all currently available records from this source."""
        ...

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)
