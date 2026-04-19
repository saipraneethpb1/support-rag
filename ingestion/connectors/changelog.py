"""Changelog / release notes connector.

Reads a single CHANGELOG.md (Keep a Changelog format) and emits one
SourceRecord per release section. This gives per-version citations
instead of one giant doc.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from ingestion.connectors.base import BaseConnector, SourceRecord


_VERSION_HEADING = re.compile(r"^##\s+\[?([^\]\s]+)\]?\s*(?:-\s*(\d{4}-\d{2}-\d{2}))?", re.MULTILINE)


class ChangelogConnector(BaseConnector):
    source_type = "changelog"

    def __init__(self, path: str | Path, base_url: str = "https://docs.example.com/changelog"):
        self.path = Path(path).resolve()
        self.base_url = base_url.rstrip("/")

    async def list_records(self) -> AsyncIterator[SourceRecord]:
        if not self.path.exists():
            return
        text = self.path.read_text(encoding="utf-8")

        matches = list(_VERSION_HEADING.finditer(text))
        if not matches:
            return

        for i, m in enumerate(matches):
            version = m.group(1)
            date_str = m.group(2)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()

            if date_str:
                try:
                    updated_at = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                except ValueError:
                    updated_at = self.now_utc()
            else:
                updated_at = self.now_utc()

            yield SourceRecord(
                source_type="changelog",
                source_id=version,
                title=f"Release {version}",
                content=body,
                url=f"{self.base_url}#{version.lower().replace('.', '-')}",
                updated_at=updated_at,
                extra_metadata={"version": version, "release_date": date_str or ""},
            )
